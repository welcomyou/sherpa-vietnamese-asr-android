package com.asrvn.offline.asr;

import com.asrvn.offline.fbank.NativeFbank;

public final class FbankExtractor {
    public static final class Features {
        public final float[] data;
        public final int frames;

        public Features(float[] data, int frames) {
            this.data = data;
            this.frames = frames;
        }
    }

    private float[] window;
    private float[][] melBank;
    private FftTables fftTables;

    public Features compute(float[] samples) {
        NativeFbank.Result result = NativeFbank.computeAsr(samples);
        return new Features(result.data, result.frames);
    }

    public Features computeJavaFallback(float[] samples) {
        int n = samples == null ? 0 : samples.length;
        if (n <= 0) return new Features(new float[0], 0);
        int numFrames = (int) Math.floor((n + AsrConstants.FRAME_SHIFT / 2.0) / AsrConstants.FRAME_SHIFT);
        if (numFrames <= 0) return new Features(new float[0], 0);

        float[] localWindow = getWindow();
        float[][] localMelBank = getMelBank();
        double[] frame = new double[AsrConstants.FRAME_LENGTH];
        double[] real = new double[AsrConstants.N_FFT];
        double[] imag = new double[AsrConstants.N_FFT];
        double[] power = new double[AsrConstants.N_FFT / 2 + 1];
        float[] features = new float[numFrames * AsrConstants.NUM_MEL_BINS];

        for (int f = 0; f < numFrames; f++) {
            int start = f * AsrConstants.FRAME_SHIFT
                    + AsrConstants.FRAME_SHIFT / 2
                    - AsrConstants.FRAME_LENGTH / 2;
            double mean = 0.0;
            for (int i = 0; i < AsrConstants.FRAME_LENGTH; i++) {
                double sample = samples[reflectIndex(start + i, n)];
                frame[i] = sample;
                mean += sample;
            }
            mean /= AsrConstants.FRAME_LENGTH;

            fill(real, 0.0);
            fill(imag, 0.0);
            for (int i = 0; i < AsrConstants.FRAME_LENGTH; i++) {
                double current = frame[i] - mean;
                double previous = i == 0 ? frame[0] - mean : frame[i - 1] - mean;
                real[i] = (current - AsrConstants.PREEMPHASIS * previous) * localWindow[i];
            }

            fftInPlace(real, imag);
            for (int i = 0; i < power.length; i++) {
                power[i] = real[i] * real[i] + imag[i] * imag[i];
            }

            int outOffset = f * AsrConstants.NUM_MEL_BINS;
            for (int mel = 0; mel < AsrConstants.NUM_MEL_BINS; mel++) {
                float[] weights = localMelBank[mel];
                double energy = 0.0;
                for (int bin = 0; bin < weights.length; bin++) {
                    energy += power[bin] * weights[bin];
                }
                features[outOffset + mel] = (float) Math.log(Math.max(energy, AsrConstants.LOG_FLOOR));
            }
        }

        return new Features(features, numFrames);
    }

    private float[] getWindow() {
        if (window != null) return window;
        window = new float[AsrConstants.FRAME_LENGTH];
        for (int i = 0; i < window.length; i++) {
            double hann = 0.5 - 0.5 * Math.cos((2.0 * Math.PI * i) / (AsrConstants.FRAME_LENGTH - 1));
            window[i] = (float) Math.pow(hann, 0.85);
        }
        return window;
    }

    private float[][] getMelBank() {
        if (melBank != null) return melBank;
        double lowMel = hzToMel(AsrConstants.LOW_FREQ);
        double highMel = hzToMel(AsrConstants.HIGH_FREQ);
        double melDelta = (highMel - lowMel) / (AsrConstants.NUM_MEL_BINS + 1);
        double[] centers = new double[AsrConstants.NUM_MEL_BINS + 2];
        for (int i = 0; i < centers.length; i++) {
            centers[i] = melToHz(lowMel + i * melDelta);
        }

        int bins = AsrConstants.N_FFT / 2 + 1;
        melBank = new float[AsrConstants.NUM_MEL_BINS][bins];
        for (int bin = 0; bin < bins; bin++) {
            double freq = (bin * AsrConstants.SAMPLE_RATE) / (double) AsrConstants.N_FFT;
            for (int mel = 0; mel < AsrConstants.NUM_MEL_BINS; mel++) {
                double left = centers[mel];
                double center = centers[mel + 1];
                double right = centers[mel + 2];
                double weight = 0.0;
                if (freq > left && freq <= center) {
                    weight = (freq - left) / Math.max(center - left, 1e-12);
                } else if (freq > center && freq < right) {
                    weight = (right - freq) / Math.max(right - center, 1e-12);
                }
                melBank[mel][bin] = (float) weight;
            }
        }
        return melBank;
    }

    private double hzToMel(double freq) {
        return 1127.0 * Math.log(1.0 + freq / 700.0);
    }

    private double melToHz(double mel) {
        return 700.0 * (Math.exp(mel / 1127.0) - 1.0);
    }

    private int reflectIndex(int index, int length) {
        if (length <= 1) return 0;
        int value = index;
        while (value < 0 || value >= length) {
            if (value < 0) value = -value - 1;
            else value = 2 * length - 1 - value;
        }
        return value;
    }

    private void fftInPlace(double[] real, double[] imag) {
        int n = real.length;
        FftTables tables = getFftTables(n);
        for (int i = 0; i < n; i++) {
            int j = tables.reverse[i];
            if (j <= i) continue;
            double tr = real[i];
            double ti = imag[i];
            real[i] = real[j];
            imag[i] = imag[j];
            real[j] = tr;
            imag[j] = ti;
        }

        for (int size = 2; size <= n; size <<= 1) {
            int half = size >> 1;
            int tableStep = n / size;
            for (int start = 0; start < n; start += size) {
                for (int j = 0; j < half; j++) {
                    int k = j * tableStep;
                    double wr = tables.cos[k];
                    double wi = tables.sin[k];
                    int even = start + j;
                    int odd = even + half;
                    double tr = wr * real[odd] - wi * imag[odd];
                    double ti = wr * imag[odd] + wi * real[odd];
                    real[odd] = real[even] - tr;
                    imag[odd] = imag[even] - ti;
                    real[even] += tr;
                    imag[even] += ti;
                }
            }
        }
    }

    private FftTables getFftTables(int size) {
        if (fftTables != null && fftTables.size == size) return fftTables;
        int levels = (int) (Math.log(size) / Math.log(2.0));
        int[] reverse = new int[size];
        for (int i = 0; i < size; i++) {
            int value = i;
            int result = 0;
            for (int bit = 0; bit < levels; bit++) {
                result = (result << 1) | (value & 1);
                value >>= 1;
            }
            reverse[i] = result;
        }
        double[] cos = new double[size / 2];
        double[] sin = new double[size / 2];
        for (int i = 0; i < size / 2; i++) {
            double angle = (-2.0 * Math.PI * i) / size;
            cos[i] = Math.cos(angle);
            sin[i] = Math.sin(angle);
        }
        fftTables = new FftTables(size, reverse, cos, sin);
        return fftTables;
    }

    private void fill(double[] values, double value) {
        for (int i = 0; i < values.length; i++) values[i] = value;
    }

    private static final class FftTables {
        final int size;
        final int[] reverse;
        final double[] cos;
        final double[] sin;

        FftTables(int size, int[] reverse, double[] cos, double[] sin) {
            this.size = size;
            this.reverse = reverse;
            this.cos = cos;
            this.sin = sin;
        }
    }
}
