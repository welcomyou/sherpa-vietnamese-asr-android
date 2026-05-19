package com.asrvn.offline.diarization;

import com.asrvn.offline.fbank.NativeFbank;

public final class CamppFbankExtractor {
    public static final class Features {
        public final float[] data;
        public final int frames;

        Features(float[] data, int frames) {
            this.data = data;
            this.frames = frames;
        }
    }

    private static final int SAMPLE_RATE = 16000;
    private static final int FRAME_LENGTH = 400;
    private static final int FRAME_SHIFT = 160;
    private static final int N_FFT = 512;
    private static final int NUM_MEL_BINS = 80;
    private static final double LOW_FREQ = 20.0;
    private static final double PREEMPHASIS = 0.97;
    private static final double ENERGY_FLOOR = 1.0;

    private float[] window;
    private float[][] melBank;
    private FftTables fftTables;

    public Features compute(float[] samples) {
        NativeFbank.Result result = NativeFbank.computeCampp(samples);
        return new Features(result.data, result.frames);
    }

    public Features computeJavaFallback(float[] samples) {
        int n = samples == null ? 0 : samples.length;
        if (n < FRAME_LENGTH) return new Features(new float[0], 0);
        int frames = 1 + (n - FRAME_LENGTH) / FRAME_SHIFT;
        float[] features = new float[frames * NUM_MEL_BINS];
        double[] means = new double[NUM_MEL_BINS];
        float[] localWindow = window();
        float[][] localMelBank = melBank();
        double[] frame = new double[FRAME_LENGTH];
        double[] real = new double[N_FFT];
        double[] imag = new double[N_FFT];
        double[] power = new double[N_FFT / 2 + 1];

        for (int f = 0; f < frames; f++) {
            int start = f * FRAME_SHIFT;
            double mean = 0.0;
            for (int i = 0; i < FRAME_LENGTH; i++) {
                mean += samples[start + i] * 32768.0;
            }
            mean /= FRAME_LENGTH;
            for (int i = 0; i < FRAME_LENGTH; i++) {
                frame[i] = samples[start + i] * 32768.0 - mean;
            }
            fill(real, 0.0);
            fill(imag, 0.0);
            double context = start > 0 ? samples[start - 1] * 32768.0 : 0.0;
            real[0] = (frame[0] - PREEMPHASIS * context) * localWindow[0];
            for (int i = 1; i < FRAME_LENGTH; i++) {
                real[i] = (frame[i] - PREEMPHASIS * frame[i - 1]) * localWindow[i];
            }

            fftInPlace(real, imag);
            for (int i = 0; i < power.length; i++) {
                power[i] = real[i] * real[i] + imag[i] * imag[i];
            }
            int out = f * NUM_MEL_BINS;
            for (int mel = 0; mel < NUM_MEL_BINS; mel++) {
                float[] weights = localMelBank[mel];
                double energy = 0.0;
                for (int bin = 0; bin < weights.length; bin++) energy += power[bin] * weights[bin];
                float value = (float) Math.log(Math.max(energy, ENERGY_FLOOR));
                features[out + mel] = value;
                means[mel] += value;
            }
        }

        for (int mel = 0; mel < NUM_MEL_BINS; mel++) means[mel] /= frames;
        for (int f = 0; f < frames; f++) {
            int out = f * NUM_MEL_BINS;
            for (int mel = 0; mel < NUM_MEL_BINS; mel++) features[out + mel] -= (float) means[mel];
        }
        return new Features(features, frames);
    }

    private float[] window() {
        if (window != null) return window;
        window = new float[FRAME_LENGTH];
        for (int i = 0; i < window.length; i++) {
            double hann = 0.5 - 0.5 * Math.cos((2.0 * Math.PI * i) / (FRAME_LENGTH - 1));
            window[i] = (float) Math.pow(hann, 0.85);
        }
        return window;
    }

    private float[][] melBank() {
        if (melBank != null) return melBank;
        double lowMel = hzToMel(LOW_FREQ);
        double highMel = hzToMel(SAMPLE_RATE / 2.0);
        double melDelta = (highMel - lowMel) / (NUM_MEL_BINS + 1);
        double[] centers = new double[NUM_MEL_BINS + 2];
        for (int i = 0; i < centers.length; i++) centers[i] = melToHz(lowMel + i * melDelta);
        int bins = N_FFT / 2 + 1;
        melBank = new float[NUM_MEL_BINS][bins];
        for (int bin = 0; bin < bins; bin++) {
            double freq = (bin * SAMPLE_RATE) / (double) N_FFT;
            for (int mel = 0; mel < NUM_MEL_BINS; mel++) {
                double left = centers[mel];
                double center = centers[mel + 1];
                double right = centers[mel + 2];
                double weight = 0.0;
                if (freq > left && freq <= center) weight = (freq - left) / Math.max(center - left, 1e-12);
                else if (freq > center && freq < right) weight = (right - freq) / Math.max(right - center, 1e-12);
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

    private void fftInPlace(double[] real, double[] imag) {
        int n = real.length;
        FftTables tables = fftTables(n);
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

    private FftTables fftTables(int size) {
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
