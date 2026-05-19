package com.asrvn.offline.diarization;

import com.asrvn.offline.fbank.NativeFbank;

public final class WespeakerFbankExtractor {
    static final class Features {
        final float[] data;
        final int frames;

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
    private static final int LOW_FREQ = 20;
    private static final double PREEMPHASIS = 0.97;
    private static final double MEL_ENERGY_FLOOR = 1.1920928955078125e-7;

    private final float[] window = new float[FRAME_LENGTH];
    private final float[][] melBank = new float[NUM_MEL_BINS][N_FFT / 2 + 1];

    WespeakerFbankExtractor() {
        for (int i = 0; i < FRAME_LENGTH; i++) {
            window[i] = (float) (0.54 - 0.46 * Math.cos((2.0 * Math.PI * i) / (FRAME_LENGTH - 1)));
        }
        double lowMel = hzToMel(LOW_FREQ);
        double highMel = hzToMel(SAMPLE_RATE / 2.0);
        double delta = (highMel - lowMel) / (NUM_MEL_BINS + 1);
        double[] centers = new double[NUM_MEL_BINS + 2];
        for (int i = 0; i < centers.length; i++) centers[i] = melToHz(lowMel + i * delta);
        for (int bin = 0; bin <= N_FFT / 2; bin++) {
            double freq = (bin * SAMPLE_RATE) / (double) N_FFT;
            for (int mel = 0; mel < NUM_MEL_BINS; mel++) {
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
    }

    Features compute(float[] audio) {
        NativeFbank.Result result = NativeFbank.computeWespeaker(audio);
        return new Features(result.data, result.frames);
    }

    Features computeJavaFallback(float[] audio) {
        if (audio.length < FRAME_LENGTH) return new Features(new float[0], 0);
        int frames = 1 + (audio.length - FRAME_LENGTH) / FRAME_SHIFT;
        float[] out = new float[frames * NUM_MEL_BINS];
        double[] means = new double[NUM_MEL_BINS];
        float[] frame = new float[FRAME_LENGTH];
        double[] real = new double[N_FFT];
        double[] imag = new double[N_FFT];
        double[] power = new double[N_FFT / 2 + 1];
        for (int f = 0; f < frames; f++) {
            int audioStart = f * FRAME_SHIFT;
            double mean = 0.0;
            for (int i = 0; i < FRAME_LENGTH; i++) mean += audio[audioStart + i] * 32768.0;
            mean /= FRAME_LENGTH;
            for (int i = 0; i < FRAME_LENGTH; i++) frame[i] = (float) (audio[audioStart + i] * 32768.0 - mean);
            java.util.Arrays.fill(real, 0.0);
            java.util.Arrays.fill(imag, 0.0);
            real[0] = (frame[0] - PREEMPHASIS * frame[0]) * window[0];
            for (int i = 1; i < FRAME_LENGTH; i++) real[i] = (frame[i] - PREEMPHASIS * frame[i - 1]) * window[i];
            fft(real, imag);
            for (int i = 0; i < power.length; i++) power[i] = real[i] * real[i] + imag[i] * imag[i];
            int row = f * NUM_MEL_BINS;
            for (int mel = 0; mel < NUM_MEL_BINS; mel++) {
                double energy = 0.0;
                for (int bin = 0; bin < power.length; bin++) energy += power[bin] * melBank[mel][bin];
                float value = (float) Math.log(Math.max(energy, MEL_ENERGY_FLOOR));
                out[row + mel] = value;
                means[mel] += value;
            }
        }
        for (int mel = 0; mel < NUM_MEL_BINS; mel++) means[mel] /= frames;
        for (int f = 0; f < frames; f++) {
            int row = f * NUM_MEL_BINS;
            for (int mel = 0; mel < NUM_MEL_BINS; mel++) out[row + mel] -= (float) means[mel];
        }
        return new Features(out, frames);
    }

    private static double hzToMel(double freq) {
        return 1127.0 * Math.log(1.0 + freq / 700.0);
    }

    private static double melToHz(double mel) {
        return 700.0 * (Math.exp(mel / 1127.0) - 1.0);
    }

    private static void fft(double[] real, double[] imag) {
        int n = real.length;
        for (int i = 1, j = 0; i < n; i++) {
            int bit = n >> 1;
            for (; (j & bit) != 0; bit >>= 1) j ^= bit;
            j ^= bit;
            if (i < j) {
                double tr = real[i]; real[i] = real[j]; real[j] = tr;
                double ti = imag[i]; imag[i] = imag[j]; imag[j] = ti;
            }
        }
        for (int len = 2; len <= n; len <<= 1) {
            double angle = -2.0 * Math.PI / len;
            double wlenR = Math.cos(angle);
            double wlenI = Math.sin(angle);
            for (int i = 0; i < n; i += len) {
                double wr = 1.0;
                double wi = 0.0;
                for (int j = 0; j < len / 2; j++) {
                    int u = i + j;
                    int v = i + j + len / 2;
                    double vr = real[v] * wr - imag[v] * wi;
                    double vi = real[v] * wi + imag[v] * wr;
                    real[v] = real[u] - vr;
                    imag[v] = imag[u] - vi;
                    real[u] += vr;
                    imag[u] += vi;
                    double nextR = wr * wlenR - wi * wlenI;
                    wi = wr * wlenI + wi * wlenR;
                    wr = nextR;
                }
            }
        }
    }
}
