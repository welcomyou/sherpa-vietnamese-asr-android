package com.asrvn.offline.audio;

import java.io.File;

public final class DecodedAudio {
    public final float[] samples;
    public final File pcmFile;
    public final int sampleCount;
    public final int sampleRate;
    public final int channels;
    public final String decoder;

    public DecodedAudio(float[] samples, int sampleRate, int channels, String decoder) {
        this(samples, null, samples == null ? 0 : samples.length, sampleRate, channels, decoder);
    }

    public DecodedAudio(File pcmFile, int sampleCount, int sampleRate, int channels, String decoder) {
        this(null, pcmFile, sampleCount, sampleRate, channels, decoder);
    }

    private DecodedAudio(float[] samples, File pcmFile, int sampleCount, int sampleRate, int channels, String decoder) {
        this.samples = samples;
        this.pcmFile = pcmFile;
        this.sampleCount = sampleCount;
        this.sampleRate = sampleRate;
        this.channels = channels;
        this.decoder = decoder;
    }

    public boolean inMemory() {
        return samples != null;
    }
}
