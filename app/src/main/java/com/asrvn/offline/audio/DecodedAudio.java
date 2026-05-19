package com.asrvn.offline.audio;

public final class DecodedAudio {
    public final float[] samples;
    public final int sampleRate;
    public final int channels;
    public final String decoder;

    public DecodedAudio(float[] samples, int sampleRate, int channels, String decoder) {
        this.samples = samples;
        this.sampleRate = sampleRate;
        this.channels = channels;
        this.decoder = decoder;
    }
}
