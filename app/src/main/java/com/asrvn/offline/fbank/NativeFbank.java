package com.asrvn.offline.fbank;

public final class NativeFbank {
    public static final class Result {
        public final float[] data;
        public final int frames;

        public Result(float[] data, int frames) {
            this.data = data;
            this.frames = frames;
        }
    }

    static {
        System.loadLibrary("asrvn_fbank");
    }

    public static native Result computeAsr(float[] samples);

    public static native Result computeCampp(float[] samples);

    public static native Result computeWespeaker(float[] samples);

    private NativeFbank() {
    }
}
