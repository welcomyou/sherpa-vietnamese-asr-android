package com.asrvn.offline.asr;

public final class AsrConstants {
    public static final int SAMPLE_RATE = 16000;
    public static final int FRAME_LENGTH = 400;
    public static final int FRAME_SHIFT = 160;
    public static final int N_FFT = 512;
    public static final int NUM_MEL_BINS = 80;
    public static final double LOW_FREQ = 20.0;
    public static final double HIGH_FREQ = 7600.0;
    public static final double PREEMPHASIS = 0.97;
    public static final double LOG_FLOOR = 1.1920928955078125e-7;
    public static final int BLANK_ID = 0;
    public static final int UNK_ID = 2;
    public static final int CONTEXT_SIZE = 2;

    private AsrConstants() {
    }
}
