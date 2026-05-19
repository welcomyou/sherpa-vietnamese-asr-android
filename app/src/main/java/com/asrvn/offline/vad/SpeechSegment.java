package com.asrvn.offline.vad;

public final class SpeechSegment {
    public final int start;
    public final int end;

    public SpeechSegment(int start, int end) {
        this.start = start;
        this.end = end;
    }
}
