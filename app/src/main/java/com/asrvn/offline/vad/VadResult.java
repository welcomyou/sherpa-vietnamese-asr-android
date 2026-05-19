package com.asrvn.offline.vad;

import java.util.List;

public final class VadResult {
    public final List<SpeechSegment> segments;
    public final float[] probabilities;
    public final boolean bypassed;

    public VadResult(List<SpeechSegment> segments, float[] probabilities, boolean bypassed) {
        this.segments = segments;
        this.probabilities = probabilities;
        this.bypassed = bypassed;
    }
}
