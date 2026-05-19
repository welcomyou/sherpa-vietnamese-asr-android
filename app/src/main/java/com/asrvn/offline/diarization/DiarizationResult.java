package com.asrvn.offline.diarization;

import java.util.List;

public final class DiarizationResult {
    public static final class Turn {
        public final double start;
        public final double end;
        public final int speaker;

        public Turn(double start, double end, int speaker) {
            this.start = start;
            this.end = end;
            this.speaker = speaker;
        }
    }

    public final List<Turn> turns;
    public final String backend;
    public final String provider;

    public DiarizationResult(List<Turn> turns, String backend, String provider) {
        this.turns = turns;
        this.backend = backend;
        this.provider = provider;
    }
}
