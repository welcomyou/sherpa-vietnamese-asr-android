package com.asrvn.offline.audio;

import java.io.File;

public final class CanonicalAudioDecoder {
    private final WavPcmDecoder wav = new WavPcmDecoder();
    private final FfmpegAudioDecoder ffmpeg = new FfmpegAudioDecoder();

    public DecodedAudio decode(File file) throws Exception {
        DecodedAudio direct = wav.decodeIfWav(file);
        if (direct != null) return direct;
        return ffmpeg.decode(file);
    }
}
