package com.asrvn.offline.audio;

import com.arthenica.ffmpegkit.FFmpegKit;
import com.arthenica.ffmpegkit.FFmpegSession;
import com.arthenica.ffmpegkit.ReturnCode;

import java.io.File;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.FloatBuffer;
import java.nio.file.Files;
import java.util.Locale;

public final class FfmpegAudioDecoder {
    private static final int TARGET_SAMPLE_RATE = 16000;

    public DecodedAudio decode(File file) throws Exception {
        File output = File.createTempFile("asrvn_decode_", ".f32", tempDirectoryFor(file));
        try {
            String[] args = new String[] {
                    "-y",
                    "-hide_banner",
                    "-nostdin",
                    "-i", file.getAbsolutePath(),
                    "-vn",
                    "-ac", "1",
                    "-ar", String.valueOf(TARGET_SAMPLE_RATE),
                    "-f", "f32le",
                    "-acodec", "pcm_f32le",
                    "-loglevel", "error",
                    output.getAbsolutePath()
            };
            FFmpegSession session = FFmpegKit.executeWithArguments(args);
            ReturnCode code = session.getReturnCode();
            if (!ReturnCode.isSuccess(code)) {
                String logs = session.getAllLogsAsString();
                String detail = logs == null || logs.isBlank() ? session.getFailStackTrace() : logs;
                throw new IllegalStateException("Canonical FFmpeg audio decode failed: "
                        + abbreviate(detail, 1200));
            }

            byte[] raw = Files.readAllBytes(output.toPath());
            if ((raw.length & 3) != 0) {
                throw new IllegalStateException("FFmpeg returned non-float32 PCM byte count: " + raw.length);
            }
            FloatBuffer floats = ByteBuffer.wrap(raw)
                    .order(ByteOrder.LITTLE_ENDIAN)
                    .asFloatBuffer();
            float[] samples = new float[floats.remaining()];
            floats.get(samples);
            return new DecodedAudio(samples, TARGET_SAMPLE_RATE, 1,
                    String.format(Locale.US, "ffmpeg-default:%dHz", TARGET_SAMPLE_RATE));
        } finally {
            try {
                Files.deleteIfExists(output.toPath());
            } catch (Exception ignored) {
            }
        }
    }

    private static File tempDirectoryFor(File input) {
        File parent = input.getParentFile();
        if (parent != null && parent.isDirectory() && parent.canWrite()) return parent;
        return null;
    }

    private static String abbreviate(String value, int maxChars) {
        if (value == null) return "";
        if (value.length() <= maxChars) return value;
        return value.substring(0, maxChars) + "...";
    }
}
