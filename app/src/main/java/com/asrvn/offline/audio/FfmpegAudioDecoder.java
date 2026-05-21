package com.asrvn.offline.audio;

import com.arthenica.ffmpegkit.FFmpegKit;
import com.arthenica.ffmpegkit.FFmpegSession;
import com.arthenica.ffmpegkit.ReturnCode;

import java.io.File;
import java.nio.file.Files;
import java.util.Locale;

public final class FfmpegAudioDecoder {
    private static final int TARGET_SAMPLE_RATE = 16000;
    private static final String DESKTOP_RESAMPLE_FILTER = "aresample=resampler=soxr:precision=20";
    private static final long MAX_IN_MEMORY_PCM_BYTES = 220L * 1024L * 1024L;

    public DecodedAudio decode(File file) throws Exception {
        File tempDir = tempDirectoryFor(file);
        deleteStaleDecodeFiles(tempDir);
        File output = File.createTempFile("asrvn_decode_", ".f32", tempDir);
        boolean keepOutput = false;
        try {
            String[] args = new String[] {
                    "-y",
                    "-hide_banner",
                    "-nostdin",
                    "-i", file.getAbsolutePath(),
                    "-vn",
                    "-af", DESKTOP_RESAMPLE_FILTER,
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

            long byteCount = output.length();
            if ((byteCount & 3L) != 0L) {
                throw new IllegalStateException("FFmpeg returned non-float32 PCM byte count: " + byteCount);
            }
            long sampleCountLong = byteCount / 4L;
            if (sampleCountLong > Integer.MAX_VALUE) {
                throw new IllegalStateException("Decoded audio is too large for this Android runtime: "
                        + sampleCountLong + " samples");
            }
            int sampleCount = (int) sampleCountLong;
            String decoder = String.format(Locale.US, "ffmpeg-soxr-f32le:%dHz", TARGET_SAMPLE_RATE);
            if (byteCount > MAX_IN_MEMORY_PCM_BYTES) {
                keepOutput = true;
                return new DecodedAudio(output, sampleCount, TARGET_SAMPLE_RATE, 1, decoder + ":disk");
            }
            float[] samples = PcmFloatFile.readAll(output, sampleCount);
            return new DecodedAudio(samples, TARGET_SAMPLE_RATE, 1, decoder);
        } finally {
            if (!keepOutput) {
                try {
                    Files.deleteIfExists(output.toPath());
                } catch (Exception ignored) {
                }
            }
        }
    }

    private static File tempDirectoryFor(File input) {
        File parent = input.getParentFile();
        if (parent != null && parent.isDirectory() && parent.canWrite()) return parent;
        return null;
    }

    private static void deleteStaleDecodeFiles(File directory) {
        if (directory == null || !directory.isDirectory()) return;
        File[] files = directory.listFiles((dir, name) -> name.startsWith("asrvn_decode_") && name.endsWith(".f32"));
        if (files == null) return;
        for (File file : files) {
            try {
                Files.deleteIfExists(file.toPath());
            } catch (Exception ignored) {
            }
        }
    }

    private static String abbreviate(String value, int maxChars) {
        if (value == null) return "";
        if (value.length() <= maxChars) return value;
        return value.substring(0, maxChars) + "...";
    }
}
