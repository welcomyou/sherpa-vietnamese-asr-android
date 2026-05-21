package com.asrvn.offline.audio;

import java.io.BufferedInputStream;
import java.io.DataInputStream;
import java.io.File;
import java.io.FileInputStream;
import java.io.RandomAccessFile;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;

public final class PcmFloatFile {
    private static final int BUFFER_BYTES = 1024 * 1024;

    private PcmFloatFile() {
    }

    public static float[] read(File file, int startSample, int sampleCount) throws Exception {
        float[] out = new float[Math.max(0, sampleCount)];
        readInto(file, startSample, out, 0, out.length);
        return out;
    }

    public static void readInto(File file, int startSample, float[] out, int outOffset, int sampleCount) throws Exception {
        if (sampleCount <= 0) return;
        try (RandomAccessFile input = new RandomAccessFile(file, "r")) {
            input.seek(Math.max(0L, startSample) * 4L);
            byte[] raw = new byte[(int) Math.min(BUFFER_BYTES, Math.max(4L, sampleCount * 4L))];
            int remaining = sampleCount;
            int write = outOffset;
            while (remaining > 0) {
                int samples = Math.min(remaining, raw.length / 4);
                int bytes = samples * 4;
                input.readFully(raw, 0, bytes);
                ByteBuffer buffer = ByteBuffer.wrap(raw, 0, bytes).order(ByteOrder.LITTLE_ENDIAN);
                for (int i = 0; i < samples; i++) out[write++] = buffer.getFloat();
                remaining -= samples;
            }
        }
    }

    public static float[] readAll(File file, int sampleCount) throws Exception {
        long byteCount = file.length();
        if ((byteCount & 3L) != 0L) {
            throw new IllegalStateException("PCM float32 byte count is not divisible by 4: " + byteCount);
        }
        if (sampleCount < 0 || sampleCount > Integer.MAX_VALUE) {
            throw new IllegalStateException("PCM sample count is too large: " + sampleCount);
        }
        float[] samples = new float[sampleCount];
        byte[] raw = new byte[BUFFER_BYTES];
        int sampleOffset = 0;
        long remainingBytes = Math.min(byteCount, sampleCount * 4L);
        try (DataInputStream input = new DataInputStream(new BufferedInputStream(new FileInputStream(file), BUFFER_BYTES))) {
            while (remainingBytes > 0) {
                int read = (int) Math.min(raw.length, remainingBytes);
                input.readFully(raw, 0, read);
                ByteBuffer buffer = ByteBuffer.wrap(raw, 0, read).order(ByteOrder.LITTLE_ENDIAN);
                int count = read / 4;
                for (int i = 0; i < count; i++) samples[sampleOffset++] = buffer.getFloat();
                remainingBytes -= read;
            }
        }
        return samples;
    }
}
