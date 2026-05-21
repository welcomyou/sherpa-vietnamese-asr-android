package com.asrvn.offline.audio;

import java.io.File;
import java.io.RandomAccessFile;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;

public final class WavPcmDecoder {
    private static final long MAX_IN_MEMORY_PCM_BYTES = 220L * 1024L * 1024L;

    public DecodedAudio decodeIfWav(File file) throws Exception {
        try (RandomAccessFile input = new RandomAccessFile(file, "r")) {
            if (input.length() < 44) return null;
            byte[] riff = new byte[4];
            input.readFully(riff);
            if (riff[0] != 'R' || riff[1] != 'I' || riff[2] != 'F' || riff[3] != 'F') return null;
            readIntLe(input);
            input.readFully(riff);
            if (riff[0] != 'W' || riff[1] != 'A' || riff[2] != 'V' || riff[3] != 'E') return null;

            int channels = 1;
            int sampleRate = 16000;
            int bits = 16;
            int audioFormat = 1;
            long dataOffset = -1;
            long dataSize = 0;
            while (input.getFilePointer() + 8 <= input.length()) {
                input.readFully(riff);
                int size = readIntLe(input);
                String id = new String(riff, java.nio.charset.StandardCharsets.US_ASCII);
                long chunkStart = input.getFilePointer();
                if ("fmt ".equals(id)) {
                    audioFormat = readShortLe(input);
                    channels = readShortLe(input);
                    sampleRate = readIntLe(input);
                    readIntLe(input);
                    readShortLe(input);
                    bits = readShortLe(input);
                } else if ("data".equals(id)) {
                    dataOffset = chunkStart;
                    dataSize = Integer.toUnsignedLong(size);
                    break;
                }
                input.seek(chunkStart + Integer.toUnsignedLong(size) + (size & 1));
            }
            if (dataOffset < 0 || audioFormat != 1 || bits != 16) return null;
            if (sampleRate != 16000) return null;
            if (dataSize > MAX_IN_MEMORY_PCM_BYTES) return null;
            int frameBytes = Math.max(1, channels * 2);
            long framesLong = dataSize / frameBytes;
            if (framesLong > Integer.MAX_VALUE) {
                throw new IllegalStateException("Decoded WAV is too large for this Android runtime: "
                        + framesLong + " samples");
            }
            float[] mono = new float[(int) framesLong];
            input.seek(dataOffset);
            readPcm16Mono(input, mono, channels, dataSize);
            return new DecodedAudio(mono, 16000, 1, "wav-pcm16-16k-direct");
        }
    }

    private static void readPcm16Mono(RandomAccessFile input, float[] mono, int channels, long dataSize) throws Exception {
        int frameBytes = Math.max(1, channels * 2);
        int bufferBytes = Math.max(frameBytes, (64 * 1024 / frameBytes) * frameBytes);
        byte[] raw = new byte[bufferBytes];
        int frameIndex = 0;
        long remaining = dataSize - (dataSize % frameBytes);
        while (remaining > 0 && frameIndex < mono.length) {
            int toRead = (int) Math.min(raw.length, remaining);
            toRead -= toRead % frameBytes;
            input.readFully(raw, 0, toRead);
            ByteBuffer buffer = ByteBuffer.wrap(raw, 0, toRead).order(ByteOrder.LITTLE_ENDIAN);
            for (int p = 0; p + frameBytes <= toRead && frameIndex < mono.length; p += frameBytes) {
                int sum = 0;
                for (int c = 0; c < channels; c++) {
                    sum += buffer.getShort(p + c * 2);
                }
                mono[frameIndex++] = (sum / (float) channels) / 32768.0f;
            }
            remaining -= toRead;
        }
    }

    private static int readShortLe(RandomAccessFile input) throws Exception {
        int b0 = input.read();
        int b1 = input.read();
        if ((b0 | b1) < 0) throw new java.io.EOFException();
        return (b0 & 0xff) | ((b1 & 0xff) << 8);
    }

    private static int readIntLe(RandomAccessFile input) throws Exception {
        int b0 = input.read();
        int b1 = input.read();
        int b2 = input.read();
        int b3 = input.read();
        if ((b0 | b1 | b2 | b3) < 0) throw new java.io.EOFException();
        return (b0 & 0xff) | ((b1 & 0xff) << 8) | ((b2 & 0xff) << 16) | ((b3 & 0xff) << 24);
    }
}
