package com.asrvn.offline.audio;

import java.io.File;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.file.Files;

public final class WavPcmDecoder {
    public DecodedAudio decodeIfWav(File file) throws Exception {
        byte[] data = Files.readAllBytes(file.toPath());
        if (data.length < 44 || data[0] != 'R' || data[1] != 'I' || data[2] != 'F' || data[3] != 'F') {
            return null;
        }
        ByteBuffer buffer = ByteBuffer.wrap(data).order(ByteOrder.LITTLE_ENDIAN);
        int channels = 1;
        int sampleRate = 16000;
        int bits = 16;
        int audioFormat = 1;
        int dataOffset = -1;
        int dataSize = 0;
        int pos = 12;
        while (pos + 8 <= data.length) {
            String id = new String(data, pos, 4, java.nio.charset.StandardCharsets.US_ASCII);
            int size = buffer.getInt(pos + 4);
            if ("fmt ".equals(id)) {
                audioFormat = buffer.getShort(pos + 8) & 0xffff;
                channels = buffer.getShort(pos + 10) & 0xffff;
                sampleRate = buffer.getInt(pos + 12);
                bits = buffer.getShort(pos + 22) & 0xffff;
            } else if ("data".equals(id)) {
                dataOffset = pos + 8;
                dataSize = size;
                break;
            }
            pos += 8 + size + (size & 1);
        }
        if (dataOffset < 0 || audioFormat != 1 || bits != 16) return null;
        if (sampleRate != 16000) return null;
        int frames = dataSize / Math.max(1, channels * 2);
        float[] mono = new float[frames];
        int p = dataOffset;
        for (int i = 0; i < frames; i++) {
            int sum = 0;
            for (int c = 0; c < channels; c++) {
                sum += buffer.getShort(p);
                p += 2;
            }
            mono[i] = (sum / (float) channels) / 32768.0f;
        }
        return new DecodedAudio(mono, 16000, 1, "wav-pcm16-16k-direct");
    }
}
