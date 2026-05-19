package com.asrvn.offline.audio;

import android.media.AudioFormat;
import android.media.MediaCodec;
import android.media.MediaExtractor;
import android.media.MediaFormat;

import java.io.File;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.Locale;

public final class MediaAudioDecoder {
    private static final long TIMEOUT_US = 10000L;

    public DecodedAudio decode(File file) throws Exception {
        MediaExtractor extractor = new MediaExtractor();
        MediaCodec codec = null;
        try {
            extractor.setDataSource(file.getAbsolutePath());
            int track = selectAudioTrack(extractor);
            if (track < 0) throw new IllegalStateException("No audio track found: " + file.getName());
            extractor.selectTrack(track);

            MediaFormat inputFormat = extractor.getTrackFormat(track);
            String mime = inputFormat.getString(MediaFormat.KEY_MIME);
            if (mime == null) throw new IllegalStateException("Audio MIME is missing.");
            codec = MediaCodec.createDecoderByType(mime);
            codec.configure(inputFormat, null, null, 0);
            codec.start();

            FloatBufferBuilder samples = new FloatBufferBuilder();
            MediaCodec.BufferInfo info = new MediaCodec.BufferInfo();
            boolean inputDone = false;
            boolean outputDone = false;
            int sampleRate = inputFormat.containsKey(MediaFormat.KEY_SAMPLE_RATE)
                    ? inputFormat.getInteger(MediaFormat.KEY_SAMPLE_RATE) : 16000;
            int channels = inputFormat.containsKey(MediaFormat.KEY_CHANNEL_COUNT)
                    ? inputFormat.getInteger(MediaFormat.KEY_CHANNEL_COUNT) : 1;
            int pcmEncoding = AudioFormat.ENCODING_PCM_16BIT;

            while (!outputDone) {
                if (!inputDone) {
                    int inIndex = codec.dequeueInputBuffer(TIMEOUT_US);
                    if (inIndex >= 0) {
                        ByteBuffer input = codec.getInputBuffer(inIndex);
                        if (input == null) continue;
                        int size = extractor.readSampleData(input, 0);
                        if (size < 0) {
                            codec.queueInputBuffer(inIndex, 0, 0, 0, MediaCodec.BUFFER_FLAG_END_OF_STREAM);
                            inputDone = true;
                        } else {
                            codec.queueInputBuffer(inIndex, 0, size, extractor.getSampleTime(), 0);
                            extractor.advance();
                        }
                    }
                }

                int outIndex = codec.dequeueOutputBuffer(info, TIMEOUT_US);
                if (outIndex == MediaCodec.INFO_OUTPUT_FORMAT_CHANGED) {
                    MediaFormat outFormat = codec.getOutputFormat();
                    if (outFormat.containsKey(MediaFormat.KEY_SAMPLE_RATE)) {
                        sampleRate = outFormat.getInteger(MediaFormat.KEY_SAMPLE_RATE);
                    }
                    if (outFormat.containsKey(MediaFormat.KEY_CHANNEL_COUNT)) {
                        channels = Math.max(1, outFormat.getInteger(MediaFormat.KEY_CHANNEL_COUNT));
                    }
                    if (outFormat.containsKey(MediaFormat.KEY_PCM_ENCODING)) {
                        pcmEncoding = outFormat.getInteger(MediaFormat.KEY_PCM_ENCODING);
                    }
                } else if (outIndex >= 0) {
                    ByteBuffer output = codec.getOutputBuffer(outIndex);
                    if (output != null && info.size > 0) {
                        output.position(info.offset);
                        output.limit(info.offset + info.size);
                        appendPcm(samples, output.slice().order(ByteOrder.LITTLE_ENDIAN), channels, pcmEncoding);
                    }
                    outputDone = (info.flags & MediaCodec.BUFFER_FLAG_END_OF_STREAM) != 0;
                    codec.releaseOutputBuffer(outIndex, false);
                }
            }

            float[] mono = samples.toArray();
            return new DecodedAudio(resampleTo16k(mono, sampleRate), 16000, 1,
                    String.format(Locale.US, "mediacodec:%s:%dHz", mime, sampleRate));
        } finally {
            if (codec != null) {
                try {
                    codec.stop();
                } catch (Exception ignored) {
                }
                codec.release();
            }
            extractor.release();
        }
    }

    private int selectAudioTrack(MediaExtractor extractor) {
        for (int i = 0; i < extractor.getTrackCount(); i++) {
            MediaFormat format = extractor.getTrackFormat(i);
            String mime = format.getString(MediaFormat.KEY_MIME);
            if (mime != null && mime.startsWith("audio/")) return i;
        }
        return -1;
    }

    private void appendPcm(FloatBufferBuilder out, ByteBuffer pcm, int channels, int encoding) {
        channels = Math.max(1, channels);
        if (encoding == AudioFormat.ENCODING_PCM_FLOAT) {
            int frames = pcm.remaining() / (channels * 4);
            for (int i = 0; i < frames; i++) {
                float sum = 0f;
                for (int c = 0; c < channels; c++) sum += pcm.getFloat();
                out.add(sum / channels);
            }
            return;
        }
        int frames = pcm.remaining() / (channels * 2);
        for (int i = 0; i < frames; i++) {
            float sum = 0f;
            for (int c = 0; c < channels; c++) sum += pcm.getShort() / 32768.0f;
            out.add(sum / channels);
        }
    }

    private float[] resampleTo16k(float[] input, int sampleRate) {
        if (sampleRate == 16000) return input;
        int outLen = Math.max(1, (int) Math.round(input.length * 16000.0 / sampleRate));
        float[] output = new float[outLen];
        double ratio = sampleRate / 16000.0;
        for (int i = 0; i < outLen; i++) {
            double src = i * ratio;
            int left = Math.min(input.length - 1, (int) Math.floor(src));
            int right = Math.min(input.length - 1, left + 1);
            double frac = src - left;
            output[i] = (float) (input[left] * (1.0 - frac) + input[right] * frac);
        }
        return output;
    }

    private static final class FloatBufferBuilder {
        private float[] data = new float[16000 * 30];
        private int size;

        void add(float value) {
            if (size == data.length) {
                float[] next = new float[data.length * 2];
                System.arraycopy(data, 0, next, 0, data.length);
                data = next;
            }
            data[size++] = value;
        }

        float[] toArray() {
            float[] out = new float[size];
            System.arraycopy(data, 0, out, 0, size);
            return out;
        }
    }
}
