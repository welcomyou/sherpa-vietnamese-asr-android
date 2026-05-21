package com.asrvn.offline.vad;

import ai.onnxruntime.OnnxTensor;
import ai.onnxruntime.OrtEnvironment;
import ai.onnxruntime.OrtSession;

import java.io.BufferedInputStream;
import java.io.DataInputStream;
import java.io.File;
import java.io.FileInputStream;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.FloatBuffer;
import java.nio.LongBuffer;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

public final class SileroVadEngine implements AutoCloseable {
    private static final int SAMPLE_RATE = 16000;
    private static final int WINDOW_SIZE = 512;
    private static final int CONTEXT_SIZE = 64;
    private static final float VAD_BOOST_TARGET = 0.071f;
    private final OrtEnvironment env;
    private final OrtSession session;
    private final List<String> inputNames;
    private final List<String> outputNames;

    public SileroVadEngine(OrtEnvironment env, byte[] model, int threads) throws Exception {
        this.env = env;
        OrtSession.SessionOptions options = new OrtSession.SessionOptions();
        options.setOptimizationLevel(OrtSession.SessionOptions.OptLevel.ALL_OPT);
        options.setIntraOpNumThreads(Math.max(1, Math.min(8, threads)));
        try {
            options.addXnnpack(java.util.Collections.singletonMap("intra_op_num_threads", String.valueOf(Math.max(1, Math.min(8, threads)))));
        } catch (Throwable ignored) {
            options.addCPU(true);
        }
        session = env.createSession(model, options);
        inputNames = new ArrayList<>(session.getInputNames());
        outputNames = new ArrayList<>(session.getOutputNames());
    }

    public VadResult run(float[] audio, boolean bypass) throws Exception {
        if (bypass) {
            List<SpeechSegment> all = new ArrayList<>();
            all.add(new SpeechSegment(0, audio.length));
            return new VadResult(all, new float[0], true);
        }
        if (audio.length < WINDOW_SIZE) {
            List<SpeechSegment> all = new ArrayList<>();
            all.add(new SpeechSegment(0, audio.length));
            return new VadResult(all, new float[0], false);
        }
        float[] audioForVad = boostedForVad(audio);
        InferenceResult first = runInference(audioForVad);
        List<SpeechSegment> segments = toSegments(first.probabilities, audio.length, 0.2f, 100, 250, 1000, 250);
        if (segments.isEmpty()) {
            InferenceResult retry = runInference(audioForVad);
            segments = toSegments(retry.probabilities, audio.length, 0.3f, 100, 150, 1000, 250);
            if (!segments.isEmpty()) return new VadResult(segments, retry.probabilities, false);
        }
        if (segments.isEmpty()) {
            List<SpeechSegment> all = new ArrayList<>();
            all.add(new SpeechSegment(0, audio.length));
            return new VadResult(all, first.probabilities, false);
        }
        return new VadResult(segments, first.probabilities, false);
    }

    public VadResult run(File pcmFloatFile, int sampleCount, boolean bypass) throws Exception {
        if (bypass) {
            List<SpeechSegment> all = new ArrayList<>();
            all.add(new SpeechSegment(0, sampleCount));
            return new VadResult(all, new float[0], true);
        }
        if (sampleCount < WINDOW_SIZE) {
            List<SpeechSegment> all = new ArrayList<>();
            all.add(new SpeechSegment(0, sampleCount));
            return new VadResult(all, new float[0], false);
        }
        float max = maxAbs(pcmFloatFile);
        float scale = max > 1e-6f && max < VAD_BOOST_TARGET ? VAD_BOOST_TARGET / max : 1.0f;
        InferenceResult first = runInference(pcmFloatFile, sampleCount, scale);
        List<SpeechSegment> segments = toSegments(first.probabilities, sampleCount, 0.2f, 100, 250, 1000, 250);
        if (segments.isEmpty()) {
            InferenceResult retry = runInference(pcmFloatFile, sampleCount, scale);
            segments = toSegments(retry.probabilities, sampleCount, 0.3f, 100, 150, 1000, 250);
            if (!segments.isEmpty()) return new VadResult(segments, retry.probabilities, false);
        }
        if (segments.isEmpty()) {
            List<SpeechSegment> all = new ArrayList<>();
            all.add(new SpeechSegment(0, sampleCount));
            return new VadResult(all, first.probabilities, false);
        }
        return new VadResult(segments, first.probabilities, false);
    }

    private InferenceResult runInference(float[] audio) throws Exception {
        int windows = audio.length / WINDOW_SIZE;
        float[] input = new float[CONTEXT_SIZE + WINDOW_SIZE];
        float[] state = new float[2 * 1 * 128];
        float[] context = new float[CONTEXT_SIZE];
        float[] probabilities = new float[windows];

        try (OnnxTensor sr = OnnxTensor.createTensor(env, LongBuffer.wrap(new long[] {SAMPLE_RATE}), new long[] {})) {
            for (int i = 0; i < windows; i++) {
                int start = i * WINDOW_SIZE;
                System.arraycopy(context, 0, input, 0, CONTEXT_SIZE);
                System.arraycopy(audio, start, input, CONTEXT_SIZE, WINDOW_SIZE);
                try (OnnxTensor inputTensor = OnnxTensor.createTensor(env, FloatBuffer.wrap(input), new long[] {1, input.length});
                     OnnxTensor stateTensor = OnnxTensor.createTensor(env, FloatBuffer.wrap(state), new long[] {2, 1, 128});
                     OrtSession.Result outputs = session.run(feed(inputTensor, stateTensor, sr))) {
                    OnnxTensor prob = (OnnxTensor) outputs.get("output").orElse(outputs.get(outputNames.get(0)).orElseThrow());
                    OnnxTensor nextState = (OnnxTensor) outputs.get("stateN").orElse(outputs.get(outputNames.get(1)).orElseThrow());
                    probabilities[i] = prob.getFloatBuffer().get(0);
                    nextState.getFloatBuffer().get(state);
                }
                System.arraycopy(audio, start + WINDOW_SIZE - CONTEXT_SIZE, context, 0, CONTEXT_SIZE);
            }
        }

        return new InferenceResult(probabilities);
    }

    private InferenceResult runInference(File pcmFloatFile, int sampleCount, float scale) throws Exception {
        int windows = sampleCount / WINDOW_SIZE;
        float[] input = new float[CONTEXT_SIZE + WINDOW_SIZE];
        float[] state = new float[2 * 1 * 128];
        float[] context = new float[CONTEXT_SIZE];
        float[] probabilities = new float[windows];
        float[] window = new float[WINDOW_SIZE];
        byte[] raw = new byte[WINDOW_SIZE * 4];

        try (DataInputStream stream = new DataInputStream(new BufferedInputStream(new FileInputStream(pcmFloatFile), raw.length * 8));
             OnnxTensor sr = OnnxTensor.createTensor(env, LongBuffer.wrap(new long[] {SAMPLE_RATE}), new long[] {})) {
            for (int i = 0; i < windows; i++) {
                stream.readFully(raw, 0, raw.length);
                ByteBuffer bytes = ByteBuffer.wrap(raw).order(ByteOrder.LITTLE_ENDIAN);
                for (int s = 0; s < WINDOW_SIZE; s++) window[s] = bytes.getFloat() * scale;

                System.arraycopy(context, 0, input, 0, CONTEXT_SIZE);
                System.arraycopy(window, 0, input, CONTEXT_SIZE, WINDOW_SIZE);
                try (OnnxTensor inputTensor = OnnxTensor.createTensor(env, FloatBuffer.wrap(input), new long[] {1, input.length});
                     OnnxTensor stateTensor = OnnxTensor.createTensor(env, FloatBuffer.wrap(state), new long[] {2, 1, 128});
                     OrtSession.Result outputs = session.run(feed(inputTensor, stateTensor, sr))) {
                    OnnxTensor prob = (OnnxTensor) outputs.get("output").orElse(outputs.get(outputNames.get(0)).orElseThrow());
                    OnnxTensor nextState = (OnnxTensor) outputs.get("stateN").orElse(outputs.get(outputNames.get(1)).orElseThrow());
                    probabilities[i] = prob.getFloatBuffer().get(0);
                    nextState.getFloatBuffer().get(state);
                }
                System.arraycopy(window, WINDOW_SIZE - CONTEXT_SIZE, context, 0, CONTEXT_SIZE);
            }
        }

        return new InferenceResult(probabilities);
    }

    private Map<String, OnnxTensor> feed(OnnxTensor input, OnnxTensor state, OnnxTensor sr) {
        Map<String, OnnxTensor> feeds = new HashMap<>();
        feeds.put("input", input);
        feeds.put("state", state);
        feeds.put("sr", sr);
        if (!inputNames.contains("input")) feeds.put(inputNames.get(0), input);
        if (!inputNames.contains("state") && inputNames.size() > 1) feeds.put(inputNames.get(1), state);
        if (!inputNames.contains("sr") && inputNames.size() > 2) feeds.put(inputNames.get(2), sr);
        return feeds;
    }

    private float[] boostedForVad(float[] audio) {
        float max = 0.0f;
        for (float sample : audio) max = Math.max(max, Math.abs(sample));
        if (max <= 1e-6f || max >= VAD_BOOST_TARGET) return audio;
        float scale = VAD_BOOST_TARGET / max;
        float[] boosted = new float[audio.length];
        for (int i = 0; i < audio.length; i++) boosted[i] = audio[i] * scale;
        return boosted;
    }

    private float maxAbs(File pcmFloatFile) throws Exception {
        float max = 0.0f;
        byte[] raw = new byte[1024 * 1024];
        try (DataInputStream input = new DataInputStream(new BufferedInputStream(new FileInputStream(pcmFloatFile), raw.length))) {
            long remaining = pcmFloatFile.length() - (pcmFloatFile.length() % 4L);
            while (remaining > 0) {
                int read = (int) Math.min(raw.length, remaining);
                input.readFully(raw, 0, read);
                ByteBuffer bytes = ByteBuffer.wrap(raw, 0, read).order(ByteOrder.LITTLE_ENDIAN);
                for (int i = 0; i < read / 4; i++) {
                    max = Math.max(max, Math.abs(bytes.getFloat()));
                }
                remaining -= read;
            }
        }
        return max;
    }

    private List<SpeechSegment> toSegments(float[] probabilities, int totalSamples, float threshold, int minSilenceMs, int minSpeechMs, int paddingMs, int mergeGapMs) {
        int minSilenceWindows = (int) Math.floor((minSilenceMs * SAMPLE_RATE) / 1000.0 / WINDOW_SIZE);
        int minSpeechWindows = (int) Math.floor((minSpeechMs * SAMPLE_RATE) / 1000.0 / WINDOW_SIZE);
        int padding = (int) ((paddingMs * SAMPLE_RATE) / 1000.0);
        int mergeGap = (int) ((mergeGapMs * SAMPLE_RATE) / 1000.0);
        List<SpeechSegment> raw = new ArrayList<>();
        boolean speech = false;
        int speechStart = 0;
        int silenceCount = 0;
        for (int i = 0; i < probabilities.length; i++) {
            if (probabilities[i] >= threshold) {
                if (!speech) {
                    speech = true;
                    speechStart = i;
                }
                silenceCount = 0;
            } else if (speech) {
                silenceCount += 1;
                if (silenceCount >= minSilenceWindows) {
                    int speechEnd = i - silenceCount + 1;
                    if (speechEnd - speechStart >= minSpeechWindows) {
                        raw.add(new SpeechSegment(
                                Math.max(0, speechStart * WINDOW_SIZE - padding),
                                Math.min(totalSamples, speechEnd * WINDOW_SIZE + padding)
                        ));
                    }
                    speech = false;
                    silenceCount = 0;
                }
            }
        }
        if (speech && probabilities.length - speechStart >= minSpeechWindows) {
            raw.add(new SpeechSegment(
                    Math.max(0, speechStart * WINDOW_SIZE - padding),
                    Math.min(totalSamples, probabilities.length * WINDOW_SIZE + padding)
            ));
        }
        if (raw.size() <= 1) return raw;
        List<SpeechSegment> merged = new ArrayList<>();
        SpeechSegment current = raw.get(0);
        for (int i = 1; i < raw.size(); i++) {
            SpeechSegment next = raw.get(i);
            if (next.start - current.end < mergeGap) {
                current = new SpeechSegment(current.start, next.end);
            } else {
                merged.add(current);
                current = next;
            }
        }
        merged.add(current);
        return merged;
    }

    @Override
    public void close() throws Exception {
        session.close();
    }

    private static final class InferenceResult {
        final float[] probabilities;

        InferenceResult(float[] probabilities) {
            this.probabilities = probabilities;
        }
    }
}
