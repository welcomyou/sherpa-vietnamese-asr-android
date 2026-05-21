package com.asrvn.offline.quality;

import ai.onnxruntime.OnnxTensor;
import ai.onnxruntime.OrtEnvironment;
import ai.onnxruntime.OrtSession;

import java.nio.FloatBuffer;
import java.util.List;
import java.util.Collections;

public final class DnsmosEngine implements AutoCloseable {
    public static final class Score {
        public final double sig;
        public final double bak;
        public final double ovrl;
        public final String provider;

        public Score(double sig, double bak, double ovrl, String provider) {
            this.sig = sig;
            this.bak = bak;
            this.ovrl = ovrl;
            this.provider = provider == null ? "cpu" : provider;
        }
    }

    private static final int SAMPLE_LENGTH = 144160;
    private static final int MIN_SAMPLES = 8000;
    private final OrtEnvironment env;
    private final OrtSession session;
    private final String provider;

    public DnsmosEngine(OrtEnvironment env, byte[] model, boolean accelerator, int threads) throws Exception {
        this.env = env;
        OrtSession.SessionOptions options = new OrtSession.SessionOptions();
        options.setOptimizationLevel(OrtSession.SessionOptions.OptLevel.ALL_OPT);
        options.setIntraOpNumThreads(Math.max(1, Math.min(8, threads)));
        String selectedProvider = "cpu";
        if (accelerator) {
            try {
                options.addNnapi();
                selectedProvider = "nnapi";
            } catch (Throwable ignored) {
                // XNNPACK/CPU fallback below.
            }
        }
        if (!"nnapi".equals(selectedProvider)) {
            try {
                options.addXnnpack(Collections.singletonMap("intra_op_num_threads", String.valueOf(Math.max(1, Math.min(8, threads)))));
                selectedProvider = "xnnpack";
            } catch (Throwable ignored) {
                options.addCPU(true);
                selectedProvider = "cpu";
            }
        }
        session = env.createSession(model, options);
        provider = selectedProvider;
    }

    public String provider() {
        return provider;
    }

    public Score computeFromSpeech(float[] speechSamples) throws Exception {
        if (speechSamples == null || speechSamples.length < MIN_SAMPLES) return null;
        double[] positions = {0.15, 0.50, 0.85};
        double sig = 0.0;
        double bak = 0.0;
        double ovrl = 0.0;
        int count = 0;
        for (double position : positions) {
            int center = (int) Math.floor(speechSamples.length * position);
            int start = Math.max(0, center - SAMPLE_LENGTH / 2);
            int end = Math.min(speechSamples.length, start + SAMPLE_LENGTH);
            if (end - start < MIN_SAMPLES) continue;
            Score score = computeSingle(speechSamples, start, end);
            sig += score.sig;
            bak += score.bak;
            ovrl += score.ovrl;
            count++;
        }
        if (count == 0) return null;
        return new Score(round2(sig / count), round2(bak / count), round2(ovrl / count), provider);
    }

    public Score computeWindows(List<float[]> windows) throws Exception {
        if (windows == null || windows.isEmpty()) return null;
        double sig = 0.0;
        double bak = 0.0;
        double ovrl = 0.0;
        int count = 0;
        for (float[] window : windows) {
            if (window == null || window.length < MIN_SAMPLES) continue;
            Score score = computeSingle(window, 0, Math.min(window.length, SAMPLE_LENGTH));
            sig += score.sig;
            bak += score.bak;
            ovrl += score.ovrl;
            count++;
        }
        if (count == 0) return null;
        return new Score(round2(sig / count), round2(bak / count), round2(ovrl / count), provider);
    }

    private Score computeSingle(float[] samples, int start, int end) throws Exception {
        float[] input = new float[SAMPLE_LENGTH];
        System.arraycopy(samples, start, input, 0, Math.min(end - start, SAMPLE_LENGTH));
        String inputName = session.getInputNames().iterator().next();
        try (OnnxTensor tensor = OnnxTensor.createTensor(env, FloatBuffer.wrap(input), new long[]{1, SAMPLE_LENGTH});
             OrtSession.Result result = session.run(Collections.singletonMap(inputName, tensor))) {
            FloatBuffer buffer = ((OnnxTensor) result.get(0)).getFloatBuffer();
            float[] raw = new float[3];
            buffer.get(raw);
            double sig = clip(-0.08397278 * raw[0] * raw[0] + 1.22083953 * raw[0] + 0.0052439);
            double bak = clip(-0.13166888 * raw[1] * raw[1] + 1.60915514 * raw[1] - 0.39604546);
            double ovrl = clip(-0.06766283 * raw[2] * raw[2] + 1.11546468 * raw[2] + 0.04602535);
            return new Score(sig, bak, ovrl, provider);
        }
    }

    private static double clip(double value) {
        return Math.max(1.0, Math.min(5.0, value));
    }

    private static double round2(double value) {
        return Math.round(value * 100.0) / 100.0;
    }

    @Override
    public void close() throws Exception {
        session.close();
    }
}
