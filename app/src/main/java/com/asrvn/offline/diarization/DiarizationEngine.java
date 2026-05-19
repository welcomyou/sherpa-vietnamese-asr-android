package com.asrvn.offline.diarization;

import ai.onnxruntime.OrtEnvironment;
import ai.onnxruntime.OrtSession;
import ai.onnxruntime.OnnxTensor;
import ai.onnxruntime.OnnxValue;

import com.asrvn.offline.asr.FbankExtractor;
import com.asrvn.offline.asr.PureOrtRecognizer;
import com.asrvn.offline.models.ModelFileRegistry;
import com.asrvn.offline.vad.SpeechSegment;

import org.apache.commons.math3.linear.EigenDecomposition;
import org.apache.commons.math3.linear.MatrixUtils;
import org.apache.commons.math3.linear.RealMatrix;

import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.nio.FloatBuffer;

public final class DiarizationEngine implements AutoCloseable {
    private static final int SAMPLE_RATE = 16000;
    private static final double CAMPP_WINDOW_SECONDS = 1.5;
    private static final double CAMPP_STEP_SECONDS = 0.6;
    private static final int CAMPP_WINDOW_FRAMES = 150;
    private static final int CAMPP_STEP_FRAMES = 60;
    private static final int CAMPP_MEL_BINS = 80;
    private static final int CAMPP_BATCH = 32;
    private static final double CAMPP_TURN_MERGE_GAP_SECONDS = 4.0;
    private static final double NEW_SPEAKER_COS = 0.62;
    private static final double MERGE_SPEAKER_COS = 0.875;
    private static final int MIN_CLUSTER_SIZE = 4;
    private static final int DIAR_CHUNK_SECONDS = 10;
    private static final int DIAR_CHUNK_SAMPLES = DIAR_CHUNK_SECONDS * SAMPLE_RATE;
    private static final int PYANNOTE_STEP_SAMPLES = SAMPLE_RATE;
    private static final int CAMPP_VAD_STEP_SAMPLES = 5 * SAMPLE_RATE;
    private static final int PYANNOTE_MAX_SPEAKERS_PER_CHUNK = 3;
    private static final int PYANNOTE_EMB_DIM = 256;
    private static final int PYANNOTE_PLDA_DIM = 128;
    private static final int WESPEAKER_MEL_BINS = 80;
    private static final int WESPEAKER_BATCH = 8;
    private static final int PYANNOTE_EMB_MIN_NUM_SAMPLES = 1680;
    private static final double PYANNOTE_RF_START = 0.0;
    private static final double PYANNOTE_RF_DURATION = 0.0619375;
    private static final double PYANNOTE_RF_STEP = 0.016875;
    private static final double PYANNOTE_DEFAULT_THRESHOLD = 0.6;
    private static final double PYANNOTE_FA = 0.07;
    private static final double PYANNOTE_FB = 0.8;
    private static final double PYANNOTE_MERGE_GAP_SECONDS = 0.0;
    private static final int PYANNOTE_BATCH = 8;
    private static final int[][] DIAR_POWERSET = new int[][]{
            {0, 0, 0},
            {1, 0, 0},
            {0, 1, 0},
            {0, 0, 1},
            {1, 1, 0},
            {1, 0, 1},
            {0, 1, 1}
    };

    private static final class WindowEmbedding {
        final float[] vector;
        final double start;
        final double end;

        WindowEmbedding(float[] vector, double start, double end) {
            this.vector = vector;
            this.start = start;
            this.end = end;
        }
    }

    private final OrtEnvironment env;
    private final ModelFileRegistry models;
    private final FbankExtractor asrFbank = new FbankExtractor();
    private final CamppFbankExtractor camppFbank = new CamppFbankExtractor();
    private final WespeakerFbankExtractor wespeakerFbank = new WespeakerFbankExtractor();
    private OrtSession camppSession;
    private OrtSession pyannoteSegSession;
    private OrtSession pyannoteEmbeddingSession;
    private PyannoteCommunityAssets pyannoteAssets;
    private String provider = "cpu";
    private String camppProvider = "cpu";
    private String pyannoteSegProvider = "cpu";
    private String pyannoteEmbeddingProvider = "cpu";
    private int lastSenkoSpeakers = 0;
    private double[] lastSenkoEigenvalues = new double[0];
    private int[] lastSenkoSpectralLabels = new int[0];
    private int[] lastSenkoAfterFilterLabels = new int[0];
    private int[] lastSenkoAfterMergeLabels = new int[0];

    public DiarizationEngine(OrtEnvironment env, ModelFileRegistry models) {
        this.env = env;
        this.models = models;
    }

    public DiarizationResult runCampp(float[] samples, boolean accelerator, int threads) throws Exception {
        return runCampp(samples, null, accelerator, threads);
    }

    public DiarizationResult runCampp(float[] samples, List<SpeechSegment> speechSegments, boolean accelerator, int threads) throws Exception {
        ensureCampp(accelerator, threads);
        ensurePyannote(false, threads);
        List<double[]> regions = pyannoteVadRegions(samples, CAMPP_VAD_STEP_SAMPLES, 0.25, 0.1);
        List<WindowEmbedding> embeddings = extractCamppEmbeddings(samples, regions);
        if (embeddings.isEmpty()) {
            return new DiarizationResult(new ArrayList<>(), "senko_campp_optimized", camppProviderSummary());
        }
        int[] labels = senkoCluster(embeddings, samples.length / (double) SAMPLE_RATE);
        List<DiarizationResult.Turn> turns = senkoPostProcess(labelsToTurns(embeddings, labels));
        List<DiarizationResult.Turn> postTurns = postProcessDiarizationTurns(turns, null);
        writeDebugDump("campp_debug.json", "{\"windows\":" + embeddings.size()
                + ",\"spectral_speakers\":" + lastSenkoSpeakers
                + ",\"eigenvalues\":" + doubleArrayJson(lastSenkoEigenvalues, Math.min(32, lastSenkoEigenvalues.length))
                + ",\"labels_spectral\":" + intArrayJson(lastSenkoSpectralLabels)
                + ",\"labels_after_filter\":" + intArrayJson(lastSenkoAfterFilterLabels)
                + ",\"labels_after_merge\":" + intArrayJson(lastSenkoAfterMergeLabels)
                + ",\"labels\":" + intArrayJson(labels)
                + ",\"raw_turns\":" + turnsJson(turns)
                + ",\"postprocess_preview_turns\":" + turnsJson(postTurns) + "}");
        return new DiarizationResult(turns, "senko_campp_optimized", camppProviderSummary());
    }

    public DiarizationResult runPyannote(float[] samples, boolean accelerator, int threads) throws Exception {
        ensurePyannote(accelerator, threads);
        ensurePyannoteCommunity(accelerator, threads);
        int[] starts = pyannoteChunkStarts(samples == null ? 0 : samples.length, PYANNOTE_STEP_SAMPLES);
        PyannoteSegmentation segmentation = runPyannoteSegmentation(samples, PYANNOTE_STEP_SAMPLES, PYANNOTE_BATCH);
        if (segmentation.chunks == 0 || segmentation.numFrames == 0) {
            return new DiarizationResult(new ArrayList<>(), "pyannote_community1_vbx", pyannoteProviderSummary() + ",segmentation_empty");
        }
        byte[] binarized = powerSetBinarize(segmentation);
        byte[] cleanBinarized = buildCleanBinarized(binarized, segmentation.chunks, segmentation.numFrames);
        float[] countData = aggregatePyannoteCount(binarized, segmentation.chunks, segmentation.numFrames);
        double[][] embeddings = extractPyannoteEmbeddings(samples, binarized, cleanBinarized, segmentation.numFrames, starts);
        int[] hardClusters = canonicalizePyannoteClusters(
                clusterPyannoteEmbeddings(embeddings, binarized, segmentation.chunks, segmentation.numFrames),
                binarized,
                segmentation.chunks,
                segmentation.numFrames);
        List<DiarizationResult.Turn> turns = reconstructPyannoteSegments(
                binarized,
                hardClusters,
                countData,
                segmentation.chunks,
                segmentation.numFrames,
                samples.length / (double) SAMPLE_RATE);
        List<DiarizationResult.Turn> postTurns = postProcessDiarizationTurns(turns, null);
        writeDebugDump("pyannote_debug.json", "{\"chunks\":" + segmentation.chunks
                + ",\"frames\":" + segmentation.numFrames
                + ",\"hard_clusters\":" + intArrayJson(hardClusters)
                + ",\"raw_turns\":" + turnsJson(turns)
                + ",\"postprocess_preview_turns\":" + turnsJson(postTurns)
                + ",\"first_embedding\":" + doubleArrayJson(firstFiniteEmbedding(embeddings), 32)
                + "}");
        return new DiarizationResult(turns, "pyannote_community1_vbx", pyannoteProviderSummary());
    }

    public static DiarizationResult postProcess(DiarizationResult result, List<PureOrtRecognizer.Word> words) {
        if (result == null || result.turns == null || result.turns.isEmpty()) return result;
        return new DiarizationResult(postProcessDiarizationTurns(result.turns, words), result.backend, result.provider);
    }

    private static List<DiarizationResult.Turn> postProcessDiarizationTurns(List<DiarizationResult.Turn> turns, List<PureOrtRecognizer.Word> words) {
        if (turns == null || turns.isEmpty()) return turns;
        List<DiarizationResult.Turn> processed = mergeSegmentsWithGap(turns, 0.3);
        processed = resolveFragmentZones(processed, 0.5, 3);
        processed = naturalTurnMerge(processed, 2.0, words);
        return mergeSegmentsWithGap(processed, 0.3);
    }

    private static List<DiarizationResult.Turn> mergeSegmentsWithGap(List<DiarizationResult.Turn> turns, double maxGap) {
        if (turns == null || turns.isEmpty()) return new ArrayList<>();
        List<DiarizationResult.Turn> sorted = new ArrayList<>(turns);
        sorted.sort((a, b) -> {
            int byStart = Double.compare(a.start, b.start);
            return byStart != 0 ? byStart : Integer.compare(a.speaker, b.speaker);
        });
        List<DiarizationResult.Turn> merged = new ArrayList<>();
        DiarizationResult.Turn first = sorted.get(0);
        merged.add(new DiarizationResult.Turn(first.start, first.end, first.speaker));
        for (int i = 1; i < sorted.size(); i++) {
            DiarizationResult.Turn turn = sorted.get(i);
            DiarizationResult.Turn prev = merged.get(merged.size() - 1);
            double gap = turn.start - prev.end;
            if (turn.speaker == prev.speaker && gap <= maxGap) {
                merged.set(merged.size() - 1, new DiarizationResult.Turn(prev.start, Math.max(prev.end, turn.end), prev.speaker));
            } else {
                merged.add(new DiarizationResult.Turn(turn.start, turn.end, turn.speaker));
            }
        }
        return merged;
    }

    private static List<DiarizationResult.Turn> resolveFragmentZones(List<DiarizationResult.Turn> turns, double shortThreshold, int minZoneSize) {
        if (turns.size() < minZoneSize) return turns;
        List<DiarizationResult.Turn> result = new ArrayList<>();
        int i = 0;
        while (i < turns.size()) {
            if (turns.get(i).end - turns.get(i).start < shortThreshold) {
                int j = i;
                while (j < turns.size() && turns.get(j).end - turns.get(j).start < shortThreshold) j++;
                if (j - i >= minZoneSize) {
                    Map<Integer, Double> durations = new HashMap<>();
                    for (int k = i; k < j; k++) {
                        DiarizationResult.Turn turn = turns.get(k);
                        durations.put(turn.speaker, durations.getOrDefault(turn.speaker, 0.0) + turn.end - turn.start);
                    }
                    int dominant = turns.get(i).speaker;
                    double best = Double.NEGATIVE_INFINITY;
                    for (Map.Entry<Integer, Double> entry : durations.entrySet()) {
                        if (entry.getValue() > best) {
                            best = entry.getValue();
                            dominant = entry.getKey();
                        }
                    }
                    result.add(new DiarizationResult.Turn(turns.get(i).start, turns.get(j - 1).end, dominant));
                    i = j;
                    continue;
                }
            }
            DiarizationResult.Turn turn = turns.get(i);
            result.add(new DiarizationResult.Turn(turn.start, turn.end, turn.speaker));
            i++;
        }
        return result;
    }

    private static List<DiarizationResult.Turn> naturalTurnMerge(List<DiarizationResult.Turn> turns, double maxPause, List<PureOrtRecognizer.Word> words) {
        if (turns.size() < 3) return turns;
        List<DiarizationResult.Turn> sorted = new ArrayList<>(turns);
        sorted.sort((a, b) -> Double.compare(a.start, b.start));

        List<Integer> speakers = new ArrayList<>();
        for (DiarizationResult.Turn turn : sorted) if (!speakers.contains(turn.speaker)) speakers.add(turn.speaker);
        List<VirtualTurn> virtualTurns = new ArrayList<>();
        for (int speaker : speakers) {
            List<Integer> indices = new ArrayList<>();
            for (int i = 0; i < sorted.size(); i++) if (sorted.get(i).speaker == speaker) indices.add(i);
            if (indices.isEmpty()) continue;
            double start = sorted.get(indices.get(0)).start;
            double end = sorted.get(indices.get(0)).end;
            List<Integer> turnIndices = new ArrayList<>();
            turnIndices.add(indices.get(0));
            for (int k = 1; k < indices.size(); k++) {
                int idx = indices.get(k);
                double gap = sorted.get(idx).start - end;
                if (gap < maxPause) {
                    end = Math.max(end, sorted.get(idx).end);
                    turnIndices.add(idx);
                } else {
                    virtualTurns.add(new VirtualTurn(start, end, speaker, turnIndices));
                    start = sorted.get(idx).start;
                    end = sorted.get(idx).end;
                    turnIndices = new ArrayList<>();
                    turnIndices.add(idx);
                }
            }
            virtualTurns.add(new VirtualTurn(start, end, speaker, turnIndices));
        }
        virtualTurns.sort((a, b) -> Double.compare(a.start, b.start));

        boolean[] secondary = new boolean[virtualTurns.size()];
        int[] primaryOf = new int[virtualTurns.size()];
        java.util.Arrays.fill(primaryOf, -1);
        for (int i = 0; i < virtualTurns.size(); i++) {
            if (secondary[i]) continue;
            VirtualTurn t1 = virtualTurns.get(i);
            for (int j = i + 1; j < virtualTurns.size(); j++) {
                if (secondary[j]) continue;
                VirtualTurn t2 = virtualTurns.get(j);
                if (t2.start >= t1.end) break;
                if (t2.end <= t1.end && t2.speaker != t1.speaker) {
                    secondary[j] = true;
                    primaryOf[j] = i;
                }
            }
        }

        Map<Integer, Integer> reassign = new HashMap<>();
        for (int j = 0; j < virtualTurns.size(); j++) {
            if (!secondary[j] || primaryOf[j] < 0) continue;
            VirtualTurn turn = virtualTurns.get(j);
            double duration = turn.end - turn.start;
            if (duration >= 2.0) continue;
            int wordCount = countWordsInRange(words, turn.start, turn.end);
            if (wordCount >= 0 && wordCount > 3) continue;
            int primarySpeaker = virtualTurns.get(primaryOf[j]).speaker;
            for (int index : turn.segmentIndices) reassign.put(index, primarySpeaker);
        }

        List<DiarizationResult.Turn> result = new ArrayList<>();
        for (int i = 0; i < sorted.size(); i++) {
            DiarizationResult.Turn turn = sorted.get(i);
            int speaker = reassign.getOrDefault(i, turn.speaker);
            result.add(new DiarizationResult.Turn(turn.start, turn.end, speaker));
        }
        return mergeSegmentsWithGap(result, 0.5);
    }

    private static int countWordsInRange(List<PureOrtRecognizer.Word> words, double start, double end) {
        if (words == null || words.isEmpty()) return -1;
        int count = 0;
        for (PureOrtRecognizer.Word word : words) {
            double mid = (word.start + word.end) * 0.5;
            if (start <= mid && mid <= end) count++;
        }
        return count;
    }

    private static final class VirtualTurn {
        final double start;
        final double end;
        final int speaker;
        final List<Integer> segmentIndices;

        VirtualTurn(double start, double end, int speaker, List<Integer> segmentIndices) {
            this.start = start;
            this.end = end;
            this.speaker = speaker;
            this.segmentIndices = segmentIndices;
        }
    }

    public void writeEncoderDebugDump(float[] samples, int threads, boolean accelerator) throws Exception {
        writeDebugFloatFile("pcm_first10.f32", samples, Math.min(samples.length, 10 * SAMPLE_RATE));
        ensureCampp(accelerator, threads);
        String camppProvider = this.camppProvider;
        ensurePyannoteCommunity(accelerator, threads);
        String pyannoteProvider = this.pyannoteEmbeddingProvider;

        int camppSamples = Math.min(samples.length, (int) Math.round(CAMPP_WINDOW_SECONDS * SAMPLE_RATE));
        float[] camppAudio = new float[camppSamples];
        System.arraycopy(samples, 0, camppAudio, 0, camppSamples);
        FbankExtractor.Features asrFeatures = asrFbank.compute(camppAudio);
        CamppFbankExtractor.Features camppFeatures = camppFbank.compute(camppAudio);
        writeDebugFloatFile("asr_fbank_first1p5.f32", asrFeatures.data, asrFeatures.data.length);
        writeDebugFloatFile("campp_fbank_first1p5.f32", camppFeatures.data, camppFeatures.data.length);
        float[] camppInput = new float[Math.max(1, camppFeatures.frames) * CAMPP_MEL_BINS];
        System.arraycopy(camppFeatures.data, 0, camppInput, 0, Math.min(camppFeatures.data.length, camppInput.length));
        float[] camppEmbedding;
        String camppInputName = camppSession.getInputNames().iterator().next();
        try (OnnxTensor tensor = OnnxTensor.createTensor(env, FloatBuffer.wrap(camppInput), new long[]{1, Math.max(1, camppFeatures.frames), CAMPP_MEL_BINS});
             OrtSession.Result result = camppSession.run(Collections.singletonMap(camppInputName, tensor))) {
            camppEmbedding = flattenFloat(result.get(0).getValue());
        }
        writeDebugFloatFile("campp_embedding_first1p5.f32", camppEmbedding, camppEmbedding.length);

        float[] pyannoteAudio = paddedChunk(samples, 0);
        WespeakerFbankExtractor.Features pyannoteFeatures = wespeakerFbank.compute(pyannoteAudio);
        writeDebugFloatFile("pyannote_fbank_first10.f32", pyannoteFeatures.data, pyannoteFeatures.data.length);
        String pyannoteInputName = pyannoteEmbeddingSession.getInputNames().contains("fbank_features")
                ? "fbank_features"
                : pyannoteEmbeddingSession.getInputNames().iterator().next();
        float[] pyannoteEncoder;
        long[] pyannoteShape;
        try (OnnxTensor tensor = OnnxTensor.createTensor(env, FloatBuffer.wrap(pyannoteFeatures.data), new long[]{1, pyannoteFeatures.frames, WESPEAKER_MEL_BINS});
             OrtSession.Result result = pyannoteEmbeddingSession.run(Collections.singletonMap(pyannoteInputName, tensor))) {
            OnnxTensor out = (OnnxTensor) result.get(0);
            pyannoteShape = out.getInfo().getShape();
            pyannoteEncoder = flattenFloat(out.getValue());
        }
        writeDebugFloatFile("pyannote_encoder_first10.f32", pyannoteEncoder, pyannoteEncoder.length);

        String json = "{"
                + "\"accelerator\":" + (accelerator ? "true" : "false") + ","
                + "\"campp_provider\":\"" + camppProvider + "\","
                + "\"pyannote_provider\":\"" + pyannoteProvider + "\","
                + "\"sample_count\":" + samples.length + ","
                + "\"duration_sec\":" + String.format(java.util.Locale.US, "%.6f", samples.length / (double) SAMPLE_RATE) + ","
                + "\"asr_fbank\":" + floatSummaryJson(asrFeatures.data, asrFeatures.frames, 80) + ","
                + "\"campp_fbank\":" + floatSummaryJson(camppFeatures.data, camppFeatures.frames, CAMPP_MEL_BINS) + ","
                + "\"campp_embedding\":" + floatSummaryJson(camppEmbedding, 1, camppEmbedding.length) + ","
                + "\"pyannote_fbank\":" + floatSummaryJson(pyannoteFeatures.data, pyannoteFeatures.frames, WESPEAKER_MEL_BINS) + ","
                + "\"pyannote_encoder_shape\":" + longArrayJson(pyannoteShape) + ","
                + "\"pyannote_encoder\":" + floatSummaryJson(pyannoteEncoder, pyannoteShape.length > 2 ? (int) pyannoteShape[2] : 1, pyannoteShape.length > 1 ? (int) pyannoteShape[1] : pyannoteEncoder.length)
                + "}";
        writeDebugDump("encoder_dump_" + (accelerator ? "accelerated" : "cpu") + ".json", json);
    }

    private static final class PyannoteCommunityAssets {
        final double[] weight;
        final double[] bias;
        final double[] mean1;
        final double[] mean2;
        final double[] lda;
        final double[] mu;
        final double[] pldaTr;
        final double[] pldaPsi;

        PyannoteCommunityAssets(double[] weight, double[] bias, Map<String, NpyNpzReader.ArrayData> plda) {
            this.weight = weight;
            this.bias = bias;
            this.mean1 = requireArray(plda, "mean1", 256);
            this.mean2 = requireArray(plda, "mean2", 128);
            this.lda = requireArray(plda, "lda", 256 * 128);
            this.mu = requireArray(plda, "mu", 128);
            this.pldaTr = requireArray(plda, "plda_tr", 128 * 128);
            this.pldaPsi = requireArray(plda, "plda_psi", 128);
        }
    }

    private static final class PyannoteSegmentation {
        final float[] logits;
        final int chunks;
        final int numFrames;
        final int numClasses;

        PyannoteSegmentation(float[] logits, int chunks, int numFrames, int numClasses) {
            this.logits = logits;
            this.chunks = chunks;
            this.numFrames = numFrames;
            this.numClasses = numClasses;
        }
    }

    private void ensureCampp(boolean accelerator, int threads) throws Exception {
        if (camppSession != null) return;
        String[] selectedProvider = new String[]{"cpu"};
        camppSession = env.createSession(models.campp(), options(accelerator, threads, selectedProvider));
        camppProvider = selectedProvider[0];
    }

    private PyannoteSegmentation runPyannoteSegmentation(float[] samples, int stepSamples, int batchLimit) throws Exception {
        int[] starts = pyannoteChunkStarts(samples == null ? 0 : samples.length, stepSamples);
        float[] allLogits = null;
        int numFrames = 0;
        int numClasses = 0;
        int write = 0;
        for (int batchStart = 0; batchStart < starts.length; batchStart += batchLimit) {
            int batchSize = Math.min(batchLimit, starts.length - batchStart);
            float[] input = new float[batchSize * DIAR_CHUNK_SAMPLES];
            for (int b = 0; b < batchSize; b++) {
                int sourceStart = starts[batchStart + b];
                int sourceEnd = Math.min(samples.length, sourceStart + DIAR_CHUNK_SAMPLES);
                if (sourceEnd > sourceStart) {
                    System.arraycopy(samples, sourceStart, input, b * DIAR_CHUNK_SAMPLES, sourceEnd - sourceStart);
                }
            }
            try (OnnxTensor tensor = OnnxTensor.createTensor(env, FloatBuffer.wrap(input), new long[]{batchSize, 1, DIAR_CHUNK_SAMPLES});
                 OrtSession.Result result = pyannoteSegSession.run(Collections.singletonMap(pyannoteSegSession.getInputNames().iterator().next(), tensor))) {
                OnnxValue value = result.get("logits").orElse(result.get(0));
                float[] flat = flattenFloat(value.getValue());
                if (numFrames == 0) {
                    long[] shape = ((OnnxTensor) value).getInfo().getShape();
                    numFrames = (int) shape[1];
                    numClasses = (int) shape[2];
                    allLogits = new float[starts.length * numFrames * numClasses];
                }
                System.arraycopy(flat, 0, allLogits, write, flat.length);
                write += flat.length;
            }
        }
        return new PyannoteSegmentation(allLogits == null ? new float[0] : allLogits, starts.length, numFrames, numClasses);
    }

    private int[] pyannoteChunkStarts(int totalSamples, int stepSamples) {
        List<Integer> starts = new ArrayList<>();
        int start = 0;
        boolean hasLast = false;
        while (true) {
            if (hasLast) break;
            if ((start + DIAR_CHUNK_SAMPLES) > totalSamples) hasLast = true;
            starts.add(start);
            start += stepSamples;
        }
        if (starts.isEmpty()) starts.add(0);
        int[] out = new int[starts.size()];
        for (int i = 0; i < starts.size(); i++) out[i] = starts.get(i);
        return out;
    }

    private byte[] powerSetBinarize(PyannoteSegmentation segmentation) {
        byte[] binarized = new byte[segmentation.chunks * segmentation.numFrames * PYANNOTE_MAX_SPEAKERS_PER_CHUNK];
        for (int c = 0; c < segmentation.chunks; c++) {
            for (int f = 0; f < segmentation.numFrames; f++) {
                int base = (c * segmentation.numFrames + f) * segmentation.numClasses;
                int bestClass = 0;
                float bestValue = Float.NEGATIVE_INFINITY;
                for (int cls = 0; cls < segmentation.numClasses; cls++) {
                    float value = segmentation.logits[base + cls];
                    if (value > bestValue) {
                        bestValue = value;
                        bestClass = cls;
                    }
                }
                int[] active = DIAR_POWERSET[Math.min(bestClass, DIAR_POWERSET.length - 1)];
                int out = (c * segmentation.numFrames + f) * PYANNOTE_MAX_SPEAKERS_PER_CHUNK;
                binarized[out] = (byte) active[0];
                binarized[out + 1] = (byte) active[1];
                binarized[out + 2] = (byte) active[2];
            }
        }
        return binarized;
    }

    private List<double[]> pyannoteVadRegions(float[] samples, int stepSamples, double minSpeech, double minSilence) throws Exception {
        double duration = samples.length / (double) SAMPLE_RATE;
        PyannoteSegmentation segmentation = runPyannoteSegmentation(samples, stepSamples, 32);
        byte[] binarized = powerSetBinarize(segmentation);
        double frameSeconds = DIAR_CHUNK_SECONDS / (double) segmentation.numFrames;
        int outFrames = (int) (duration / frameSeconds) + 1;
        float[] speechCount = new float[outFrames];
        float[] totalCount = new float[outFrames];
        int[] starts = pyannoteChunkStarts(samples.length, stepSamples);
        for (int c = 0; c < segmentation.chunks; c++) {
            double chunkStart = starts[c] / (double) SAMPLE_RATE;
            for (int f = 0; f < segmentation.numFrames; f++) {
                int out = (int) ((chunkStart + f * frameSeconds) / frameSeconds);
                if (out < 0 || out >= outFrames) continue;
                int active = 0;
                for (int s = 0; s < PYANNOTE_MAX_SPEAKERS_PER_CHUNK; s++) {
                    active += pyannoteBinarizedAt(binarized, segmentation.numFrames, c, f, s);
                }
                if (active > 0) speechCount[out] += 1;
                totalCount[out] += 1;
            }
        }
        boolean[] speech = new boolean[outFrames];
        for (int i = 0; i < outFrames; i++) speech[i] = totalCount[i] > 0 && speechCount[i] / totalCount[i] > 0.5;
        List<double[]> regions = frameMaskToRegions(speech, frameSeconds, duration, minSpeech, minSilence);
        if (regions.isEmpty()) regions.add(new double[]{0.0, duration});
        return regions;
    }

    private List<double[]> frameMaskToRegions(boolean[] mask, double frameSeconds, double duration, double minSpeech, double minSilence) {
        List<double[]> raw = new ArrayList<>();
        boolean active = false;
        double start = 0.0;
        for (int i = 0; i < mask.length; i++) {
            double t = i * frameSeconds;
            if (mask[i] && !active) {
                start = t;
                active = true;
            } else if (!mask[i] && active) {
                if (t - start >= minSpeech) raw.add(new double[]{start, Math.min(t, duration)});
                active = false;
            }
        }
        if (active) {
            double end = Math.min(mask.length * frameSeconds, duration);
            if (end - start >= minSpeech) raw.add(new double[]{start, end});
        }
        if (raw.isEmpty()) return raw;
        List<double[]> merged = new ArrayList<>();
        merged.add(raw.get(0));
        for (int i = 1; i < raw.size(); i++) {
            double[] last = merged.get(merged.size() - 1);
            double[] next = raw.get(i);
            if (next[0] - last[1] < minSilence) {
                last[1] = next[1];
            } else {
                merged.add(next);
            }
        }
        return merged;
    }

    private int pyannoteBinarizedAt(byte[] binarized, int numFrames, int chunk, int frame, int speaker) {
        return binarized[(chunk * numFrames + frame) * PYANNOTE_MAX_SPEAKERS_PER_CHUNK + speaker];
    }

    private int closestPyannoteFrame(double time) {
        return roundHalfToEven((time - PYANNOTE_RF_START - 0.5 * PYANNOTE_RF_DURATION) / PYANNOTE_RF_STEP);
    }

    private int roundHalfToEven(double value) {
        int floor = (int) Math.floor(value);
        double diff = value - floor;
        if (diff < 0.5) return floor;
        if (diff > 0.5) return floor + 1;
        return floor % 2 == 0 ? floor : floor + 1;
    }

    private float[] aggregatePyannoteCount(byte[] binarized, int chunks, int numFrames) {
        int outFrames = closestPyannoteFrame((chunks - 1) + DIAR_CHUNK_SECONDS + 0.5 * PYANNOTE_RF_DURATION) + 1;
        float[] output = new float[outFrames];
        float[] weight = new float[outFrames];
        for (int c = 0; c < chunks; c++) {
            int startFrame = closestPyannoteFrame(c + 0.5 * PYANNOTE_RF_DURATION);
            for (int f = 0; f < numFrames; f++) {
                int out = startFrame + f;
                if (out < 0 || out >= outFrames) continue;
                int count = 0;
                for (int s = 0; s < PYANNOTE_MAX_SPEAKERS_PER_CHUNK; s++) {
                    count += pyannoteBinarizedAt(binarized, numFrames, c, f, s);
                }
                output[out] += count;
                weight[out] += 1;
            }
        }
        for (int i = 0; i < outFrames; i++) {
            output[i] = weight[i] > 0 ? roundHalfToEven(output[i] / weight[i]) : 0;
        }
        return output;
    }

    private byte[] buildCleanBinarized(byte[] binarized, int chunks, int numFrames) {
        byte[] clean = new byte[binarized.length];
        for (int c = 0; c < chunks; c++) {
            for (int f = 0; f < numFrames; f++) {
                int active = 0;
                for (int s = 0; s < PYANNOTE_MAX_SPEAKERS_PER_CHUNK; s++) active += pyannoteBinarizedAt(binarized, numFrames, c, f, s);
                if (active >= 2) continue;
                int base = (c * numFrames + f) * PYANNOTE_MAX_SPEAKERS_PER_CHUNK;
                clean[base] = binarized[base];
                clean[base + 1] = binarized[base + 1];
                clean[base + 2] = binarized[base + 2];
            }
        }
        return clean;
    }

    private double[][] extractPyannoteEmbeddings(float[] samples, byte[] binarized, byte[] cleanBinarized, int numFrames, int[] starts) throws Exception {
        int rows = starts.length * PYANNOTE_MAX_SPEAKERS_PER_CHUNK;
        double[][] embeddings = new double[rows][PYANNOTE_EMB_DIM];
        for (double[] row : embeddings) java.util.Arrays.fill(row, Double.NaN);
        int minSegFrames = (int) Math.ceil(numFrames * PYANNOTE_EMB_MIN_NUM_SAMPLES / (double) DIAR_CHUNK_SAMPLES);
        int[] featureIndex = null;
        String inputName = pyannoteEmbeddingSession.getInputNames().contains("fbank_features")
                ? "fbank_features"
                : pyannoteEmbeddingSession.getInputNames().iterator().next();
        for (int batchStart = 0; batchStart < starts.length; batchStart += WESPEAKER_BATCH) {
            int batchSize = Math.min(WESPEAKER_BATCH, starts.length - batchStart);
            WespeakerFbankExtractor.Features[] fbanks = new WespeakerFbankExtractor.Features[batchSize];
            int maxFrames = 0;
            for (int b = 0; b < batchSize; b++) {
                fbanks[b] = wespeakerFbank.compute(paddedChunk(samples, starts[batchStart + b]));
                maxFrames = Math.max(maxFrames, fbanks[b].frames);
            }
            float[] input = new float[batchSize * maxFrames * WESPEAKER_MEL_BINS];
            for (int b = 0; b < batchSize; b++) {
                WespeakerFbankExtractor.Features fbank = fbanks[b];
                for (int f = 0; f < fbank.frames; f++) {
                    System.arraycopy(fbank.data, f * WESPEAKER_MEL_BINS, input, (b * maxFrames + f) * WESPEAKER_MEL_BINS, WESPEAKER_MEL_BINS);
                }
            }
            try (OnnxTensor tensor = OnnxTensor.createTensor(env, FloatBuffer.wrap(input), new long[]{batchSize, maxFrames, WESPEAKER_MEL_BINS});
                 OrtSession.Result result = pyannoteEmbeddingSession.run(Collections.singletonMap(inputName, tensor))) {
                OnnxValue value = result.get(0);
                long[] shape = ((OnnxTensor) value).getInfo().getShape();
                int featureDim = (int) shape[1];
                int featureFrames = (int) shape[2];
                if (featureDim * 2 != 5120) throw new IllegalStateException("Unsupported Pyannote embedding feature dim: " + featureDim);
                if (featureIndex == null || featureIndex.length != featureFrames) {
                    featureIndex = new int[featureFrames];
                    for (int i = 0; i < featureFrames; i++) featureIndex[i] = Math.max(0, Math.min(numFrames - 1, (int) Math.floor(i * numFrames / (double) featureFrames)));
                }
                float[] features = flattenFloat(value.getValue());
                int frameStride = featureDim * featureFrames;
                for (int b = 0; b < batchSize; b++) {
                    int c = batchStart + b;
                    for (int s = 0; s < PYANNOTE_MAX_SPEAKERS_PER_CHUNK; s++) {
                        int fullSum = maskSum(binarized, numFrames, c, s);
                        if (fullSum == 0) continue;
                        byte[] sourceMask = maskSum(cleanBinarized, numFrames, c, s) > minSegFrames ? cleanBinarized : binarized;
                        float[] weights = new float[featureFrames];
                        for (int t = 0; t < featureFrames; t++) weights[t] = pyannoteBinarizedAt(sourceMask, numFrames, c, featureIndex[t], s);
                        double[] stats = maskedStatsPool(features, b * frameStride, featureDim, featureFrames, weights);
                        embeddings[c * PYANNOTE_MAX_SPEAKERS_PER_CHUNK + s] = projectPyannoteStats(stats);
                    }
                }
            }
        }
        return embeddings;
    }

    private float[] paddedChunk(float[] samples, int start) {
        float[] chunk = new float[DIAR_CHUNK_SAMPLES];
        int end = Math.min(samples.length, start + DIAR_CHUNK_SAMPLES);
        if (end > start) System.arraycopy(samples, start, chunk, 0, end - start);
        return chunk;
    }

    private int maskSum(byte[] mask, int numFrames, int chunk, int speaker) {
        int sum = 0;
        for (int f = 0; f < numFrames; f++) sum += pyannoteBinarizedAt(mask, numFrames, chunk, f, speaker);
        return sum;
    }

    private double[] maskedStatsPool(float[] features, int offset, int featureDim, int featureFrames, float[] weights) {
        double[] stats = new double[featureDim * 2];
        double v1 = 1e-8;
        double v2 = 0.0;
        for (int t = 0; t < featureFrames; t++) {
            v1 += weights[t];
            v2 += weights[t] * weights[t];
        }
        for (int d = 0; d < featureDim; d++) {
            int base = offset + d * featureFrames;
            double mean = 0.0;
            for (int t = 0; t < featureFrames; t++) mean += features[base + t] * weights[t];
            mean /= v1;
            stats[d] = mean;
            double variance = 0.0;
            for (int t = 0; t < featureFrames; t++) {
                double diff = features[base + t] - mean;
                variance += diff * diff * weights[t];
            }
            stats[featureDim + d] = Math.sqrt(Math.max(0.0, variance / (v1 - v2 / v1 + 1e-8)));
        }
        return stats;
    }

    private double[] projectPyannoteStats(double[] stats) {
        double[] out = new double[PYANNOTE_EMB_DIM];
        for (int row = 0; row < PYANNOTE_EMB_DIM; row++) {
            double sum = pyannoteAssets.bias[row];
            int base = row * 5120;
            for (int i = 0; i < 5120; i++) sum += stats[i] * pyannoteAssets.weight[base + i];
            out[row] = sum;
        }
        return out;
    }

    private int[] clusterPyannoteEmbeddings(double[][] embeddings, byte[] binarized, int chunks, int numFrames) {
        List<Integer> trainIndices = new ArrayList<>();
        for (int c = 0; c < chunks; c++) {
            for (int s = 0; s < PYANNOTE_MAX_SPEAKERS_PER_CHUNK; s++) {
                int singleActive = 0;
                for (int f = 0; f < numFrames; f++) {
                    int count = 0;
                    for (int k = 0; k < PYANNOTE_MAX_SPEAKERS_PER_CHUNK; k++) count += pyannoteBinarizedAt(binarized, numFrames, c, f, k);
                    if (count == 1) singleActive += pyannoteBinarizedAt(binarized, numFrames, c, f, s);
                }
                int index = c * PYANNOTE_MAX_SPEAKERS_PER_CHUNK + s;
                if (singleActive >= 0.2 * numFrames && isFiniteEmbedding(embeddings[index])) trainIndices.add(index);
            }
        }
        if (trainIndices.size() < 2) return localPyannoteClusters(binarized, chunks, numFrames);
        double[][] train = new double[trainIndices.size()][PYANNOTE_EMB_DIM];
        double[][] normed = new double[trainIndices.size()][PYANNOTE_EMB_DIM];
        for (int i = 0; i < trainIndices.size(); i++) {
            train[i] = embeddings[trainIndices.get(i)];
            normed[i] = l2Double(train[i]);
        }
        int[] ahcLabels = hierarchicalCentroidLabels(normed, PYANNOTE_DEFAULT_THRESHOLD);
        double[][] embPlda = pldaTransformBatch(xvecTransformBatch(train));
        VbxResult vbx = vbxCluster(embPlda, pyannoteAssets.pldaPsi, ahcLabels);
        List<Integer> activeClusters = new ArrayList<>();
        for (int k = 0; k < vbx.pi.length; k++) if (vbx.pi[k] > 1e-7) activeClusters.add(k);
        if (activeClusters.isEmpty()) activeClusters.add(0);
        double[][] centroids = new double[activeClusters.size()][PYANNOTE_EMB_DIM];
        for (int out = 0; out < activeClusters.size(); out++) {
            int cluster = activeClusters.get(out);
            double weightSum = 0.0;
            for (int i = 0; i < train.length; i++) {
                double weight = vbx.gamma[i][cluster];
                weightSum += weight;
                for (int d = 0; d < PYANNOTE_EMB_DIM; d++) centroids[out][d] += train[i][d] * weight;
            }
            for (int d = 0; d < PYANNOTE_EMB_DIM; d++) centroids[out][d] /= Math.max(1e-8, weightSum);
        }
        int[] hard = new int[chunks * PYANNOTE_MAX_SPEAKERS_PER_CHUNK];
        java.util.Arrays.fill(hard, -2);
        for (int c = 0; c < chunks; c++) {
            double[] cost = new double[PYANNOTE_MAX_SPEAKERS_PER_CHUNK * centroids.length];
            double minCost = Double.POSITIVE_INFINITY;
            for (int s = 0; s < PYANNOTE_MAX_SPEAKERS_PER_CHUNK; s++) {
                double[] embedding = embeddings[c * PYANNOTE_MAX_SPEAKERS_PER_CHUNK + s];
                for (int k = 0; k < centroids.length; k++) {
                    double value = 2.0 - cosineDistance(embedding, centroids[k]);
                    cost[s * centroids.length + k] = value;
                    minCost = Math.min(minCost, value);
                }
            }
            for (int s = 0; s < PYANNOTE_MAX_SPEAKERS_PER_CHUNK; s++) {
                if (maskSum(binarized, numFrames, c, s) > 0 && isFiniteEmbedding(embeddings[c * PYANNOTE_MAX_SPEAKERS_PER_CHUNK + s])) continue;
                for (int k = 0; k < centroids.length; k++) cost[s * centroids.length + k] = minCost - 1.0;
            }
            int[] assigned = constrainedArgmax(cost, PYANNOTE_MAX_SPEAKERS_PER_CHUNK, centroids.length);
            for (int s = 0; s < PYANNOTE_MAX_SPEAKERS_PER_CHUNK; s++) {
                hard[c * PYANNOTE_MAX_SPEAKERS_PER_CHUNK + s] = maskSum(binarized, numFrames, c, s) > 0 ? assigned[s] : -2;
            }
        }
        return hard;
    }

    private int[] hierarchicalCentroidLabels(double[][] points, double threshold) {
        int n = points.length;
        if (n <= 1) return new int[n];
        int[] parent = new int[n];
        List<ClusterNode> clusters = new ArrayList<>();
        for (int i = 0; i < n; i++) {
            parent[i] = i;
            clusters.add(new ClusterNode(i, points[i].clone(), 1, new int[]{i}));
        }
        int nextId = n;
        while (clusters.size() > 1) {
            int bestI = -1;
            int bestJ = -1;
            double bestDist = Double.POSITIVE_INFINITY;
            for (int i = 0; i < clusters.size() - 1; i++) {
                for (int j = i + 1; j < clusters.size(); j++) {
                    double dist = Math.sqrt(squaredDistance(clusters.get(i).centroid, clusters.get(j).centroid));
                    if (dist < bestDist) {
                        bestDist = dist;
                        bestI = i;
                        bestJ = j;
                    }
                }
            }
            ClusterNode a = clusters.get(bestI);
            ClusterNode b = clusters.get(bestJ);
            if (bestDist <= threshold) {
                for (int left : a.members) for (int right : b.members) union(parent, left, right);
            }
            int size = a.size + b.size;
            double[] centroid = new double[a.centroid.length];
            for (int d = 0; d < centroid.length; d++) centroid[d] = (a.centroid[d] * a.size + b.centroid[d] * b.size) / size;
            int[] members = new int[a.members.length + b.members.length];
            System.arraycopy(a.members, 0, members, 0, a.members.length);
            System.arraycopy(b.members, 0, members, a.members.length, b.members.length);
            if (bestI < bestJ) {
                clusters.remove(bestJ);
                clusters.remove(bestI);
            } else {
                clusters.remove(bestI);
                clusters.remove(bestJ);
            }
            clusters.add(new ClusterNode(nextId++, centroid, size, members));
        }
        int[] labels = new int[n];
        Map<Integer, Integer> remap = new HashMap<>();
        int next = 0;
        for (int i = 0; i < n; i++) {
            int root = root(parent, i);
            Integer label = remap.get(root);
            if (label == null) {
                label = next++;
                remap.put(root, label);
            }
            labels[i] = label;
        }
        return labels;
    }

    private static final class ClusterNode {
        final int id;
        final double[] centroid;
        final int size;
        final int[] members;

        ClusterNode(int id, double[] centroid, int size, int[] members) {
            this.id = id;
            this.centroid = centroid;
            this.size = size;
            this.members = members;
        }
    }

    private double[][] xvecTransformBatch(double[][] vectors) {
        double[][] out = new double[vectors.length][PYANNOTE_PLDA_DIM];
        double sqrtIn = Math.sqrt(PYANNOTE_EMB_DIM);
        double sqrtOut = Math.sqrt(PYANNOTE_PLDA_DIM);
        for (int row = 0; row < vectors.length; row++) {
            double[] centered = new double[PYANNOTE_EMB_DIM];
            for (int i = 0; i < PYANNOTE_EMB_DIM; i++) centered[i] = vectors[row][i] - pyannoteAssets.mean1[i];
            double[] normed = l2Double(centered);
            double[] transformed = new double[PYANNOTE_PLDA_DIM];
            for (int j = 0; j < PYANNOTE_PLDA_DIM; j++) {
                double sum = 0.0;
                for (int i = 0; i < PYANNOTE_EMB_DIM; i++) sum += normed[i] * sqrtIn * pyannoteAssets.lda[i * PYANNOTE_PLDA_DIM + j];
                transformed[j] = sum - pyannoteAssets.mean2[j];
            }
            double[] normalized = l2Double(transformed);
            for (int j = 0; j < PYANNOTE_PLDA_DIM; j++) out[row][j] = normalized[j] * sqrtOut;
        }
        return out;
    }

    private double[][] pldaTransformBatch(double[][] vectors) {
        double[][] out = new double[vectors.length][PYANNOTE_PLDA_DIM];
        for (int row = 0; row < vectors.length; row++) {
            for (int k = 0; k < PYANNOTE_PLDA_DIM; k++) {
                double sum = 0.0;
                int base = k * PYANNOTE_PLDA_DIM;
                for (int i = 0; i < PYANNOTE_PLDA_DIM; i++) sum += (vectors[row][i] - pyannoteAssets.mu[i]) * pyannoteAssets.pldaTr[base + i];
                out[row][k] = sum;
            }
        }
        return out;
    }

    private static final class VbxResult {
        final double[][] gamma;
        final double[] pi;

        VbxResult(double[][] gamma, double[] pi) {
            this.gamma = gamma;
            this.pi = pi;
        }
    }

    private VbxResult vbxCluster(double[][] features, double[] pldaPsi, int[] initialLabels) {
        int frames = features.length;
        int dim = features[0].length;
        int clusters = 0;
        for (int label : initialLabels) clusters = Math.max(clusters, label + 1);
        double[][] gamma = new double[frames][clusters];
        for (int t = 0; t < frames; t++) {
            double[] logits = new double[clusters];
            logits[Math.max(0, initialLabels[t])] = 7.0;
            double max = Double.NEGATIVE_INFINITY;
            for (double v : logits) max = Math.max(max, v);
            double sum = 0.0;
            for (int k = 0; k < clusters; k++) {
                gamma[t][k] = Math.exp(logits[k] - max);
                sum += gamma[t][k];
            }
            for (int k = 0; k < clusters; k++) gamma[t][k] /= sum;
        }
        double[] pi = new double[clusters];
        java.util.Arrays.fill(pi, 1.0 / clusters);
        double[] g = new double[frames];
        double[][] rho = new double[frames][dim];
        double[] sqrtPsi = new double[dim];
        for (int d = 0; d < dim; d++) sqrtPsi[d] = Math.sqrt(Math.max(0.0, pldaPsi[d]));
        double logTwoPi = Math.log(2.0 * Math.PI);
        for (int t = 0; t < frames; t++) {
            double norm = 0.0;
            for (int d = 0; d < dim; d++) {
                double value = features[t][d];
                norm += value * value;
                rho[t][d] = value * sqrtPsi[d];
            }
            g[t] = -0.5 * (norm + dim * logTwoPi);
        }
        double previousElbo = Double.NEGATIVE_INFINITY;
        for (int iteration = 0; iteration < 20; iteration++) {
            double[] gammaSum = new double[clusters];
            for (int t = 0; t < frames; t++) for (int k = 0; k < clusters; k++) gammaSum[k] += gamma[t][k];
            double[][] invL = new double[clusters][dim];
            double[][] alpha = new double[clusters][dim];
            for (int k = 0; k < clusters; k++) {
                for (int d = 0; d < dim; d++) invL[k][d] = 1.0 / (1.0 + (PYANNOTE_FA / PYANNOTE_FB) * gammaSum[k] * pldaPsi[d]);
                for (int t = 0; t < frames; t++) {
                    double weight = gamma[t][k];
                    if (weight <= 0) continue;
                    for (int d = 0; d < dim; d++) alpha[k][d] += weight * rho[t][d];
                }
                for (int d = 0; d < dim; d++) alpha[k][d] *= (PYANNOTE_FA / PYANNOTE_FB) * invL[k][d];
            }
            double elbo = 0.0;
            for (int t = 0; t < frames; t++) {
                double[] logProb = new double[clusters];
                double maxLog = Double.NEGATIVE_INFINITY;
                for (int k = 0; k < clusters; k++) {
                    double dot = 0.0;
                    double penalty = 0.0;
                    for (int d = 0; d < dim; d++) {
                        dot += rho[t][d] * alpha[k][d];
                        penalty += (invL[k][d] + alpha[k][d] * alpha[k][d]) * pldaPsi[d];
                    }
                    logProb[k] = PYANNOTE_FA * (dot - 0.5 * penalty + g[t]) + Math.log(pi[k] + 1e-8);
                    maxLog = Math.max(maxLog, logProb[k]);
                }
                double sumExp = 0.0;
                for (int k = 0; k < clusters; k++) sumExp += Math.exp(logProb[k] - maxLog);
                double logPx = maxLog + Math.log(sumExp);
                elbo += logPx;
                for (int k = 0; k < clusters; k++) gamma[t][k] = Math.exp(logProb[k] - logPx);
            }
            java.util.Arrays.fill(pi, 0.0);
            for (int t = 0; t < frames; t++) for (int k = 0; k < clusters; k++) pi[k] += gamma[t][k];
            double piSum = 0.0;
            for (double v : pi) piSum += v;
            for (int k = 0; k < clusters; k++) pi[k] /= Math.max(1e-12, piSum);
            for (int k = 0; k < clusters; k++) {
                for (int d = 0; d < dim; d++) {
                    elbo += PYANNOTE_FB * 0.5 * (Math.log(invL[k][d]) - invL[k][d] - alpha[k][d] * alpha[k][d] + 1.0);
                }
            }
            if (iteration > 0 && elbo - previousElbo < 1e-4) break;
            previousElbo = elbo;
        }
        return new VbxResult(gamma, pi);
    }

    private boolean isFiniteEmbedding(double[] vector) {
        return vector != null && vector.length > 0 && Double.isFinite(vector[0]);
    }

    private double[] l2Double(double[] vector) {
        double norm = 0.0;
        for (double v : vector) norm += v * v;
        norm = Math.sqrt(Math.max(norm, 1e-20));
        double[] out = new double[vector.length];
        for (int i = 0; i < vector.length; i++) out[i] = vector[i] / norm;
        return out;
    }

    private double cosineDistance(double[] a, double[] b) {
        if (!isFiniteEmbedding(a)) return 1.0;
        double dot = 0.0;
        double na = 0.0;
        double nb = 0.0;
        int n = Math.min(a.length, b.length);
        for (int i = 0; i < n; i++) {
            dot += a[i] * b[i];
            na += a[i] * a[i];
            nb += b[i] * b[i];
        }
        return 1.0 - dot / (Math.sqrt(na) * Math.sqrt(nb) + 1e-10);
    }

    private int[] constrainedArgmax(double[] cost, int rows, int cols) {
        int[] best = new int[rows];
        java.util.Arrays.fill(best, -2);
        if (cols <= 0) return best;
        double bestScore = Double.NEGATIVE_INFINITY;
        if (cols == 1) {
            for (int row = 0; row < rows; row++) {
                if (cost[row] > bestScore) {
                    bestScore = cost[row];
                    java.util.Arrays.fill(best, -2);
                    best[row] = 0;
                }
            }
            return best;
        }
        if (cols == 2) {
            for (int r0 = 0; r0 < rows; r0++) {
                for (int r1 = 0; r1 < rows; r1++) {
                    if (r1 == r0) continue;
                    double score = cost[r0 * cols] + cost[r1 * cols + 1];
                    if (score > bestScore) {
                        bestScore = score;
                        java.util.Arrays.fill(best, -2);
                        best[r0] = 0;
                        best[r1] = 1;
                    }
                }
            }
            return best;
        }
        for (int c0 = 0; c0 < cols; c0++) {
            for (int c1 = 0; c1 < cols; c1++) {
                if (c1 == c0) continue;
                int c2 = -1;
                double c2Score = Double.NEGATIVE_INFINITY;
                for (int col = 0; col < cols; col++) {
                    if (col == c0 || col == c1) continue;
                    double value = cost[2 * cols + col];
                    if (value > c2Score) {
                        c2Score = value;
                        c2 = col;
                    }
                }
                if (c2 < 0) continue;
                double score = cost[c0] + cost[cols + c1] + cost[2 * cols + c2];
                if (score > bestScore) {
                    bestScore = score;
                    best[0] = c0;
                    best[1] = c1;
                    best[2] = c2;
                }
            }
        }
        return best;
    }

    private int root(int[] parent, int value) {
        int root = value;
        while (parent[root] != root) root = parent[root];
        while (parent[value] != value) {
            int next = parent[value];
            parent[value] = root;
            value = next;
        }
        return root;
    }

    private void union(int[] parent, int a, int b) {
        int ra = root(parent, a);
        int rb = root(parent, b);
        if (ra == rb) return;
        if (ra < rb) parent[rb] = ra;
        else parent[ra] = rb;
    }

    private int[] localPyannoteClusters(byte[] binarized, int chunks, int numFrames) {
        int[] hard = new int[chunks * PYANNOTE_MAX_SPEAKERS_PER_CHUNK];
        for (int c = 0; c < chunks; c++) {
            for (int s = 0; s < PYANNOTE_MAX_SPEAKERS_PER_CHUNK; s++) {
                int active = 0;
                for (int f = 0; f < numFrames; f++) active += pyannoteBinarizedAt(binarized, numFrames, c, f, s);
                hard[c * PYANNOTE_MAX_SPEAKERS_PER_CHUNK + s] = active > 0 ? s : -2;
            }
        }
        return hard;
    }

    private int[] canonicalizePyannoteClusters(int[] hardClusters, byte[] binarized, int chunks, int numFrames) {
        List<Integer> seen = new ArrayList<>();
        for (int cluster : hardClusters) {
            if (cluster < 0 || seen.contains(cluster)) continue;
            seen.add(cluster);
        }
        List<int[]> keys = new ArrayList<>();
        for (Integer cluster : seen) {
            int bestChunk = Integer.MAX_VALUE;
            int bestFrame = Integer.MAX_VALUE;
            int bestSpeaker = Integer.MAX_VALUE;
            for (int c = 0; c < chunks; c++) {
                for (int s = 0; s < PYANNOTE_MAX_SPEAKERS_PER_CHUNK; s++) {
                    if (hardClusters[c * PYANNOTE_MAX_SPEAKERS_PER_CHUNK + s] != cluster) continue;
                    for (int f = 0; f < numFrames; f++) {
                        if (pyannoteBinarizedAt(binarized, numFrames, c, f, s) == 0) continue;
                        if (c < bestChunk || (c == bestChunk && f < bestFrame) || (c == bestChunk && f == bestFrame && s < bestSpeaker)) {
                            bestChunk = c;
                            bestFrame = f;
                            bestSpeaker = s;
                        }
                        break;
                    }
                }
            }
            keys.add(new int[]{cluster, bestChunk, bestFrame, bestSpeaker});
        }
        keys.sort((a, b) -> {
            for (int i = 1; i < 4; i++) if (a[i] != b[i]) return Integer.compare(a[i], b[i]);
            return Integer.compare(a[0], b[0]);
        });
        Map<Integer, Integer> remap = new HashMap<>();
        for (int i = 0; i < keys.size(); i++) remap.put(keys.get(i)[0], i);
        int[] out = new int[hardClusters.length];
        for (int i = 0; i < hardClusters.length; i++) {
            int cluster = hardClusters[i];
            out[i] = cluster >= 0 ? remap.getOrDefault(cluster, -2) : -2;
        }
        return out;
    }

    private List<DiarizationResult.Turn> reconstructPyannoteSegments(byte[] binarized, int[] hardClusters, float[] countData, int chunks, int numFrames, double duration) {
        int numClusters = 0;
        for (int hard : hardClusters) numClusters = Math.max(numClusters, hard + 1);
        if (numClusters <= 0) return new ArrayList<>();
        int outFrames = countData.length;
        float[] activations = new float[outFrames * numClusters];
        for (int c = 0; c < chunks; c++) {
            int startFrame = closestPyannoteFrame(c + 0.5 * PYANNOTE_RF_DURATION);
            for (int f = 0; f < numFrames; f++) {
                int out = startFrame + f;
                if (out < 0 || out >= outFrames) continue;
                for (int k = 0; k < numClusters; k++) {
                    int value = 0;
                    for (int s = 0; s < PYANNOTE_MAX_SPEAKERS_PER_CHUNK; s++) {
                        if (hardClusters[c * PYANNOTE_MAX_SPEAKERS_PER_CHUNK + s] == k) {
                            value = Math.max(value, pyannoteBinarizedAt(binarized, numFrames, c, f, s));
                        }
                    }
                    activations[out * numClusters + k] += value;
                }
            }
        }

        byte[] binary = new byte[outFrames * numClusters];
        for (int t = 0; t < outFrames; t++) {
            if (Math.min(1, Math.max(0, Math.round(countData[t]))) < 1) continue;
            int bestSpeaker = 0;
            float bestValue = Float.NEGATIVE_INFINITY;
            for (int k = 0; k < numClusters; k++) {
                float value = activations[t * numClusters + k];
                if (value > bestValue) {
                    bestValue = value;
                    bestSpeaker = k;
                }
            }
            binary[t * numClusters + bestSpeaker] = 1;
        }

        List<DiarizationResult.Turn> raw = new ArrayList<>();
        for (int speaker = 0; speaker < numClusters; speaker++) {
            boolean active = false;
            double start = PYANNOTE_RF_START + 0.5 * PYANNOTE_RF_DURATION;
            double lastTime = start;
            for (int t = 0; t < outFrames; t++) {
                double time = PYANNOTE_RF_START + t * PYANNOTE_RF_STEP + 0.5 * PYANNOTE_RF_DURATION;
                lastTime = time;
                boolean value = binary[t * numClusters + speaker] > 0;
                if (active && !value) {
                    raw.add(new DiarizationResult.Turn(start, time, speaker));
                    active = false;
                } else if (!active && value) {
                    start = time;
                    active = true;
                }
            }
            if (active) raw.add(new DiarizationResult.Turn(start, lastTime, speaker));
        }
        raw.sort((a, b) -> Double.compare(a.start, b.start));
        List<DiarizationResult.Turn> clipped = new ArrayList<>();
        Map<Integer, Integer> speakerMap = new HashMap<>();
        int nextSpeaker = 0;
        for (DiarizationResult.Turn turn : raw) {
            if (turn.start >= duration) continue;
            double start = Math.max(0, turn.start);
            double end = turn.end;
            if (end > start) {
                Integer speaker = speakerMap.get(turn.speaker);
                if (speaker == null) {
                    speaker = nextSpeaker++;
                    speakerMap.put(turn.speaker, speaker);
                }
                clipped.add(new DiarizationResult.Turn(start, end, speaker));
            }
        }
        return mergeDiarSegmentsWithGap(clipped, PYANNOTE_MERGE_GAP_SECONDS);
    }

    private List<DiarizationResult.Turn> mergeDiarSegmentsWithGap(List<DiarizationResult.Turn> segments, double maxGap) {
        List<DiarizationResult.Turn> merged = new ArrayList<>();
        for (DiarizationResult.Turn turn : segments) {
            if (!merged.isEmpty()) {
                DiarizationResult.Turn prev = merged.get(merged.size() - 1);
                if (prev.speaker == turn.speaker && turn.start - prev.end <= maxGap) {
                    merged.set(merged.size() - 1, new DiarizationResult.Turn(prev.start, Math.max(prev.end, turn.end), prev.speaker));
                    continue;
                }
            }
            merged.add(turn);
        }
        return merged;
    }

    private float[] flattenFloat(Object value) {
        if (value instanceof float[]) return (float[]) value;
        if (value instanceof float[][][]) {
            float[][][] data = (float[][][]) value;
            int n = data.length * data[0].length * data[0][0].length;
            float[] out = new float[n];
            int p = 0;
            for (float[][] batch : data) for (float[] row : batch) for (float v : row) out[p++] = v;
            return out;
        }
        if (value instanceof float[][]) {
            float[][] data = (float[][]) value;
            int n = data.length * data[0].length;
            float[] out = new float[n];
            int p = 0;
            for (float[] row : data) for (float v : row) out[p++] = v;
            return out;
        }
        throw new IllegalStateException("Unsupported ORT float output: " + value.getClass());
    }

    private static final class FbankWindow {
        final CamppFbankExtractor.Features features;
        final int startFrame;
        final int frameCount;
        final double start;
        final double end;

        FbankWindow(CamppFbankExtractor.Features features, int startFrame, int frameCount, double start, double end) {
            this.features = features;
            this.startFrame = startFrame;
            this.frameCount = frameCount;
            this.start = start;
            this.end = end;
        }
    }

    private List<WindowEmbedding> extractCamppEmbeddings(float[] samples, List<double[]> regions) throws Exception {
        List<FbankWindow> windows = new ArrayList<>();
        for (double[] region : regions) {
            int start = Math.max(0, (int) Math.floor(region[0] * SAMPLE_RATE));
            int end = Math.min(samples.length, (int) Math.ceil(region[1] * SAMPLE_RATE));
            if (end - start < 400) continue;
            float[] regionAudio = new float[end - start];
            System.arraycopy(samples, start, regionAudio, 0, regionAudio.length);
            CamppFbankExtractor.Features regionFbank = camppFbank.compute(regionAudio);
            if (regionFbank.frames < 10) continue;
            if (regionFbank.frames < CAMPP_WINDOW_FRAMES) {
                windows.add(new FbankWindow(regionFbank, 0, regionFbank.frames, region[0], region[1]));
                continue;
            }
            int position = 0;
            while (position + CAMPP_WINDOW_FRAMES < regionFbank.frames) {
                double windowStart = region[0] + position * 0.01;
                windows.add(new FbankWindow(regionFbank, position, CAMPP_WINDOW_FRAMES, windowStart, windowStart + CAMPP_WINDOW_SECONDS));
                position += CAMPP_STEP_FRAMES;
            }
            int tail = Math.max(0, regionFbank.frames - CAMPP_WINDOW_FRAMES);
            double tailStart = region[0] + tail * 0.01;
            windows.add(new FbankWindow(regionFbank, tail, CAMPP_WINDOW_FRAMES, tailStart, tailStart + CAMPP_WINDOW_SECONDS));
        }
        List<WindowEmbedding> out = new ArrayList<>();
        for (int batchStart = 0; batchStart < windows.size(); batchStart += CAMPP_BATCH) {
            int batchSize = Math.min(CAMPP_BATCH, windows.size() - batchStart);
            int maxFrames = 0;
            for (int b = 0; b < batchSize; b++) {
                maxFrames = Math.max(maxFrames, windows.get(batchStart + b).frameCount);
            }
            if (maxFrames < 10) continue;
            float[] input = new float[batchSize * maxFrames * CAMPP_MEL_BINS];
            for (int b = 0; b < batchSize; b++) {
                FbankWindow window = windows.get(batchStart + b);
                for (int f = 0; f < window.frameCount; f++) {
                    System.arraycopy(
                            window.features.data,
                            (window.startFrame + f) * CAMPP_MEL_BINS,
                            input,
                            (b * maxFrames + f) * CAMPP_MEL_BINS,
                            CAMPP_MEL_BINS);
                }
            }
            try (OnnxTensor feats = OnnxTensor.createTensor(env, FloatBuffer.wrap(input), new long[]{batchSize, maxFrames, CAMPP_MEL_BINS});
                 OrtSession.Result result = camppSession.run(Collections.singletonMap("feats", feats))) {
                OnnxValue value = result.get("embs").orElse(result.get(0));
                float[][] vectors = (float[][]) value.getValue();
                for (int b = 0; b < batchSize; b++) {
                    FbankWindow window = windows.get(batchStart + b);
                    out.add(new WindowEmbedding(l2(vectors[b]), window.start, window.end));
                }
            }
        }
        return out;
    }

    private List<double[]> speechRegions(float[] samples, List<SpeechSegment> segments) {
        List<double[]> regions = new ArrayList<>();
        if (segments != null) {
            for (SpeechSegment segment : segments) {
                if (segment.end > segment.start) regions.add(new double[]{
                        segment.start / (double) SAMPLE_RATE,
                        segment.end / (double) SAMPLE_RATE
                });
            }
        }
        if (regions.isEmpty() && samples != null && samples.length > 0) {
            regions.add(new double[]{0, samples.length / (double) SAMPLE_RATE});
        }
        return regions;
    }

    private int[] senkoCluster(List<WindowEmbedding> embeddings, double durationSeconds) {
        int n = embeddings.size();
        if (n < 10) {
            int[] labels = new int[n];
            for (int i = 0; i < n; i++) labels[i] = 1;
            rerankLabels(labels, embeddings);
            return labels;
        }
        int[] labels;
        if (durationSeconds < 1200.0) {
            labels = senkoSpectral(embeddings, 1, Math.min(15, n), 0.012, 6);
        } else {
            // Android native does not ship UMAP/HDBSCAN yet. Keep a deterministic
            // spectral path instead of the previous online centroid heuristic.
            labels = senkoSpectral(embeddings, 1, Math.min(15, n), 0.012, 6);
        }
        lastSenkoSpectralLabels = labels.clone();
        filterMinorClusters(labels, embeddings);
        lastSenkoAfterFilterLabels = labels.clone();
        mergeCloseClusters(labels, embeddings);
        lastSenkoAfterMergeLabels = labels.clone();
        rerankLabels(labels, embeddings);
        return labels;
    }

    private int[] senkoSpectral(List<WindowEmbedding> embeddings, int minSpeakers, int maxSpeakers, double pval, int minPnum) {
        int n = embeddings.size();
        if (n <= 1) return new int[n];
        double[][] affinity = new double[n][n];
        for (int i = 0; i < n; i++) {
            for (int j = i; j < n; j++) {
                double cos = cosine(embeddings.get(i).vector, embeddings.get(j).vector);
                affinity[i][j] = cos;
                affinity[j][i] = cos;
            }
        }
        int nElems = (int) ((1.0 - pval) * n);
        nElems = Math.min(nElems, n - minPnum);
        nElems = Math.max(nElems, 0);
        for (int i = 0; i < n; i++) {
            int[] order = argsort(affinity[i]);
            for (int k = 0; k < nElems; k++) affinity[i][order[k]] = 0.0;
        }
        for (int i = 0; i < n; i++) {
            affinity[i][i] = 0.0;
            for (int j = i + 1; j < n; j++) {
                double value = 0.5 * (affinity[i][j] + affinity[j][i]);
                affinity[i][j] = value;
                affinity[j][i] = value;
            }
        }
        double[][] laplacian = new double[n][n];
        for (int i = 0; i < n; i++) {
            double degree = 0.0;
            for (int j = 0; j < n; j++) degree += Math.abs(affinity[i][j]);
            laplacian[i][i] = degree;
            for (int j = 0; j < n; j++) {
                if (i != j) laplacian[i][j] = -affinity[i][j];
            }
        }

        RealMatrix matrix = MatrixUtils.createRealMatrix(laplacian);
        EigenDecomposition eig = new EigenDecomposition(matrix);
        double[] values = eig.getRealEigenvalues();
        int[] order = argsort(values);
        int upper = Math.min(maxSpeakers + 1, n);
        int lower = Math.max(1, minSpeakers);
        int speakers = 1;
        double bestGap = Double.NEGATIVE_INFINITY;
        for (int idx = lower - 1; idx < upper - 1; idx++) {
            double gap = values[order[idx + 1]] - values[order[idx]];
            if (gap > bestGap) {
                bestGap = gap;
                speakers = idx + 1;
            }
        }
        speakers = Math.max(1, Math.min(speakers, n));
        lastSenkoSpeakers = speakers;
        lastSenkoEigenvalues = new double[Math.min(upper, n)];
        for (int i = 0; i < lastSenkoEigenvalues.length; i++) {
            lastSenkoEigenvalues[i] = values[order[i]];
        }
        double[][] spectral = new double[n][speakers];
        for (int k = 0; k < speakers; k++) {
            double[] vector = eig.getEigenvector(order[k]).toArray();
            for (int i = 0; i < n; i++) spectral[i][k] = vector[i];
        }
        return kmeansPlusPlus(spectral, speakers, 300, 1, 0L);
    }

    private int[] kmeansPlusPlus(double[][] points, int clusters, int iterations, int nInit, long seed) {
        int n = points.length;
        int dim = points[0].length;
        if (clusters <= 1 || n <= 1) return new int[n];
        int[] bestLabels = new int[n];
        double bestInertia = Double.POSITIVE_INFINITY;
        for (int init = 0; init < Math.max(1, nInit); init++) {
            NumpyMt19937 rng = new NumpyMt19937((int) (seed + init));
            double[][] centers = initKMeansPlusPlus(points, clusters, rng);
            int[] labels = new int[n];
            double previousInertia = Double.POSITIVE_INFINITY;
            for (int iter = 0; iter < iterations; iter++) {
                boolean changed = false;
                double inertia = 0.0;
                for (int i = 0; i < n; i++) {
                    int best = 0;
                    double bestDist = Double.POSITIVE_INFINITY;
                    for (int k = 0; k < clusters; k++) {
                        double dist = squaredDistance(points[i], centers[k]);
                        if (dist < bestDist) {
                            bestDist = dist;
                            best = k;
                        }
                    }
                    inertia += bestDist;
                    if (labels[i] != best) {
                        labels[i] = best;
                        changed = true;
                    }
                }
                double[][] next = new double[clusters][dim];
                int[] counts = new int[clusters];
                for (int i = 0; i < n; i++) {
                    counts[labels[i]]++;
                    for (int d = 0; d < dim; d++) next[labels[i]][d] += points[i][d];
                }
                for (int k = 0; k < clusters; k++) {
                    if (counts[k] == 0) {
                        int farthest = farthestPoint(points, centers, rng);
                        System.arraycopy(points[farthest], 0, next[k], 0, dim);
                    } else {
                        for (int d = 0; d < dim; d++) next[k][d] /= counts[k];
                    }
                }
                centers = next;
                if (!changed || Math.abs(previousInertia - inertia) <= 1e-4 * Math.max(1.0, previousInertia)) {
                    previousInertia = inertia;
                    break;
                }
                previousInertia = inertia;
            }
            double inertia = inertia(points, centers);
            if (inertia < bestInertia) {
                bestInertia = inertia;
                bestLabels = assignLabels(points, centers);
            }
        }
        return relabelByFirstOccurrence(bestLabels);
    }

    private double[][] initKMeansPlusPlus(double[][] points, int clusters, NumpyMt19937 rng) {
        int n = points.length;
        int dim = points[0].length;
        double[][] centers = new double[clusters][dim];
        double[] closest = new double[n];
        java.util.Arrays.fill(closest, Double.POSITIVE_INFINITY);
        int first = rng.randomInt(n);
        System.arraycopy(points[first], 0, centers[0], 0, dim);
        updateClosest(points, centers[0], closest);
        int trials = 2 + (int) Math.log(clusters);
        for (int k = 1; k < clusters; k++) {
            double currentPotential = 0.0;
            for (double v : closest) currentPotential += v;
            if (currentPotential <= 0.0 || !Double.isFinite(currentPotential)) {
                System.arraycopy(points[rng.randomInt(n)], 0, centers[k], 0, dim);
                updateClosest(points, centers[k], closest);
                continue;
            }
            int bestCandidate = 0;
            double bestPotential = Double.POSITIVE_INFINITY;
            for (int trial = 0; trial < trials; trial++) {
                double threshold = rng.randomDouble() * currentPotential;
                double cumulative = 0.0;
                int candidate = n - 1;
                for (int i = 0; i < n; i++) {
                    cumulative += closest[i];
                    if (cumulative >= threshold) {
                        candidate = i;
                        break;
                    }
                }
                double potential = 0.0;
                for (int i = 0; i < n; i++) {
                    potential += Math.min(closest[i], squaredDistance(points[i], points[candidate]));
                }
                if (potential < bestPotential) {
                    bestPotential = potential;
                    bestCandidate = candidate;
                }
            }
            System.arraycopy(points[bestCandidate], 0, centers[k], 0, dim);
            updateClosest(points, centers[k], closest);
        }
        return centers;
    }

    private void updateClosest(double[][] points, double[] center, double[] closest) {
        for (int i = 0; i < points.length; i++) {
            double dist = squaredDistance(points[i], center);
            if (dist < closest[i]) closest[i] = dist;
        }
    }

    private int farthestPoint(double[][] points, double[][] centers, NumpyMt19937 rng) {
        int best = rng.randomInt(points.length);
        double bestDist = Double.NEGATIVE_INFINITY;
        for (int i = 0; i < points.length; i++) {
            double dist = Double.POSITIVE_INFINITY;
            for (double[] center : centers) {
                boolean empty = true;
                for (double v : center) {
                    if (v != 0.0) {
                        empty = false;
                        break;
                    }
                }
                if (!empty) dist = Math.min(dist, squaredDistance(points[i], center));
            }
            if (dist > bestDist) {
                bestDist = dist;
                best = i;
            }
        }
        return best;
    }

    private static final class NumpyMt19937 {
        private final int[] mt = new int[624];
        private int index = 624;

        NumpyMt19937(int seed) {
            mt[0] = seed;
            for (int i = 1; i < mt.length; i++) {
                int prev = mt[i - 1] ^ (mt[i - 1] >>> 30);
                mt[i] = 1812433253 * prev + i;
            }
        }

        private int nextInt32() {
            if (index >= mt.length) {
                for (int i = 0; i < mt.length; i++) {
                    int y = (mt[i] & 0x80000000) | (mt[(i + 1) % mt.length] & 0x7fffffff);
                    mt[i] = mt[(i + 397) % mt.length] ^ (y >>> 1);
                    if ((y & 1) != 0) mt[i] ^= 0x9908b0df;
                }
                index = 0;
            }
            int y = mt[index++];
            y ^= y >>> 11;
            y ^= (y << 7) & 0x9d2c5680;
            y ^= (y << 15) & 0xefc60000;
            y ^= y >>> 18;
            return y;
        }

        double randomDouble() {
            long a = (nextInt32() >>> 5) & 0x7ffffffL;
            long b = (nextInt32() >>> 6) & 0x3ffffffL;
            return (a * 67108864.0 + b) / 9007199254740992.0;
        }

        int randomInt(int bound) {
            if (bound <= 1) return 0;
            int max = bound - 1;
            int mask = max;
            mask |= mask >>> 1;
            mask |= mask >>> 2;
            mask |= mask >>> 4;
            mask |= mask >>> 8;
            mask |= mask >>> 16;
            int value;
            do {
                value = nextInt32() & mask;
            } while (value > max);
            return value;
        }
    }

    private double inertia(double[][] points, double[][] centers) {
        double total = 0.0;
        for (double[] point : points) {
            double best = Double.POSITIVE_INFINITY;
            for (double[] center : centers) best = Math.min(best, squaredDistance(point, center));
            total += best;
        }
        return total;
    }

    private int[] assignLabels(double[][] points, double[][] centers) {
        int[] labels = new int[points.length];
        for (int i = 0; i < points.length; i++) {
            int best = 0;
            double bestDist = Double.POSITIVE_INFINITY;
            for (int k = 0; k < centers.length; k++) {
                double dist = squaredDistance(points[i], centers[k]);
                if (dist < bestDist) {
                    bestDist = dist;
                    best = k;
                }
            }
            labels[i] = best;
        }
        return labels;
    }

    private int[] relabelByFirstOccurrence(int[] labels) {
        Map<Integer, Integer> remap = new HashMap<>();
        int next = 0;
        int[] out = new int[labels.length];
        for (int i = 0; i < labels.length; i++) {
            Integer value = remap.get(labels[i]);
            if (value == null) {
                value = next++;
                remap.put(labels[i], value);
            }
            out[i] = value;
        }
        return out;
    }

    private double squaredDistance(double[] a, double[] b) {
        double sum = 0.0;
        for (int i = 0; i < a.length; i++) {
            double diff = a[i] - b[i];
            sum += diff * diff;
        }
        return sum;
    }

    private int[] argsort(double[] values) {
        Integer[] order = new Integer[values.length];
        for (int i = 0; i < values.length; i++) order[i] = i;
        java.util.Arrays.sort(order, (a, b) -> Double.compare(values[a], values[b]));
        int[] out = new int[values.length];
        for (int i = 0; i < values.length; i++) out[i] = order[i];
        return out;
    }

    private int[] clusterEmbeddings(List<WindowEmbedding> embeddings) {
        int n = embeddings.size();
        int[] labels = new int[n];
        if (n < 10) return labels;
        List<float[]> centroids = new ArrayList<>();
        List<Integer> counts = new ArrayList<>();
        centroids.add(embeddings.get(0).vector.clone());
        counts.add(1);
        for (int i = 1; i < n; i++) {
            float[] vector = embeddings.get(i).vector;
            int best = 0;
            double bestCos = -2;
            for (int c = 0; c < centroids.size(); c++) {
                double cos = cosine(vector, centroids.get(c));
                if (cos > bestCos) {
                    bestCos = cos;
                    best = c;
                }
            }
            if (bestCos < NEW_SPEAKER_COS && centroids.size() < 8) {
                labels[i] = centroids.size();
                centroids.add(vector.clone());
                counts.add(1);
            } else {
                labels[i] = best;
                updateCentroid(centroids.get(best), counts.get(best), vector);
                counts.set(best, counts.get(best) + 1);
            }
        }
        mergeCloseClusters(labels, embeddings);
        filterMinorClusters(labels, embeddings);
        rerankLabels(labels, embeddings);
        return labels;
    }

    private List<DiarizationResult.Turn> labelsToTurns(List<WindowEmbedding> embeddings, int[] labels) {
        List<DiarizationResult.Turn> raw = new ArrayList<>();
        int current = labels[0];
        double start = embeddings.get(0).start;
        double end = embeddings.get(0).end;
        for (int i = 1; i < embeddings.size(); i++) {
            WindowEmbedding item = embeddings.get(i);
            if (labels[i] == current && item.start - end < 0.01) {
                end = Math.max(end, item.end);
            } else {
                raw.add(new DiarizationResult.Turn(start, end, current));
                current = labels[i];
                start = item.start;
                end = item.end;
            }
        }
        raw.add(new DiarizationResult.Turn(start, end, current));
        return raw;
    }

    private List<DiarizationResult.Turn> senkoPostProcess(List<DiarizationResult.Turn> turns) {
        if (turns.isEmpty()) return turns;
        List<DiarizationResult.Turn> adjusted = new ArrayList<>(turns);
        for (int i = 0; i < adjusted.size() - 1; i++) {
            DiarizationResult.Turn current = adjusted.get(i);
            DiarizationResult.Turn next = adjusted.get(i + 1);
            if (current.end > next.start) {
                double midpoint = (current.end + next.start) / 2.0;
                adjusted.set(i, new DiarizationResult.Turn(current.start, midpoint, current.speaker));
                adjusted.set(i + 1, new DiarizationResult.Turn(midpoint, next.end, next.speaker));
            }
        }
        adjusted = mergeSameSpeaker(adjusted, CAMPP_TURN_MERGE_GAP_SECONDS);
        List<DiarizationResult.Turn> merged = new ArrayList<>();
        for (int i = 0; i < adjusted.size(); i++) {
            DiarizationResult.Turn turn = adjusted.get(i);
            if (turn.end - turn.start <= 0.78) {
                DiarizationResult.Turn prev = merged.isEmpty() ? null : merged.get(merged.size() - 1);
                DiarizationResult.Turn next = i + 1 < adjusted.size() ? adjusted.get(i + 1) : null;
                if (prev != null && next != null && prev.speaker == next.speaker) {
                    merged.set(merged.size() - 1, new DiarizationResult.Turn(prev.start, turn.end, prev.speaker));
                    continue;
                }
                continue;
            }
            merged.add(turn);
        }
        return rerankTurns(mergeAdjacentSameSpeaker(merged));
    }

    private List<DiarizationResult.Turn> mergeSameSpeaker(List<DiarizationResult.Turn> turns, double maxGapSeconds) {
        List<DiarizationResult.Turn> merged = new ArrayList<>();
        for (DiarizationResult.Turn turn : turns) {
            if (!merged.isEmpty()) {
                DiarizationResult.Turn prev = merged.get(merged.size() - 1);
                if (prev.speaker == turn.speaker && turn.start - prev.end <= maxGapSeconds) {
                    merged.set(merged.size() - 1, new DiarizationResult.Turn(prev.start, Math.max(prev.end, turn.end), prev.speaker));
                    continue;
                }
            }
            merged.add(turn);
        }
        return merged;
    }

    private List<DiarizationResult.Turn> mergeAdjacentSameSpeaker(List<DiarizationResult.Turn> turns) {
        List<DiarizationResult.Turn> merged = new ArrayList<>();
        for (DiarizationResult.Turn turn : turns) {
            if (!merged.isEmpty()) {
                DiarizationResult.Turn prev = merged.get(merged.size() - 1);
                if (prev.speaker == turn.speaker) {
                    merged.set(merged.size() - 1, new DiarizationResult.Turn(prev.start, Math.max(prev.end, turn.end), prev.speaker));
                    continue;
                }
            }
            merged.add(turn);
        }
        return merged;
    }

    private List<DiarizationResult.Turn> rerankTurns(List<DiarizationResult.Turn> turns) {
        Map<Integer, Double> durations = new HashMap<>();
        for (DiarizationResult.Turn turn : turns) {
            durations.put(turn.speaker, durations.getOrDefault(turn.speaker, 0.0) + Math.max(0.0, turn.end - turn.start));
        }
        List<Integer> ordered = new ArrayList<>(durations.keySet());
        ordered.sort((a, b) -> Double.compare(durations.get(b), durations.get(a)));
        Map<Integer, Integer> remap = new HashMap<>();
        for (int i = 0; i < ordered.size(); i++) remap.put(ordered.get(i), i);
        List<DiarizationResult.Turn> out = new ArrayList<>();
        for (DiarizationResult.Turn turn : turns) {
            out.add(new DiarizationResult.Turn(turn.start, turn.end, remap.getOrDefault(turn.speaker, 0)));
        }
        return out;
    }

    private void mergeCloseClusters(int[] labels, List<WindowEmbedding> embeddings) {
        boolean changed;
        do {
            changed = false;
            Map<Integer, float[]> centroids = centroids(labels, embeddings);
            Integer mergeA = null;
            Integer mergeB = null;
            double best = MERGE_SPEAKER_COS;
            List<Integer> keys = new ArrayList<>(centroids.keySet());
            for (int i = 0; i < keys.size(); i++) {
                for (int j = i + 1; j < keys.size(); j++) {
                    double cos = cosine(centroids.get(keys.get(i)), centroids.get(keys.get(j)));
                    if (cos > best) {
                        best = cos;
                        mergeA = keys.get(i);
                        mergeB = keys.get(j);
                    }
                }
            }
            if (mergeA != null && mergeB != null) {
                for (int i = 0; i < labels.length; i++) if (labels[i] == mergeB) labels[i] = mergeA;
                changed = true;
            }
        } while (changed);
    }

    private void filterMinorClusters(int[] labels, List<WindowEmbedding> embeddings) {
        Map<Integer, Integer> counts = new HashMap<>();
        for (int label : labels) counts.put(label, counts.getOrDefault(label, 0) + 1);
        Map<Integer, float[]> centroids = centroids(labels, embeddings);
        List<Integer> majorLabels = new ArrayList<>();
        List<Integer> minorLabels = new ArrayList<>();
        for (Map.Entry<Integer, Integer> entry : counts.entrySet()) {
            if (entry.getValue() >= MIN_CLUSTER_SIZE) majorLabels.add(entry.getKey());
            else minorLabels.add(entry.getKey());
        }
        if (minorLabels.isEmpty()) return;
        if (majorLabels.isEmpty()) {
            java.util.Arrays.fill(labels, 0);
            return;
        }
        for (int i = 0; i < labels.length; i++) {
            if (!minorLabels.contains(labels[i])) continue;
            int bestLabel = labels[i];
            double bestCos = -2;
            for (Integer label : majorLabels) {
                double cos = cosine(embeddings.get(i).vector, centroids.get(label));
                if (cos > bestCos) {
                    bestCos = cos;
                    bestLabel = label;
                }
            }
            labels[i] = bestLabel;
        }
    }

    private void rerankLabels(int[] labels, List<WindowEmbedding> embeddings) {
        Map<Integer, Double> durations = new HashMap<>();
        for (int i = 0; i < labels.length; i++) {
            WindowEmbedding w = embeddings.get(i);
            durations.put(labels[i], durations.getOrDefault(labels[i], 0.0) + Math.max(0.0, w.end - w.start));
        }
        List<Integer> ordered = new ArrayList<>(durations.keySet());
        ordered.sort((a, b) -> Double.compare(durations.get(b), durations.get(a)));
        Map<Integer, Integer> remap = new HashMap<>();
        for (int i = 0; i < ordered.size(); i++) remap.put(ordered.get(i), i);
        for (int i = 0; i < labels.length; i++) labels[i] = remap.getOrDefault(labels[i], 0);
    }

    private Map<Integer, float[]> centroids(int[] labels, List<WindowEmbedding> embeddings) {
        Map<Integer, float[]> sums = new HashMap<>();
        Map<Integer, Integer> counts = new HashMap<>();
        for (int i = 0; i < labels.length; i++) {
            int label = labels[i];
            float[] vector = embeddings.get(i).vector;
            float[] sum = sums.get(label);
            if (sum == null) {
                sum = new float[vector.length];
                sums.put(label, sum);
            }
            for (int d = 0; d < sum.length; d++) sum[d] += vector[d];
            counts.put(label, counts.getOrDefault(label, 0) + 1);
        }
        for (Map.Entry<Integer, float[]> entry : sums.entrySet()) {
            int count = Math.max(1, counts.getOrDefault(entry.getKey(), 1));
            float[] value = entry.getValue();
            for (int d = 0; d < value.length; d++) value[d] /= count;
            l2(value);
        }
        return sums;
    }

    private void updateCentroid(float[] centroid, int count, float[] vector) {
        int next = count + 1;
        for (int i = 0; i < centroid.length; i++) {
            centroid[i] = (centroid[i] * count + vector[i]) / next;
        }
        l2(centroid);
    }

    private float[] l2(float[] vector) {
        double sum = 0.0;
        for (float v : vector) sum += v * v;
        double norm = Math.sqrt(Math.max(sum, 1e-12));
        for (int i = 0; i < vector.length; i++) vector[i] = (float) (vector[i] / norm);
        return vector;
    }

    private double cosine(float[] a, float[] b) {
        double sum = 0.0;
        int n = Math.min(a.length, b.length);
        for (int i = 0; i < n; i++) sum += a[i] * b[i];
        return sum;
    }

    private double[] firstFiniteEmbedding(double[][] embeddings) {
        for (double[] embedding : embeddings) if (isFiniteEmbedding(embedding)) return embedding;
        return new double[0];
    }

    private String intArrayJson(int[] values) {
        StringBuilder json = new StringBuilder("[");
        for (int i = 0; i < values.length; i++) {
            if (i > 0) json.append(',');
            json.append(values[i]);
        }
        json.append(']');
        return json.toString();
    }

    private String doubleArrayJson(double[] values, int limit) {
        StringBuilder json = new StringBuilder("[");
        int n = Math.min(values.length, limit);
        for (int i = 0; i < n; i++) {
            if (i > 0) json.append(',');
            json.append(String.format(java.util.Locale.US, "%.9g", values[i]));
        }
        json.append(']');
        return json.toString();
    }

    private String longArrayJson(long[] values) {
        StringBuilder json = new StringBuilder("[");
        for (int i = 0; i < values.length; i++) {
            if (i > 0) json.append(',');
            json.append(values[i]);
        }
        json.append(']');
        return json.toString();
    }

    private String floatSummaryJson(float[] values, int frames, int dims) {
        double sum = 0.0;
        double abs = 0.0;
        double min = Double.POSITIVE_INFINITY;
        double max = Double.NEGATIVE_INFINITY;
        for (float value : values) {
            sum += value;
            abs += Math.abs(value);
            min = Math.min(min, value);
            max = Math.max(max, value);
        }
        if (values.length == 0) {
            min = 0.0;
            max = 0.0;
        }
        StringBuilder first = new StringBuilder("[");
        int n = Math.min(values.length, 64);
        for (int i = 0; i < n; i++) {
            if (i > 0) first.append(',');
            first.append(String.format(java.util.Locale.US, "%.9g", values[i]));
        }
        first.append(']');
        return "{"
                + "\"frames\":" + frames + ","
                + "\"dims\":" + dims + ","
                + "\"count\":" + values.length + ","
                + "\"sum\":" + String.format(java.util.Locale.US, "%.12g", sum) + ","
                + "\"abs\":" + String.format(java.util.Locale.US, "%.12g", abs) + ","
                + "\"min\":" + String.format(java.util.Locale.US, "%.12g", min) + ","
                + "\"max\":" + String.format(java.util.Locale.US, "%.12g", max) + ","
                + "\"first64\":" + first
                + "}";
    }

    private String turnsJson(List<DiarizationResult.Turn> turns) {
        StringBuilder json = new StringBuilder("[");
        for (int i = 0; i < turns.size(); i++) {
            DiarizationResult.Turn turn = turns.get(i);
            if (i > 0) json.append(',');
            json.append('{')
                    .append("\"start\":").append(String.format(java.util.Locale.US, "%.4f", turn.start)).append(',')
                    .append("\"end\":").append(String.format(java.util.Locale.US, "%.4f", turn.end)).append(',')
                    .append("\"speaker\":").append(turn.speaker)
                    .append('}');
        }
        json.append(']');
        return json.toString();
    }

    private void writeDebugDump(String name, String json) {
        try {
            java.io.File dir = new java.io.File("/sdcard/Android/data/com.asrvn.offline/files/debug");
            if (!dir.exists() && !dir.mkdirs()) return;
            java.nio.file.Files.write(new java.io.File(dir, name).toPath(), json.getBytes(java.nio.charset.StandardCharsets.UTF_8));
        } catch (Throwable ignored) {
        }
    }

    private void writeDebugFloatFile(String name, float[] values, int count) {
        try {
            java.io.File dir = new java.io.File("/sdcard/Android/data/com.asrvn.offline/files/debug");
            if (!dir.exists() && !dir.mkdirs()) return;
            int n = Math.max(0, Math.min(count, values == null ? 0 : values.length));
            java.nio.ByteBuffer buffer = java.nio.ByteBuffer.allocate(n * 4).order(java.nio.ByteOrder.LITTLE_ENDIAN);
            for (int i = 0; i < n; i++) buffer.putFloat(values[i]);
            java.nio.file.Files.write(new java.io.File(dir, name).toPath(), buffer.array());
        } catch (Throwable ignored) {
        }
    }

    private void ensurePyannote(boolean accelerator, int threads) throws Exception {
        if (pyannoteSegSession == null) {
            String[] selectedProvider = new String[]{"cpu"};
            pyannoteSegSession = env.createSession(models.pyannoteSegmentation(), options(false, threads, selectedProvider));
            pyannoteSegProvider = selectedProvider[0];
        }
    }

    private void ensurePyannoteCommunity(boolean accelerator, int threads) throws Exception {
        if (pyannoteAssets == null) {
            NpyNpzReader.ArrayData weight = NpyNpzReader.npy(models.pyannoteResnetWeight());
            NpyNpzReader.ArrayData bias = NpyNpzReader.npy(models.pyannoteResnetBias());
            if (weight.data.length != PYANNOTE_EMB_DIM * 5120 || bias.data.length != PYANNOTE_EMB_DIM) {
                throw new IllegalStateException("Invalid Pyannote Community-1 projection asset shapes.");
            }
            pyannoteAssets = new PyannoteCommunityAssets(weight.data, bias.data, NpyNpzReader.npz(models.pyannotePldaPrepared()));
        }
        if (pyannoteEmbeddingSession == null) {
            String[] selectedProvider = new String[]{"cpu"};
            pyannoteEmbeddingSession = env.createSession(models.pyannoteEmbeddingEncoder(), options(accelerator, threads, selectedProvider));
            pyannoteEmbeddingProvider = selectedProvider[0];
        }
    }

    private static double[] requireArray(Map<String, NpyNpzReader.ArrayData> arrays, String key, int length) {
        NpyNpzReader.ArrayData data = arrays.get(key);
        if (data == null || data.data.length != length) throw new IllegalStateException("Pyannote PLDA archive missing or invalid: " + key);
        return data.data;
    }

    private OrtSession.SessionOptions options(boolean accelerator, int threads) throws Exception {
        return options(accelerator, threads, null);
    }

    private OrtSession.SessionOptions options(boolean accelerator, int threads, String[] providerOut) throws Exception {
        OrtSession.SessionOptions options = new OrtSession.SessionOptions();
        options.setOptimizationLevel(OrtSession.SessionOptions.OptLevel.ALL_OPT);
        options.setIntraOpNumThreads(Math.max(1, Math.min(8, threads)));
        String selectedProvider = "cpu";
        if (accelerator) {
            try {
                options.addNnapi();
                selectedProvider = "nnapi";
            } catch (Throwable ignored) {
                // Keep XNNPACK/CPU fallback below.
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
        provider = selectedProvider;
        if (providerOut != null && providerOut.length > 0) providerOut[0] = selectedProvider;
        return options;
    }

    private String camppProviderSummary() {
        return "pyannote_vad=" + pyannoteSegProvider + ",campp_embedding=" + camppProvider + ",senko_spectral=cpu";
    }

    private String pyannoteProviderSummary() {
        return "pyannote_segmentation=" + pyannoteSegProvider + ",pyannote_embedding=" + pyannoteEmbeddingProvider + ",vbx=cpu";
    }

    @Override
    public void close() throws Exception {
        if (camppSession != null) camppSession.close();
        if (pyannoteSegSession != null) pyannoteSegSession.close();
        if (pyannoteEmbeddingSession != null) pyannoteEmbeddingSession.close();
    }
}
