package com.asrvn.offline.punctuation;

import ai.onnxruntime.OnnxTensor;
import ai.onnxruntime.OnnxValue;
import ai.onnxruntime.OrtEnvironment;
import ai.onnxruntime.OrtSession;

import com.asrvn.offline.asr.PureOrtRecognizer;
import com.asrvn.offline.models.ModelFileRegistry;

import java.nio.FloatBuffer;
import java.nio.LongBuffer;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashMap;
import java.util.HashSet;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Optional;
import java.util.Set;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

public final class PunctuationRestorer implements AutoCloseable {
    private static final String START_TOKEN = "$START";
    private static final String UNK_TOKEN = "@@UNKNOWN@@";
    private static final String PAD_TOKEN = "@@PADDING@@";
    private static final int MAX_LEN = 80;
    private static final int MIN_LEN = 3;
    private static final int CHUNK_SIZE = 56;
    private static final int OVERLAP_SIZE = 16;
    private static final int MIN_WORDS_CUT = 6;
    private static final int STRIDE = CHUNK_SIZE - OVERLAP_SIZE;
    private static final int ITERATIONS = 3;
    private static final int MINI_BATCH_SIZE = 8;
    private static final double DEFAULT_CONFIDENCE = 0.5 - (6 - 1) * (1.3 / 9.0);
    private static final double DEFAULT_CASE_CONFIDENCE = -1.5 + (6 - 1) * (2.0 / 9.0);
    private static final Locale VI = Locale.forLanguageTag("vi-VN");
    private static final Set<String> ALLOWED_PUNCT = new HashSet<>(Arrays.asList(":", ".", ",", "?"));

    public interface ProgressListener {
        void onProgress(int iteration, int done, int total);
    }

    public static final class Result {
        public final String text;
        public final PureOrtRecognizer.DecodeResult asr;
        public final double elapsedSeconds;
        public final int chunks;
        public final String executionProvider;

        Result(String text, PureOrtRecognizer.DecodeResult asr, double elapsedSeconds, int chunks, String executionProvider) {
            this.text = text;
            this.asr = asr;
            this.elapsedSeconds = elapsedSeconds;
            this.chunks = chunks;
            this.executionProvider = executionProvider;
        }
    }

    private static final class Vocab {
        final Map<String, Integer> tokenToId = new HashMap<>();
        int padId = 0;
        int unkId = 1;
        int startId = 0;
    }

    private static final class EncodedRow {
        final long[] inputIds;
        final long[] offsets;

        EncodedRow(long[] inputIds, long[] offsets) {
            this.inputIds = inputIds;
            this.offsets = offsets;
        }
    }

    private static final class EncodedBatch implements AutoCloseable {
        final Map<String, OnnxTensor> inputs = new LinkedHashMap<>();

        EncodedBatch(OrtEnvironment env, List<List<String>> tokenBatch, Vocab vocab) throws Exception {
            List<EncodedRow> rows = new ArrayList<>();
            int maxSeqLen = 0;
            int maxOffsetsLen = 0;
            for (List<String> sequence : tokenBatch) {
                List<String> prefixed = new ArrayList<>();
                prefixed.add(START_TOKEN);
                for (int i = 0; i < Math.min(sequence.size(), MAX_LEN); i++) prefixed.add(sequence.get(i));

                List<Long> inputIds = new ArrayList<>();
                List<Long> offsets = new ArrayList<>();
                for (String word : prefixed) {
                    offsets.add((long) inputIds.size());
                    for (String piece : wordPieceTokenize(word, vocab)) {
                        if (START_TOKEN.equals(piece)) {
                            inputIds.add((long) vocab.startId);
                        } else {
                            inputIds.add((long) vocab.tokenToId.getOrDefault(piece, vocab.unkId));
                        }
                    }
                }

                long[] ids = new long[inputIds.size()];
                for (int i = 0; i < inputIds.size(); i++) ids[i] = inputIds.get(i);
                long[] offs = new long[offsets.size()];
                for (int i = 0; i < offsets.size(); i++) offs[i] = offsets.get(i);
                rows.add(new EncodedRow(ids, offs));
                maxSeqLen = Math.max(maxSeqLen, ids.length);
                maxOffsetsLen = Math.max(maxOffsetsLen, offs.length);
            }

            int batchSize = rows.size();
            long[] inputIds = new long[Math.max(1, batchSize * maxSeqLen)];
            long[] attentionMask = new long[Math.max(1, batchSize * maxSeqLen)];
            long[] tokenTypeIds = new long[Math.max(1, batchSize * maxSeqLen)];
            long[] inputOffsets = new long[Math.max(1, batchSize * maxOffsetsLen)];

            Arrays.fill(inputIds, vocab.padId);
            for (int b = 0; b < batchSize; b++) {
                EncodedRow row = rows.get(b);
                for (int i = 0; i < maxSeqLen; i++) {
                    int pos = b * maxSeqLen + i;
                    if (i < row.inputIds.length) {
                        inputIds[pos] = row.inputIds[i];
                        attentionMask[pos] = 1;
                    }
                    tokenTypeIds[pos] = 0;
                }
                for (int i = 0; i < maxOffsetsLen; i++) {
                    inputOffsets[b * maxOffsetsLen + i] = i < row.offsets.length ? row.offsets[i] : 0;
                }
            }

            inputs.put("input_ids", OnnxTensor.createTensor(env, LongBuffer.wrap(inputIds), new long[]{batchSize, maxSeqLen}));
            inputs.put("attention_mask", OnnxTensor.createTensor(env, LongBuffer.wrap(attentionMask), new long[]{batchSize, maxSeqLen}));
            inputs.put("token_type_ids", OnnxTensor.createTensor(env, LongBuffer.wrap(tokenTypeIds), new long[]{batchSize, maxSeqLen}));
            inputs.put("input_offsets", OnnxTensor.createTensor(env, LongBuffer.wrap(inputOffsets), new long[]{batchSize, maxOffsetsLen}));
        }

        @Override
        public void close() {
            for (OnnxTensor tensor : inputs.values()) {
                try {
                    tensor.close();
                } catch (Exception ignored) {
                }
            }
        }
    }

    private static final class Prediction {
        final double[][] probabilities;
        final int[][] indexes;
        final double[] errorProbabilities;

        Prediction(double[][] probabilities, int[][] indexes, double[] errorProbabilities) {
            this.probabilities = probabilities;
            this.indexes = indexes;
            this.errorProbabilities = errorProbabilities;
        }
    }

    private static final class PredictResult {
        final List<double[]> probabilities = new ArrayList<>();
        final List<int[]> indexes = new ArrayList<>();
        final List<Double> errorProbabilities = new ArrayList<>();
    }

    private static final class SplitChunks {
        final List<List<String>> chunks;
        final List<int[]> ranges;
        final List<double[]> hintChunks;

        SplitChunks(List<List<String>> chunks, List<int[]> ranges, List<double[]> hintChunks) {
            this.chunks = chunks;
            this.ranges = ranges;
            this.hintChunks = hintChunks;
        }
    }

    private static final class Action {
        final int start;
        final int end;
        final String label;

        Action(int start, int end, String label) {
            this.start = start;
            this.end = end;
            this.label = label;
        }
    }

    private final OrtEnvironment env;
    private final OrtSession session;
    private final String provider;
    private final Vocab vocab;
    private final List<String> labels;
    private final int noopIndex;
    private final int incorrIndex;
    private final int appendPeriodIndex;
    private final int appendCommaIndex;
    private final List<Integer> caseIndexes = new ArrayList<>();

    public PunctuationRestorer(OrtEnvironment env, ModelFileRegistry models, boolean accelerator, int threads) throws Exception {
        this.env = env;
        this.vocab = parseVocab(new String(models.punctuationVocab(), StandardCharsets.UTF_8));
        this.labels = parseLines(new String(models.punctuationLabels(), StandardCharsets.UTF_8));
        List<String> tags = parseLines(new String(models.punctuationDetectionTags(), StandardCharsets.UTF_8));
        this.noopIndex = labels.indexOf("$KEEP");
        this.incorrIndex = Math.max(0, tags.indexOf("INCORRECT"));
        this.appendPeriodIndex = labels.indexOf("$APPEND_.");
        this.appendCommaIndex = labels.indexOf("$APPEND_,");
        for (int i = 0; i < labels.size(); i++) {
            if (labels.get(i).startsWith("$TRANSFORM_CASE_")) caseIndexes.add(i);
        }

        OrtSession created = null;
        String selectedProvider = "cpu";
        Throwable acceleratorError = null;
        if (accelerator) {
            try {
                OrtSession.SessionOptions options = options(true, threads);
                created = env.createSession(models.punctuationModelPath(), options);
                selectedProvider = "nnapi";
            } catch (Throwable error) {
                acceleratorError = error;
            }
        }
        if (created == null) {
            String[] fallbackProvider = new String[]{"cpu"};
            OrtSession.SessionOptions options = options(false, threads, fallbackProvider);
            created = env.createSession(models.punctuationModelPath(), options);
            selectedProvider = fallbackProvider[0];
        }
        this.session = created;
        this.provider = selectedProvider;
    }

    public String provider() {
        return provider;
    }

    public Result restore(PureOrtRecognizer.DecodeResult asr, double[] pauseHints, ProgressListener progress) throws Exception {
        return restore(asr, pauseHints, DEFAULT_CONFIDENCE, DEFAULT_CASE_CONFIDENCE, progress);
    }

    public Result restore(
            PureOrtRecognizer.DecodeResult asr,
            double[] pauseHints,
            double punctuationConfidence,
            double caseConfidence,
            ProgressListener progress
    ) throws Exception {
        String raw = asr == null || asr.text == null ? "" : asr.text.trim();
        if (raw.isEmpty()) return new Result("", asr, 0, 0, provider);
        long started = System.nanoTime();
        List<String> words = splitWords(raw);
        double[] hints = pauseHints != null && pauseHints.length == words.size() ? pauseHints : null;
        SplitChunks split = splitChunks(words, hints);
        List<List<String>> finalBatch = new ArrayList<>();
        Map<Integer, Set<String>> previousPredictions = new HashMap<>();
        List<Integer> predIds = new ArrayList<>();
        for (int i = 0; i < split.chunks.size(); i++) {
            List<String> copy = new ArrayList<>(split.chunks.get(i));
            finalBatch.add(copy);
            previousPredictions.put(i, new HashSet<>(Collections.singleton(joinKey(copy))));
            if (copy.size() >= MIN_LEN) predIds.add(i);
        }

        for (int iter = 0; iter < ITERATIONS && !predIds.isEmpty(); iter++) {
            final int iteration = iter;
            List<List<String>> origBatch = new ArrayList<>();
            for (int id : predIds) origBatch.add(finalBatch.get(id));
            List<double[]> currentHints = null;
            if (iter == 0 && split.hintChunks != null) {
                currentHints = new ArrayList<>();
                for (int id : predIds) currentHints.add(split.hintChunks.get(id));
            }
            PredictResult predicted = predict(origBatch, currentHints, punctuationConfidence, caseConfidence, (done, total) -> {
                if (progress != null) progress.onProgress(iteration, done, total);
            });
            List<List<String>> predBatch = postprocessBatch(origBatch, predicted);
            List<Integer> nextPredIds = new ArrayList<>();
            for (int i = 0; i < predIds.size(); i++) {
                int originalId = predIds.get(i);
                String oldKey = joinKey(finalBatch.get(originalId));
                String newKey = joinKey(predBatch.get(i));
                if (oldKey.equals(newKey)) continue;
                finalBatch.set(originalId, predBatch.get(i));
                Set<String> seen = previousPredictions.get(originalId);
                if (!seen.contains(newKey)) {
                    seen.add(newKey);
                    nextPredIds.add(originalId);
                }
            }
            predIds = nextPredIds;
        }

        String restored = mergePunctuationChunks(finalBatch.subList(split.ranges.get(0)[0], split.ranges.get(0)[1]));
        restored = restored.replaceAll("\\s+([:.,?])", "$1");
        restored = postProcessText(restored);
        PureOrtRecognizer.DecodeResult updated = applyToDecodeResult(asr, restored);
        double elapsed = (System.nanoTime() - started) / 1_000_000_000.0;
        return new Result(restored, updated, elapsed, split.chunks.size(), provider);
    }

    private OrtSession.SessionOptions options(boolean accelerator, int threads) throws Exception {
        return options(accelerator, threads, null);
    }

    private OrtSession.SessionOptions options(boolean accelerator, int threads, String[] providerOut) throws Exception {
        OrtSession.SessionOptions options = new OrtSession.SessionOptions();
        options.setOptimizationLevel(OrtSession.SessionOptions.OptLevel.ALL_OPT);
        options.setIntraOpNumThreads(Math.max(1, Math.min(8, threads)));
        options.setInterOpNumThreads(1);
        if (accelerator) {
            options.addNnapi();
        } else {
            try {
                options.addXnnpack(Collections.singletonMap("intra_op_num_threads", String.valueOf(Math.max(1, Math.min(8, threads)))));
                if (providerOut != null) providerOut[0] = "xnnpack";
            } catch (Throwable ignored) {
                options.addCPU(true);
                if (providerOut != null) providerOut[0] = "cpu";
            }
        }
        return options;
    }

    private PredictResult predict(
            List<List<String>> tokenBatch,
            List<double[]> pauseHintsBatch,
            double punctuationConfidence,
            double caseConfidence,
            MiniProgress progress
    ) throws Exception {
        PredictResult all = new PredictResult();
        int total = tokenBatch.size();
        for (int start = 0; start < total; start += MINI_BATCH_SIZE) {
            int end = Math.min(total, start + MINI_BATCH_SIZE);
            List<List<String>> slice = tokenBatch.subList(start, end);
            try (EncodedBatch encoded = new EncodedBatch(env, slice, vocab);
                 OrtSession.Result outputs = session.run(encoded.inputs)) {
                OnnxTensor logitsTensor = outputTensor(outputs, "logits", 0);
                OnnxTensor detectTensor = outputTensor(outputs, "detect_logits", 1);
                List<double[]> hintSlice = pauseHintsBatch == null ? null : pauseHintsBatch.subList(start, end);
                Prediction converted = convertOutputs(logitsTensor, detectTensor, hintSlice, punctuationConfidence, caseConfidence);
                Collections.addAll(all.probabilities, converted.probabilities);
                Collections.addAll(all.indexes, converted.indexes);
                for (double value : converted.errorProbabilities) all.errorProbabilities.add(value);
            }
            if (progress != null) progress.onProgress(end, total);
        }
        return all;
    }

    private OnnxTensor outputTensor(OrtSession.Result outputs, String name, int index) throws Exception {
        Optional<OnnxValue> byName = outputs.get(name);
        if (byName.isPresent()) return (OnnxTensor) byName.get();
        return (OnnxTensor) outputs.get(index);
    }

    private Prediction convertOutputs(
            OnnxTensor logitsTensor,
            OnnxTensor detectTensor,
            List<double[]> pauseHintsBatch,
            double punctuationConfidence,
            double caseConfidence
    ) throws Exception {
        long[] logitsShape = logitsTensor.getInfo().getShape();
        long[] detectShape = detectTensor.getInfo().getShape();
        int batchSize = (int) logitsShape[0];
        int numWords = (int) logitsShape[1];
        int labelCount = (int) logitsShape[2];
        int detectCount = (int) detectShape[2];
        float[] logits = floatArray(logitsTensor);
        float[] detect = floatArray(detectTensor);
        double[][] probabilities = new double[batchSize][numWords];
        int[][] indexes = new int[batchSize][numWords];
        double[] errorProbs = new double[batchSize];

        for (int b = 0; b < batchSize; b++) {
            double maxError = 0;
            for (int w = 0; w < numWords; w++) {
                int labelBase = (b * numWords + w) * labelCount;
                Double pauseGap = null;
                if (pauseHintsBatch != null && b < pauseHintsBatch.size() && pauseHintsBatch.get(b) != null && w > 0) {
                    double[] hints = pauseHintsBatch.get(b);
                    if (w - 1 < hints.length) pauseGap = hints[w - 1];
                }
                Argmax label = softmaxArgmax(logits, labelBase, labelCount, pauseGap, punctuationConfidence, caseConfidence, true);
                probabilities[b][w] = label.probability;
                indexes[b][w] = label.index;

                int detectBase = (b * numWords + w) * detectCount;
                Argmax detectProb = softmaxArgmax(detect, detectBase, detectCount, null, false);
                if (detectProb.index == incorrIndex && detectProb.probability > maxError) maxError = detectProb.probability;
            }
            errorProbs[b] = maxError;
        }
        return new Prediction(probabilities, indexes, errorProbs);
    }

    private Argmax softmaxArgmax(float[] data, int base, int count, Double pauseGap) {
        return softmaxArgmax(data, base, count, pauseGap, DEFAULT_CONFIDENCE, DEFAULT_CASE_CONFIDENCE, true);
    }

    private Argmax softmaxArgmax(float[] data, int base, int count, Double pauseGap, boolean adjustPunctuation) {
        return softmaxArgmax(data, base, count, pauseGap, DEFAULT_CONFIDENCE, DEFAULT_CASE_CONFIDENCE, adjustPunctuation);
    }

    private Argmax softmaxArgmax(
            float[] data,
            int base,
            int count,
            Double pauseGap,
            double punctuationConfidence,
            double caseConfidence,
            boolean adjustPunctuation
    ) {
        double maxLogit = Double.NEGATIVE_INFINITY;
        for (int i = 0; i < count; i++) maxLogit = Math.max(maxLogit, data[base + i]);
        double[] adjusted = new double[count];
        double sum = 0;
        for (int i = 0; i < count; i++) {
            double value = Math.exp(data[base + i] - maxLogit);
            adjusted[i] = value;
            sum += value;
        }
        for (int i = 0; i < count; i++) adjusted[i] /= Math.max(sum, 1e-12);
        if (adjustPunctuation) {
            if (noopIndex >= 0 && noopIndex < count) adjusted[noopIndex] += punctuationConfidence;
            for (int index : caseIndexes) if (index >= 0 && index < count) adjusted[index] += caseConfidence;
            if (pauseGap != null && Double.isFinite(pauseGap)) {
                int current = argmax(adjusted);
                if (pauseGap >= 1.0 && current == noopIndex && appendPeriodIndex >= 0 && appendPeriodIndex < count) {
                    adjusted[noopIndex] -= 0.2;
                    adjusted[appendPeriodIndex] += 0.2;
                } else if (pauseGap >= 0.2 && current == noopIndex && appendCommaIndex >= 0 && appendCommaIndex < count) {
                    adjusted[appendCommaIndex] += 0.2;
                } else if (pauseGap < 0.1 && appendCommaIndex >= 0 && appendCommaIndex < count) {
                    adjusted[appendCommaIndex] -= 0.3;
                }
            }
        }
        int best = argmax(adjusted);
        return new Argmax(best, adjusted[best]);
    }

    private int argmax(double[] values) {
        int best = 0;
        double bestValue = Double.NEGATIVE_INFINITY;
        for (int i = 0; i < values.length; i++) {
            if (values[i] > bestValue) {
                bestValue = values[i];
                best = i;
            }
        }
        return best;
    }

    private List<List<String>> postprocessBatch(List<List<String>> tokenBatch, PredictResult predicted) {
        List<List<String>> results = new ArrayList<>();
        for (int b = 0; b < tokenBatch.size(); b++) {
            List<String> tokens = tokenBatch.get(b);
            int length = Math.min(tokens.size(), MAX_LEN);
            int[] idxs = predicted.indexes.get(b);
            double[] probabilities = predicted.probabilities.get(b);
            double errorProb = predicted.errorProbabilities.get(b);
            if (max(idxs) == noopIndex || errorProb < 0) {
                results.add(tokens);
                continue;
            }
            List<Action> edits = new ArrayList<>();
            for (int i = 0; i <= length && i < idxs.length; i++) {
                if (idxs[i] == noopIndex) continue;
                String label = idxs[i] >= 0 && idxs[i] < labels.size() ? labels.get(idxs[i]) : null;
                Action action = getAction(i, probabilities[i], label);
                if (action != null) edits.add(action);
            }
            results.add(applyEdits(tokens, edits));
        }
        return results;
    }

    private Action getAction(int index, double probability, String label) {
        if (label == null || label.isEmpty() || "$KEEP".equals(label) || UNK_TOKEN.equals(label) || PAD_TOKEN.equals(label)) return null;
        if ("$DELETE".equals(label) || label.startsWith("$REPLACE_")) return null;
        if (label.startsWith("$APPEND_")) {
            String added = label.replace("$APPEND_", "");
            if (!ALLOWED_PUNCT.contains(added)) return null;
            return new Action(index, index, label);
        }
        if (label.startsWith("$TRANSFORM_CASE_")) {
            return new Action(index - 1, index, label);
        }
        return null;
    }

    private List<String> applyEdits(List<String> tokens, List<Action> edits) {
        List<String> target = new ArrayList<>(tokens);
        int shift = 0;
        for (Action edit : edits) {
            int targetPos = edit.start + shift;
            if (edit.start < 0) continue;
            String sourceToken = targetPos < target.size() ? target.get(targetPos) : "";
            if (edit.label.isEmpty()) {
                if (targetPos >= 0 && targetPos < target.size()) {
                    target.remove(targetPos);
                    shift -= 1;
                }
            } else if (edit.start == edit.end) {
                String word = edit.label.replace("$APPEND_", "");
                if ((targetPos < target.size() && target.get(targetPos).equals(word))
                        || (targetPos > 0 && target.get(targetPos - 1).equals(word))) {
                    continue;
                }
                int insertAt = Math.max(0, Math.min(targetPos, target.size()));
                target.add(insertAt, word);
                shift += 1;
            } else if (edit.label.startsWith("$TRANSFORM_") && targetPos >= 0 && targetPos < target.size()) {
                target.set(targetPos, applyCaseTransform(sourceToken, edit.label));
            }
        }
        return target;
    }

    private SplitChunks splitChunks(List<String> tokens, double[] pauseHints) {
        List<List<String>> chunks = new ArrayList<>();
        List<double[]> hintChunks = pauseHints == null ? null : new ArrayList<>();
        List<int[]> indices = new ArrayList<>();
        int start = chunks.size();
        int count = tokens.size();
        if (count <= CHUNK_SIZE) {
            addChunk(chunks, hintChunks, tokens, pauseHints, 0, count);
        } else if (count < CHUNK_SIZE * 2 - OVERLAP_SIZE) {
            int splitIdx = (count + OVERLAP_SIZE + 1) / 2;
            addChunk(chunks, hintChunks, tokens, pauseHints, 0, splitIdx);
            addChunk(chunks, hintChunks, tokens, pauseHints, splitIdx - OVERLAP_SIZE, count);
        } else {
            for (int i = 0; i < count - OVERLAP_SIZE; i += STRIDE) {
                addChunk(chunks, hintChunks, tokens, pauseHints, i, Math.min(count, i + CHUNK_SIZE));
            }
        }
        indices.add(new int[]{start, chunks.size()});
        return new SplitChunks(chunks, indices, hintChunks);
    }

    private void addChunk(List<List<String>> chunks, List<double[]> hintChunks, List<String> tokens, double[] hints, int start, int end) {
        chunks.add(new ArrayList<>(tokens.subList(start, end)));
        if (hintChunks != null && hints != null) {
            hintChunks.add(Arrays.copyOfRange(hints, start, end));
        }
    }

    private String mergePunctuationChunks(List<List<String>> chunks) {
        List<String> result = new ArrayList<>();
        for (List<String> chunk : chunks) {
            result = applyChunkMerging(result, chunk);
        }
        return String.join(" ", result);
    }

    private List<String> applyChunkMerging(List<String> tokens, List<String> nextTokens) {
        if (tokens.isEmpty()) return new ArrayList<>(nextTokens);
        List<Integer> sourceIdx = new ArrayList<>();
        List<String> sourceTokens = new ArrayList<>();
        for (int i = tokens.size() - 1; i >= 0 && sourceIdx.size() < OVERLAP_SIZE; i--) {
            if (!ALLOWED_PUNCT.contains(tokens.get(i))) {
                sourceIdx.add(0, i);
                sourceTokens.add(0, tokens.get(i).toLowerCase(VI));
            }
        }
        List<Integer> targetIdx = new ArrayList<>();
        List<String> targetTokens = new ArrayList<>();
        for (int i = 0; i < nextTokens.size() && targetIdx.size() < OVERLAP_SIZE; i++) {
            if (!ALLOWED_PUNCT.contains(nextTokens.get(i))) {
                targetIdx.add(i);
                targetTokens.add(nextTokens.get(i).toLowerCase(VI));
            }
        }
        int maxOverlap = Math.min(sourceTokens.size(), targetTokens.size());
        for (int size = maxOverlap; size >= MIN_WORDS_CUT; size--) {
            int sourceStart = sourceTokens.size() - size;
            String left = String.join("\u0001", sourceTokens.subList(sourceStart, sourceTokens.size()));
            String right = String.join("\u0001", targetTokens.subList(0, size));
            if (!left.equals(right)) continue;
            int keep = Math.min(OVERLAP_SIZE - MIN_WORDS_CUT, size - 1);
            int tailIdx = sourceIdx.get(sourceStart + keep);
            int headIdx = targetIdx.get(keep);
            List<String> merged = new ArrayList<>(tokens.subList(0, tailIdx));
            merged.addAll(nextTokens.subList(headIdx, nextTokens.size()));
            return merged;
        }
        int mergeHeadPos = Math.min(targetIdx.size() - 1, OVERLAP_SIZE - MIN_WORDS_CUT);
        int mergeHead = mergeHeadPos >= 0 ? targetIdx.get(mergeHeadPos) : 0;
        List<String> merged = new ArrayList<>(tokens);
        merged.addAll(nextTokens.subList(mergeHead, nextTokens.size()));
        return merged;
    }

    private PureOrtRecognizer.DecodeResult applyToDecodeResult(PureOrtRecognizer.DecodeResult asr, String restored) {
        List<PureOrtRecognizer.Word> source = asr == null ? Collections.emptyList() : asr.words;
        if (source == null || source.isEmpty()) return new PureOrtRecognizer.DecodeResult(restored, Collections.emptyList(), 0, 0);
        List<String> tokens = splitWords(restored);
        List<PureOrtRecognizer.Word> words = new ArrayList<>();
        int out = 0;
        for (PureOrtRecognizer.Word word : source) {
            String text = word.text;
            while (out < tokens.size()) {
                String candidate = tokens.get(out++);
                if (isPurePunctuation(candidate)) {
                    if (!words.isEmpty()) {
                        PureOrtRecognizer.Word prev = words.remove(words.size() - 1);
                        words.add(new PureOrtRecognizer.Word(prev.text + candidate, prev.start, prev.end, prev.probability));
                    }
                    continue;
                }
                text = candidate;
                break;
            }
            while (out < tokens.size() && isPurePunctuation(tokens.get(out))) {
                text += tokens.get(out++);
            }
            words.add(new PureOrtRecognizer.Word(text, word.start, word.end, word.probability));
        }
        if (out < tokens.size() && !words.isEmpty()) {
            StringBuilder tail = new StringBuilder(words.get(words.size() - 1).text);
            while (out < tokens.size()) {
                String token = tokens.get(out++);
                if (isPurePunctuation(token)) tail.append(token);
            }
            PureOrtRecognizer.Word prev = words.remove(words.size() - 1);
            words.add(new PureOrtRecognizer.Word(tail.toString(), prev.start, prev.end, prev.probability));
        }
        return new PureOrtRecognizer.DecodeResult(restored, words, asr.frames, asr.tokens);
    }

    private boolean isPurePunctuation(String value) {
        return value != null && ALLOWED_PUNCT.contains(value);
    }

    private String postProcessText(String text) {
        String output = text.replace(':', ' ');
        output = output.replaceAll(",+", ",");
        output = output.replaceAll("\\.{4,}", "...");
        output = output.replaceAll(",\\s*\\.", ".");

        String[] sentences = output.split("(?<=[.!?])\\s+");
        List<String> cleaned = new ArrayList<>();
        for (String sentence : sentences) {
            int commaCount = countChar(sentence, ',');
            String[] words = sentence.trim().isEmpty() ? new String[0] : sentence.trim().split("\\s+");
            if (words.length < 8 && commaCount > 1) {
                int firstComma = sentence.indexOf(',');
                if (firstComma >= 0) {
                    sentence = sentence.substring(0, firstComma + 1) + sentence.substring(firstComma + 1).replace(",", "");
                }
            }
            cleaned.add(sentence);
        }
        output = String.join(" ", cleaned);
        output = output.replaceAll("([,.!?])([^\\s])", "$1 $2");
        output = output.replaceAll("\\s+([,.!?])", "$1");
        output = output.replaceAll("^,\\s*", "");
        output = output.replaceAll("\\.\\s*,", ". ");
        output = output.replaceAll("\\s+", " ");
        output = capitalizeSentenceStarts(output);
        return output.trim();
    }

    private String capitalizeSentenceStarts(String text) {
        Pattern pattern = Pattern.compile("(^|[.!?]\\s+)(\\p{L})");
        Matcher matcher = pattern.matcher(text);
        StringBuffer buffer = new StringBuffer();
        while (matcher.find()) {
            matcher.appendReplacement(buffer, Matcher.quoteReplacement(matcher.group(1) + matcher.group(2).toUpperCase(VI)));
        }
        matcher.appendTail(buffer);
        return buffer.toString();
    }

    private String applyCaseTransform(String token, String label) {
        if (label.endsWith("LOWER")) return token.toLowerCase(VI);
        if (label.endsWith("UPPER")) return token.toUpperCase(VI);
        if (label.endsWith("CAPITAL")) {
            if (token.isEmpty()) return token;
            return token.substring(0, 1).toUpperCase(VI) + token.substring(1).toLowerCase(VI);
        }
        if (label.endsWith("CAPITAL_1")) {
            if (token.length() < 2) return token;
            String tail = token.substring(1);
            return token.substring(0, 1) + tail.substring(0, 1).toUpperCase(VI) + tail.substring(1).toLowerCase(VI);
        }
        if (label.endsWith("UPPER_-1")) {
            if (token.isEmpty()) return token;
            return token.substring(0, token.length() - 1).toUpperCase(VI) + token.substring(token.length() - 1);
        }
        return token;
    }

    private static List<String> wordPieceTokenize(String word, Vocab vocab) {
        if (START_TOKEN.equals(word)) return Collections.singletonList(START_TOKEN);
        if (word == null || word.isEmpty()) return Collections.emptyList();
        if (word.length() > 100) return Collections.singletonList("[UNK]");
        List<String> pieces = new ArrayList<>();
        int start = 0;
        while (start < word.length()) {
            int end = word.length();
            String current = null;
            while (start < end) {
                String sub = word.substring(start, end);
                if (start > 0) sub = "##" + sub;
                if (vocab.tokenToId.containsKey(sub)) {
                    current = sub;
                    break;
                }
                end -= 1;
            }
            if (current == null) return Collections.singletonList("[UNK]");
            pieces.add(current);
            start = end;
        }
        return pieces;
    }

    private static Vocab parseVocab(String text) {
        Vocab vocab = new Vocab();
        String[] lines = text.split("\\r?\\n");
        int index = 0;
        for (String raw : lines) {
            if (raw.isEmpty()) continue;
            vocab.tokenToId.put(raw, index++);
        }
        vocab.padId = vocab.tokenToId.getOrDefault("[PAD]", 0);
        vocab.unkId = vocab.tokenToId.getOrDefault("[UNK]", 1);
        vocab.startId = vocab.tokenToId.getOrDefault(START_TOKEN, vocab.tokenToId.size());
        return vocab;
    }

    private static List<String> parseLines(String text) {
        List<String> lines = new ArrayList<>();
        for (String raw : text.split("\\r?\\n")) {
            if (!raw.isEmpty()) lines.add(raw);
        }
        return lines;
    }

    private static List<String> splitWords(String text) {
        if (text == null || text.trim().isEmpty()) return Collections.emptyList();
        List<String> out = new ArrayList<>();
        for (String token : text.trim().split("\\s+")) {
            if (!token.isEmpty()) out.add(token);
        }
        return out;
    }

    private static String joinKey(List<String> values) {
        return String.join("\u0001", values);
    }

    private static int max(int[] values) {
        int max = Integer.MIN_VALUE;
        for (int value : values) max = Math.max(max, value);
        return max;
    }

    private static int countChar(String value, char target) {
        int count = 0;
        for (int i = 0; i < value.length(); i++) if (value.charAt(i) == target) count++;
        return count;
    }

    private static float[] floatArray(OnnxTensor tensor) throws Exception {
        FloatBuffer buffer = tensor.getFloatBuffer();
        float[] values = new float[buffer.remaining()];
        buffer.get(values);
        return values;
    }

    @Override
    public void close() throws Exception {
        session.close();
    }

    private interface MiniProgress {
        void onProgress(int done, int total);
    }

    private static final class Argmax {
        final int index;
        final double probability;

        Argmax(int index, double probability) {
            this.index = index;
            this.probability = probability;
        }
    }
}
