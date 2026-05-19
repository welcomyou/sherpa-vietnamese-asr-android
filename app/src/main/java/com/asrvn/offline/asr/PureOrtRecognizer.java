package com.asrvn.offline.asr;

import ai.onnxruntime.OnnxTensor;
import ai.onnxruntime.OnnxValue;
import ai.onnxruntime.OrtEnvironment;
import ai.onnxruntime.OrtSession;
import ai.onnxruntime.TensorInfo;

import java.nio.FloatBuffer;
import java.nio.LongBuffer;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;

public final class PureOrtRecognizer implements AutoCloseable {
    private static final double WORD_ASSIGN_MAX_DURATION_SECONDS = 0.40;

    public static final class Word {
        public final String text;
        public final double start;
        public final double end;
        public final double probability;

        public Word(String text, double start, double end, double probability) {
            this.text = text;
            this.start = start;
            this.end = end;
            this.probability = probability;
        }
    }

    public static final class DecodeResult {
        public final String text;
        public final List<Word> words;
        public final int frames;
        public final int tokens;

        public DecodeResult(String text, List<Word> words, int frames, int tokens) {
            this.text = text;
            this.words = words;
            this.frames = frames;
            this.tokens = tokens;
        }
    }

    public static final class ModelFiles {
        public final byte[] encoder;
        public final byte[] decoder;
        public final byte[] joiner;
        public final byte[] tokens;
        public final byte[] bpeVocab;

        public ModelFiles(byte[] encoder, byte[] decoder, byte[] joiner, byte[] tokens, byte[] bpeVocab) {
            this.encoder = encoder;
            this.decoder = decoder;
            this.joiner = joiner;
            this.tokens = tokens;
            this.bpeVocab = bpeVocab;
        }
    }

    public static final class ModelFilePaths {
        public final String encoder;
        public final String decoder;
        public final String joiner;
        public final byte[] tokens;
        public final byte[] bpeVocab;

        public ModelFilePaths(String encoder, String decoder, String joiner, byte[] tokens, byte[] bpeVocab) {
            this.encoder = encoder;
            this.decoder = decoder;
            this.joiner = joiner;
            this.tokens = tokens;
            this.bpeVocab = bpeVocab;
        }
    }

    private final OrtEnvironment env;
    private final OrtSession encoder;
    private final OrtSession decoder;
    private final OrtSession joiner;
    private final String[] idToToken;
    private final int vocabSize;
    private final Map<String, float[]> decoderCache = new HashMap<>();
    private final ContextGraph contextGraph;
    private final FbankExtractor fbank = new FbankExtractor();
    private final List<String> encoderInputs;
    private final List<String> encoderOutputs;
    private final List<String> decoderInputs;
    private final List<String> decoderOutputs;
    private final List<String> joinerInputs;
    private final List<String> joinerOutputs;
    private final int maxActivePaths;

    public PureOrtRecognizer(OrtEnvironment env, ModelFiles files, String hotwordsText, double hotwordsScore) throws Exception {
        this(env, files, hotwordsText, hotwordsScore, 8, false, 4);
    }

    public PureOrtRecognizer(
            OrtEnvironment env,
            ModelFiles files,
            String hotwordsText,
            double hotwordsScore,
            int maxActivePaths,
            boolean nnapiPreferred,
            int cpuThreads
    ) throws Exception {
        this.env = env;
        this.maxActivePaths = Math.max(1, Math.min(16, maxActivePaths));
        this.encoder = env.createSession(files.encoder, sessionOptions(nnapiPreferred, cpuThreads));
        this.decoder = env.createSession(files.decoder, sessionOptions(false, cpuThreads));
        this.joiner = env.createSession(files.joiner, sessionOptions(nnapiPreferred, cpuThreads));
        this.encoderInputs = ordered(encoder.getInputNames());
        this.encoderOutputs = ordered(encoder.getOutputNames());
        this.decoderInputs = ordered(decoder.getInputNames());
        this.decoderOutputs = ordered(decoder.getOutputNames());
        this.joinerInputs = ordered(joiner.getInputNames());
        this.joinerOutputs = ordered(joiner.getOutputNames());

        TokenParser parser = new TokenParser();
        TokenParser.ParsedTokens parsed = parser.parseTokens(files.tokens);
        this.idToToken = parsed.idToToken;
        this.vocabSize = idToToken.length;
        TokenParser.BpeVocab tokenizer = files.bpeVocab == null
                ? null
                : parser.parseBpeVocab(files.bpeVocab, parsed.pieceToId);
        this.contextGraph = tokenizer == null ? null : ContextGraph.fromHotwords(hotwordsText, tokenizer, hotwordsScore);
    }

    public PureOrtRecognizer(
            OrtEnvironment env,
            ModelFilePaths files,
            String hotwordsText,
            double hotwordsScore,
            int maxActivePaths,
            boolean nnapiPreferred,
            int cpuThreads
    ) throws Exception {
        this.env = env;
        this.maxActivePaths = Math.max(1, Math.min(16, maxActivePaths));
        this.encoder = env.createSession(files.encoder, sessionOptions(nnapiPreferred, cpuThreads));
        this.decoder = env.createSession(files.decoder, sessionOptions(false, cpuThreads));
        this.joiner = env.createSession(files.joiner, sessionOptions(nnapiPreferred, cpuThreads));
        this.encoderInputs = ordered(encoder.getInputNames());
        this.encoderOutputs = ordered(encoder.getOutputNames());
        this.decoderInputs = ordered(decoder.getInputNames());
        this.decoderOutputs = ordered(decoder.getOutputNames());
        this.joinerInputs = ordered(joiner.getInputNames());
        this.joinerOutputs = ordered(joiner.getOutputNames());

        TokenParser parser = new TokenParser();
        TokenParser.ParsedTokens parsed = parser.parseTokens(files.tokens);
        this.idToToken = parsed.idToToken;
        this.vocabSize = idToToken.length;
        TokenParser.BpeVocab tokenizer = files.bpeVocab == null
                ? null
                : parser.parseBpeVocab(files.bpeVocab, parsed.pieceToId);
        this.contextGraph = tokenizer == null ? null : ContextGraph.fromHotwords(hotwordsText, tokenizer, hotwordsScore);
    }

    private OrtSession.SessionOptions sessionOptions(boolean nnapiPreferred, int cpuThreads) throws Exception {
        OrtSession.SessionOptions options = new OrtSession.SessionOptions();
        options.setOptimizationLevel(OrtSession.SessionOptions.OptLevel.BASIC_OPT);
        options.setIntraOpNumThreads(Math.max(1, Math.min(8, cpuThreads)));
        options.setInterOpNumThreads(1);
        if (nnapiPreferred) {
            try {
                options.addNnapi();
                return options;
            } catch (Throwable ignored) {
                // NNAPI is best-effort. CPU remains the memory-conservative fallback.
            }
        }
        options.addCPU(true);
        return options;
    }

    private List<String> ordered(Set<String> names) {
        return new ArrayList<>(names);
    }

    public FbankExtractor.Features computeFeatures(float[] samples) {
        return fbank.compute(samples);
    }

    public OnnxTensor makeInt64Tensor(long[] values, long[] dims) throws Exception {
        return OnnxTensor.createTensor(env, LongBuffer.wrap(values), dims);
    }

    public int vocabSize() {
        return vocabSize;
    }

    public String token(int id) {
        return id >= 0 && id < idToToken.length && idToToken[id] != null ? idToToken[id] : "";
    }

    public ContextGraph contextGraph() {
        return contextGraph;
    }

    public DecodeResult decode(float[] samples, double timeOffsetSeconds) throws Exception {
        FbankExtractor.Features features = fbank.compute(samples);
        if (features.frames <= 0) return new DecodeResult("", Collections.emptyList(), 0, 0);
        SearchResult search = ortBeamSearch(features);
        List<Word> words = tokensToWords(search, samples.length, timeOffsetSeconds);
        StringBuilder text = new StringBuilder();
        for (Word word : words) {
            if (text.length() > 0) text.append(' ');
            text.append(word.text);
        }
        return new DecodeResult(text.toString(), words, search.t, search.tokenIds.length);
    }

    private SearchResult ortBeamSearch(FbankExtractor.Features features) throws Exception {
        float[] encData;
        int t;
        int encDim;
        try (OnnxTensor x = OnnxTensor.createTensor(env, FloatBuffer.wrap(features.data),
                new long[] {1, features.frames, AsrConstants.NUM_MEL_BINS});
             OnnxTensor xLens = makeInt64Tensor(new long[] {features.frames}, new long[] {1});
             OrtSession.Result encOutputs = encoder.run(mapOf(encoderInputs.get(0), x, encoderInputs.get(1), xLens))) {
            OnnxTensor encOut = (OnnxTensor) encOutputs.get(encoderOutputs.get(0)).orElseThrow();
            OnnxTensor encLens = (OnnxTensor) encOutputs.get(encoderOutputs.get(1)).orElseThrow();
            long[] dims = ((TensorInfo) encOut.getInfo()).getShape();
            t = (int) encLens.getLongBuffer().get(0);
            if (t <= 0) return new SearchResult(new int[0], new int[0], new double[0], 0);
            encDim = (int) dims[2];
            encData = copyFloat(encOut);
        }

        if (!decoderCache.containsKey("0,0")) {
            decoderCache.put("0,0", runDecoderRows(Collections.singletonList(new int[] {AsrConstants.BLANK_ID, AsrConstants.BLANK_ID}))[0]);
        }

        Map<String, Hypothesis> hyps = new HashMap<>();
        hyps.put("-1,0", new Hypothesis(
                new int[] {-1, AsrConstants.BLANK_ID},
                0.0,
                new int[0],
                new double[0],
                contextGraph == null ? null : contextGraph.root
        ));

        for (int frame = 0; frame < t; frame++) {
            List<Hypothesis> prev = new ArrayList<>(hyps.values());
            int beamCount = prev.size();
            List<int[]> missingCtxs = new ArrayList<>();
            List<Integer> missingIndexes = new ArrayList<>();
            float[][] decRows = new float[beamCount][];

            for (int i = 0; i < beamCount; i++) {
                int[] ys = prev.get(i).ys;
                int left = Math.max(0, ys[ys.length - 2]);
                int right = Math.max(0, ys[ys.length - 1]);
                String key = left + "," + right;
                float[] cached = decoderCache.get(key);
                if (cached != null) {
                    decRows[i] = cached;
                } else {
                    missingCtxs.add(new int[] {left, right});
                    missingIndexes.add(i);
                }
            }

            if (!missingCtxs.isEmpty()) {
                float[][] rows = runDecoderRows(missingCtxs);
                for (int i = 0; i < rows.length; i++) {
                    int[] ctx = missingCtxs.get(i);
                    decoderCache.put(ctx[0] + "," + ctx[1], rows[i]);
                    decRows[missingIndexes.get(i)] = rows[i];
                }
            }

            int decDim = decRows[0].length;
            float[] encBuf = new float[beamCount * encDim];
            float[] decBuf = new float[beamCount * decDim];
            int encOffset = frame * encDim;
            for (int i = 0; i < beamCount; i++) {
                System.arraycopy(encData, encOffset, encBuf, i * encDim, encDim);
                System.arraycopy(decRows[i], 0, decBuf, i * decDim, decDim);
            }

            float[] logits;
            int vocab;
            try (OnnxTensor encTensor = OnnxTensor.createTensor(env, FloatBuffer.wrap(encBuf), new long[] {beamCount, encDim});
                 OnnxTensor decTensor = OnnxTensor.createTensor(env, FloatBuffer.wrap(decBuf), new long[] {beamCount, decDim});
                 OrtSession.Result output = joiner.run(mapOf(joinerInputs.get(0), encTensor, joinerInputs.get(1), decTensor))) {
                OnnxTensor logitsTensor = (OnnxTensor) output.get(joinerOutputs.get(0)).orElseThrow();
                long[] dims = ((TensorInfo) logitsTensor.getInfo()).getShape();
                vocab = (int) dims[dims.length - 1];
                logits = copyFloat(logitsTensor);
            }

            int topK = Math.min(maxActivePaths, beamCount * vocab);
            double[] topScores = new double[topK];
            int[] topIndexes = new int[topK];
            Arrays.fill(topScores, Double.NEGATIVE_INFINITY);
            Arrays.fill(topIndexes, -1);
            double[] rowLogDen = new double[beamCount];

            for (int i = 0; i < beamCount; i++) {
                int rowOffset = i * vocab;
                double maxLogit = Double.NEGATIVE_INFINITY;
                for (int v = 0; v < vocab; v++) maxLogit = Math.max(maxLogit, logits[rowOffset + v]);
                double sumExp = 0.0;
                for (int v = 0; v < vocab; v++) sumExp += Math.exp(logits[rowOffset + v] - maxLogit);
                rowLogDen[i] = maxLogit + Math.log(sumExp);
                double prevLp = prev.get(i).lp;
                for (int v = 0; v < vocab; v++) {
                    insertTopK(topScores, topIndexes, logits[rowOffset + v] - rowLogDen[i] + prevLp, i * vocab + v);
                }
            }

            Map<String, Hypothesis> nextHyps = new HashMap<>();
            for (int k = 0; k < topK; k++) {
                int flat = topIndexes[k];
                if (flat < 0) continue;
                int hi = flat / vocab;
                int token = flat % vocab;
                double score = topScores[k];
                Hypothesis source = prev.get(hi);
                int[] ys = source.ys;
                int[] frames = source.frames;
                double[] probs = source.tokenLogProbs;
                ContextGraph.State state = source.contextState;

                if (token != AsrConstants.BLANK_ID) {
                    ys = append(ys, token);
                    frames = append(frames, frame);
                    probs = append(probs, logits[hi * vocab + token] - rowLogDen[hi]);
                    if (contextGraph != null && state != null && token != AsrConstants.UNK_ID) {
                        ContextGraph.Advance advanced = contextGraph.forwardOneStep(state, token);
                        score += advanced.score;
                        state = advanced.state;
                    }
                }

                String key = join(ys);
                Hypothesis old = nextHyps.get(key);
                if (old != null) {
                    old.lp = logAdd(old.lp, score);
                } else {
                    nextHyps.put(key, new Hypothesis(ys, score, frames, probs, state));
                }
            }
            hyps = nextHyps;
        }

        if (contextGraph != null) {
            for (Hypothesis hyp : hyps.values()) {
                if (hyp.contextState != null) hyp.lp += contextGraph.finalizeScore(hyp.contextState);
            }
        }

        Hypothesis best = null;
        double bestScore = Double.NEGATIVE_INFINITY;
        for (Hypothesis hyp : hyps.values()) {
            double score = hyp.lp / Math.max(1, hyp.ys.length);
            if (score > bestScore) {
                bestScore = score;
                best = hyp;
            }
        }
        if (best == null) return new SearchResult(new int[0], new int[0], new double[0], t);
        int[] tokenIds = Arrays.stream(Arrays.copyOfRange(best.ys, AsrConstants.CONTEXT_SIZE, best.ys.length))
                .filter(id -> id > 0)
                .toArray();
        return new SearchResult(tokenIds, best.frames, best.tokenLogProbs, t);
    }

    private float[][] runDecoderRows(List<int[]> ctxs) throws Exception {
        long[] input = new long[ctxs.size() * AsrConstants.CONTEXT_SIZE];
        for (int i = 0; i < ctxs.size(); i++) {
            input[i * 2] = ctxs.get(i)[0];
            input[i * 2 + 1] = ctxs.get(i)[1];
        }
        try (OnnxTensor tensor = makeInt64Tensor(input, new long[] {ctxs.size(), AsrConstants.CONTEXT_SIZE});
             OrtSession.Result output = decoder.run(Collections.singletonMap(decoderInputs.get(0), tensor))) {
            OnnxTensor dec = (OnnxTensor) output.get(decoderOutputs.get(0)).orElseThrow();
            long[] dims = ((TensorInfo) dec.getInfo()).getShape();
            int dim = (int) dims[dims.length - 1];
            float[] values = copyFloat(dec);
            float[][] rows = new float[ctxs.size()][dim];
            for (int i = 0; i < ctxs.size(); i++) {
                System.arraycopy(values, i * dim, rows[i], 0, dim);
            }
            return rows;
        }
    }

    private List<Word> tokensToWords(SearchResult search, int sampleCount, double timeOffset) {
        if (search.tokenIds.length == 0 || search.frames.length == 0 || search.t == 0) return Collections.emptyList();
        double chunkDuration = sampleCount / (double) AsrConstants.SAMPLE_RATE;
        double[] ts = new double[search.frames.length];
        for (int i = 0; i < ts.length; i++) ts[i] = (search.frames[i] / (double) search.t) * chunkDuration;
        double avgBpeDur = ts.length >= 2 ? (ts[ts.length - 1] - ts[0]) / (ts.length - 1) : 0.08;
        List<MutableWord> merged = new ArrayList<>();
        MutableWord current = null;
        for (int j = 0; j < search.tokenIds.length; j++) {
            String token = token(search.tokenIds[j]).toLowerCase(new java.util.Locale("vi", "VN"));
            double localStart = ts[Math.min(j, ts.length - 1)];
            double localEnd = j < ts.length - 1 ? ts[j + 1] : localStart + avgBpeDur;
            double prob = Math.exp(j < search.tokenLogProbs.length ? search.tokenLogProbs[j] : 0.0);
            boolean startsWord = token.startsWith(" ") || token.startsWith("\u2581");
            String clean = token.replaceFirst("^[ \\u2581]+", "");
            if (startsWord || current == null) {
                if (current != null) merged.add(current);
                current = new MutableWord(clean, localStart + timeOffset, localEnd + timeOffset, localStart, localEnd, prob);
            } else {
                current.text += token;
                current.end = localEnd + timeOffset;
                current.localEnd = localEnd;
                current.probSum += prob;
                current.probCount += 1;
            }
        }
        if (current != null) merged.add(current);
        List<Word> words = new ArrayList<>();
        for (int i = 0; i < merged.size(); i++) {
            MutableWord word = merged.get(i);
            double estimatedEnd = word.start + avgBpeDur;
            if (i < merged.size() - 1) estimatedEnd = Math.min(estimatedEnd, merged.get(i + 1).start);
            estimatedEnd = Math.min(estimatedEnd, word.start + WORD_ASSIGN_MAX_DURATION_SECONDS);
            if (estimatedEnd <= word.start) estimatedEnd = word.start + Math.min(0.01, WORD_ASSIGN_MAX_DURATION_SECONDS);
            if (!word.text.isEmpty()) words.add(new Word(word.text, word.start, estimatedEnd, word.probSum / word.probCount));
        }
        return words;
    }

    private Map<String, OnnxTensor> mapOf(String k1, OnnxTensor v1, String k2, OnnxTensor v2) {
        Map<String, OnnxTensor> map = new HashMap<>();
        map.put(k1, v1);
        map.put(k2, v2);
        return map;
    }

    private float[] copyFloat(OnnxTensor tensor) {
        FloatBuffer buffer = tensor.getFloatBuffer();
        float[] output = new float[buffer.remaining()];
        buffer.get(output);
        return output;
    }

    private void insertTopK(double[] scores, int[] indexes, double score, int index) {
        int k = scores.length;
        if (score <= scores[k - 1]) return;
        int pos = k - 1;
        while (pos > 0 && score > scores[pos - 1]) {
            scores[pos] = scores[pos - 1];
            indexes[pos] = indexes[pos - 1];
            pos -= 1;
        }
        scores[pos] = score;
        indexes[pos] = index;
    }

    private double logAdd(double a, double b) {
        double x = Math.max(a, b);
        double y = Math.min(a, b);
        double diff = y - x;
        return diff < -36.0 ? x : x + Math.log1p(Math.exp(diff));
    }

    private int[] append(int[] input, int value) {
        int[] output = Arrays.copyOf(input, input.length + 1);
        output[input.length] = value;
        return output;
    }

    private double[] append(double[] input, double value) {
        double[] output = Arrays.copyOf(input, input.length + 1);
        output[input.length] = value;
        return output;
    }

    private String join(int[] values) {
        StringBuilder builder = new StringBuilder();
        for (int i = 0; i < values.length; i++) {
            if (i > 0) builder.append(',');
            builder.append(values[i]);
        }
        return builder.toString();
    }

    @Override
    public void close() throws Exception {
        encoder.close();
        decoder.close();
        joiner.close();
    }

    private static final class Hypothesis {
        final int[] ys;
        double lp;
        final int[] frames;
        final double[] tokenLogProbs;
        final ContextGraph.State contextState;

        Hypothesis(int[] ys, double lp, int[] frames, double[] tokenLogProbs, ContextGraph.State contextState) {
            this.ys = ys;
            this.lp = lp;
            this.frames = frames;
            this.tokenLogProbs = tokenLogProbs;
            this.contextState = contextState;
        }
    }

    private static final class SearchResult {
        final int[] tokenIds;
        final int[] frames;
        final double[] tokenLogProbs;
        final int t;

        SearchResult(int[] tokenIds, int[] frames, double[] tokenLogProbs, int t) {
            this.tokenIds = tokenIds;
            this.frames = frames;
            this.tokenLogProbs = tokenLogProbs;
            this.t = t;
        }
    }

    private static final class MutableWord {
        String text;
        double start;
        double end;
        double localStart;
        double localEnd;
        double probSum;
        int probCount;

        MutableWord(String text, double start, double end, double localStart, double localEnd, double prob) {
            this.text = text;
            this.start = start;
            this.end = end;
            this.localStart = localStart;
            this.localEnd = localEnd;
            this.probSum = prob;
            this.probCount = 1;
        }
    }
}
