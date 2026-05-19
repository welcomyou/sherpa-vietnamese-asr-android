package com.asrvn.offline.asr;

import java.text.Normalizer;
import java.util.ArrayDeque;
import java.util.ArrayList;
import java.util.List;
import java.util.Locale;
import java.util.Map;

public final class ContextGraph {
    public static final class Advance {
        public final double score;
        public final State state;

        Advance(double score, State state) {
            this.score = score;
            this.state = state;
        }
    }

    public static final class State {
        int token;
        double tokenScore;
        double nodeScore;
        double outputScore;
        boolean isEnd;
        final java.util.HashMap<Integer, State> next = new java.util.HashMap<>();
        State fail;
        State output;

        State(int token) {
            this.token = token;
        }
    }

    public final State root = new State(-1);
    public int phraseCount;

    public ContextGraph() {
        root.fail = root;
    }

    public void build(List<int[]> tokenSequences, List<Double> scores) {
        for (int s = 0; s < tokenSequences.size(); s++) {
            int[] seq = tokenSequences.get(s);
            double score = scores.get(s);
            if (seq.length == 0) continue;
            State node = root;
            for (int j = 0; j < seq.length; j++) {
                int tid = seq[j];
                boolean isLast = j == seq.length - 1;
                State existing = node.next.get(tid);
                if (existing == null) {
                    existing = new State(tid);
                    existing.tokenScore = score;
                    existing.nodeScore = node.nodeScore + score;
                    existing.outputScore = isLast ? node.nodeScore + score : 0.0;
                    existing.isEnd = isLast;
                    node.next.put(tid, existing);
                } else {
                    existing.tokenScore = Math.max(score, existing.tokenScore);
                    existing.nodeScore = node.nodeScore + existing.tokenScore;
                    if (isLast) {
                        existing.isEnd = true;
                        existing.outputScore = existing.nodeScore;
                    } else if (existing.isEnd) {
                        existing.outputScore = existing.nodeScore;
                    }
                }
                node = existing;
            }
            phraseCount += 1;
        }
        fillFailOutput();
    }

    public Advance forwardOneStep(State state, int tokenId) {
        State node;
        double score;
        if (state.next.containsKey(tokenId)) {
            node = state.next.get(tokenId);
            score = node.tokenScore;
        } else {
            node = state.fail;
            while (!node.next.containsKey(tokenId)) {
                node = node.fail;
                if (node.token == -1) break;
            }
            if (node.next.containsKey(tokenId)) node = node.next.get(tokenId);
            score = node.nodeScore - state.nodeScore;
        }

        if (node.outputScore != 0.0) {
            double outputScore = node.nodeScore;
            if (!node.isEnd && node.output != null) outputScore = node.output.nodeScore;
            return new Advance(score + outputScore - node.nodeScore, root);
        }

        return new Advance(score, node);
    }

    public double finalizeScore(State state) {
        return -state.nodeScore;
    }

    private void fillFailOutput() {
        ArrayDeque<State> queue = new ArrayDeque<>();
        for (State child : root.next.values()) {
            child.fail = root;
            queue.add(child);
        }

        while (!queue.isEmpty()) {
            State current = queue.removeFirst();
            for (Map.Entry<Integer, State> entry : current.next.entrySet()) {
                int tid = entry.getKey();
                State child = entry.getValue();
                State fail = current.fail;
                if (fail.next.containsKey(tid)) {
                    fail = fail.next.get(tid);
                } else {
                    fail = fail.fail;
                    while (!fail.next.containsKey(tid)) {
                        fail = fail.fail;
                        if (fail.token == -1) break;
                    }
                    if (fail.next.containsKey(tid)) fail = fail.next.get(tid);
                }

                child.fail = fail;
                State output = fail;
                while (!output.isEnd) {
                    output = output.fail;
                    if (output.token == -1) {
                        output = null;
                        break;
                    }
                }
                child.output = output;
                if (output != null) child.outputScore += output.outputScore;
                queue.add(child);
            }
        }
    }

    public static ContextGraph fromHotwords(String hotwordsText, TokenParser.BpeVocab tokenizer, double defaultScore) {
        List<int[]> tokenSequences = new ArrayList<>();
        List<Double> scores = new ArrayList<>();
        for (String raw : hotwordsText.split("\\R")) {
            String line = raw.trim();
            if (line.isEmpty() || line.startsWith("#")) continue;
            double score = defaultScore;
            int colon = line.lastIndexOf(':');
            if (colon > 0) {
                try {
                    score = Double.parseDouble(line.substring(colon + 1).trim());
                    line = line.substring(0, colon).trim();
                } catch (NumberFormatException ignored) {
                }
            }
            int[] ids = sentencePieceEncode(line, tokenizer);
            if (ids.length == 0) continue;
            tokenSequences.add(ids);
            scores.add(score);
        }
        if (tokenSequences.isEmpty()) return null;
        ContextGraph graph = new ContextGraph();
        graph.build(tokenSequences, scores);
        return graph;
    }

    private static int[] sentencePieceEncode(String text, TokenParser.BpeVocab tokenizer) {
        String normalized = TokenParser.normalizeHotwordText(text);
        normalized = Normalizer.normalize(normalized, Normalizer.Form.NFKC)
                .replaceAll("\\s+", " ")
                .trim()
                .toUpperCase(new Locale("vi", "VN"));
        if (normalized.isEmpty()) return new int[0];
        String input = "\u2581" + normalized.replace(" ", "\u2581");
        int n = input.length();
        double[] best = new double[n + 1];
        int[] prev = new int[n + 1];
        int[] prevId = new int[n + 1];
        for (int i = 0; i <= n; i++) {
            best[i] = Double.NEGATIVE_INFINITY;
            prev[i] = -1;
            prevId[i] = AsrConstants.UNK_ID;
        }
        best[0] = 0.0;
        for (int i = 0; i < n; i++) {
            if (!Double.isFinite(best[i])) continue;
            boolean matched = false;
            int maxLen = Math.min(tokenizer.maxPieceLength, n - i);
            for (int len = 1; len <= maxLen; len++) {
                String piece = input.substring(i, i + len);
                TokenParser.Piece entry = tokenizer.pieces.get(piece);
                if (entry == null) continue;
                matched = true;
                int j = i + len;
                double candidate = best[i] + entry.score;
                if (candidate > best[j]) {
                    best[j] = candidate;
                    prev[j] = i;
                    prevId[j] = entry.id;
                }
            }
            if (!matched) {
                int j = Math.min(n, i + Character.charCount(input.codePointAt(i)));
                double candidate = best[i] - 20.0;
                if (candidate > best[j]) {
                    best[j] = candidate;
                    prev[j] = i;
                    prevId[j] = AsrConstants.UNK_ID;
                }
            }
        }
        if (!Double.isFinite(best[n])) return new int[0];
        ArrayList<Integer> ids = new ArrayList<>();
        for (int pos = n; pos > 0; ) {
            ids.add(0, prevId[pos]);
            pos = prev[pos];
            if (pos < 0) return new int[0];
        }
        int[] output = new int[ids.size()];
        for (int i = 0; i < ids.size(); i++) output[i] = ids.get(i);
        return output;
    }
}
