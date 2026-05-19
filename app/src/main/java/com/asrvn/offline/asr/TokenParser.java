package com.asrvn.offline.asr;

import java.nio.charset.StandardCharsets;
import java.util.HashMap;
import java.util.Locale;
import java.util.Map;

public final class TokenParser {
    public static final class ParsedTokens {
        public final String[] idToToken;
        public final Map<String, Integer> pieceToId;

        ParsedTokens(String[] idToToken, Map<String, Integer> pieceToId) {
            this.idToToken = idToToken;
            this.pieceToId = pieceToId;
        }
    }

    public static final class BpeVocab {
        public final Map<String, Piece> pieces;
        public final int maxPieceLength;

        BpeVocab(Map<String, Piece> pieces, int maxPieceLength) {
            this.pieces = pieces;
            this.maxPieceLength = maxPieceLength;
        }
    }

    public static final class Piece {
        public final int id;
        public final double score;

        Piece(int id, double score) {
            this.id = id;
            this.score = score;
        }
    }

    public ParsedTokens parseTokens(byte[] bytes) {
        String text = new String(bytes, StandardCharsets.UTF_8);
        Map<Integer, String> byId = new HashMap<>();
        Map<String, Integer> pieceToId = new HashMap<>();
        int maxId = 0;
        for (String raw : text.split("\\R")) {
            String line = raw.trim();
            if (line.isEmpty()) continue;
            String[] parts = line.split("\\s+");
            if (parts.length < 2) continue;
            int id;
            try {
                id = Integer.parseInt(parts[parts.length - 1]);
            } catch (NumberFormatException ignored) {
                continue;
            }
            String piece = line.substring(0, line.lastIndexOf(parts[parts.length - 1])).trim();
            byId.put(id, piece);
            pieceToId.put(piece, id);
            maxId = Math.max(maxId, id);
        }
        String[] idToToken = new String[maxId + 1];
        for (Map.Entry<Integer, String> entry : byId.entrySet()) {
            idToToken[entry.getKey()] = entry.getValue();
        }
        return new ParsedTokens(idToToken, pieceToId);
    }

    public BpeVocab parseBpeVocab(byte[] bytes, Map<String, Integer> pieceToId) {
        String text = new String(bytes, StandardCharsets.UTF_8);
        Map<String, Piece> pieces = new HashMap<>();
        int maxPieceLength = 1;
        for (String raw : text.split("\\R")) {
            String line = raw.trim();
            if (line.isEmpty()) continue;
            int tab = line.lastIndexOf('\t');
            if (tab <= 0) continue;
            String piece = line.substring(0, tab);
            double score;
            try {
                score = Double.parseDouble(line.substring(tab + 1));
            } catch (NumberFormatException ignored) {
                continue;
            }
            Integer id = pieceToId.get(piece);
            if (id == null) continue;
            pieces.put(piece, new Piece(id, score));
            maxPieceLength = Math.max(maxPieceLength, piece.length());
        }
        return new BpeVocab(pieces, maxPieceLength);
    }

    public static String normalizeHotwordText(String value) {
        if (value == null) return "";
        return value.trim().replaceAll("\\s+", " ").toUpperCase(new Locale("vi", "VN"));
    }
}
