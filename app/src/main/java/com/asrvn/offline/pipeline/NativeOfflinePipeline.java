package com.asrvn.offline.pipeline;

import android.content.Context;
import android.net.Uri;
import android.os.SystemClock;
import android.util.Log;

import ai.onnxruntime.OrtEnvironment;

import com.asrvn.offline.asr.PureOrtRecognizer;
import com.asrvn.offline.audio.CanonicalAudioDecoder;
import com.asrvn.offline.audio.DecodedAudio;
import com.asrvn.offline.config.NativeSettings;
import com.asrvn.offline.diarization.DiarizationEngine;
import com.asrvn.offline.diarization.DiarizationResult;
import com.asrvn.offline.models.ModelFileRegistry;
import com.asrvn.offline.punctuation.PunctuationRestorer;
import com.asrvn.offline.quality.DnsmosEngine;
import com.asrvn.offline.storage.NativeFileLibrary;
import com.asrvn.offline.vad.SileroVadEngine;
import com.asrvn.offline.vad.SpeechSegment;
import com.asrvn.offline.vad.VadResult;

import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;

public final class NativeOfflinePipeline {
    private static final String TAG = "ASRVN";
    private static final int SAMPLE_RATE = 16000;
    private static final int ASR_CHUNK_SAMPLES = 30 * SAMPLE_RATE;
    private static final int ASR_OVERLAP_SAMPLES = 3 * SAMPLE_RATE;
    private static final double WORD_ASSIGN_MAX_DURATION_SECONDS = 0.40;

    public interface ProgressListener {
        void onProgress(String phase, int percent, String message);
        default void onComplete(
                NativeFileLibrary.LibraryItem item,
                String resultJson,
                PureOrtRecognizer.DecodeResult asr,
                DiarizationResult diarization
        ) {
        }
        void onError(Throwable error);
    }

    private final Context context;
    private final NativeFileLibrary library;
    private final NativeSettings settings;
    private final ModelFileRegistry models;
    private final PipelineCheckpointStore checkpoints;
    private final ScreenWakeLock wakeLock;
    private final CanonicalAudioDecoder audioDecoder = new CanonicalAudioDecoder();
    private final ExecutorService executor = Executors.newSingleThreadExecutor();

    public NativeOfflinePipeline(Context context) {
        this.context = context.getApplicationContext();
        this.library = new NativeFileLibrary(this.context);
        this.settings = new NativeSettings(this.context);
        this.models = new ModelFileRegistry(this.context);
        this.checkpoints = new PipelineCheckpointStore(this.context);
        this.wakeLock = new ScreenWakeLock(this.context);
    }

    public void importFile(Uri uri, ProgressListener listener) {
        executor.execute(() -> {
            NativeFileLibrary.LibraryItem item = null;
            long pipelineStart = SystemClock.elapsedRealtime();
            try {
                wakeLock.acquire();
                listener.onProgress("Reading input", 2, "Importing source file");
                item = library.importSource(uri);
                listener.onProgress("Library", 5, "Stored " + item.displayName);

                if (!models.hasRequiredCoreModels()) {
                    listener.onProgress("Models", 8, models.missingSummary());
                    return;
                }

                checkpoints.writeStage(item.id, "import", "{\"status\":\"source_ready\"}");
                long stageStart = SystemClock.elapsedRealtime();
                listener.onProgress("Audio decode", 10, "Decoding audio");
                DecodedAudio decoded = audioDecoder.decode(item.sourceFile);
                listener.onProgress("Audio decode", 14, decoded.decoder + " in " + ms(stageStart));
                checkpoints.writeStage(item.id, "audio", "{\"samples\":" + decoded.samples.length + "}");

                OrtEnvironment env = OrtEnvironment.getEnvironment();
                stageStart = SystemClock.elapsedRealtime();
                listener.onProgress("Silero VAD", 15, settings.bypassVad() ? "Bypassed" : "Running exact recurrent VAD");
                VadResult vad;
                try (SileroVadEngine vadEngine = new SileroVadEngine(env, models.sileroVad(), settings.cpuThreads())) {
                    vad = vadEngine.run(decoded.samples, settings.bypassVad());
                }
                listener.onProgress("Silero VAD", 30, vad.segments.size() + " segment(s), " + ms(stageStart));
                checkpoints.writeStage(item.id, "vad", "{\"segments\":" + vad.segments.size() + "}");
                DnsmosEngine.Score quality = null;
                if (models.hasDnsmosModel()) {
                    long qualityStart = System.nanoTime();
                    listener.onProgress("DNSMOS quality", 32, "Running");
                    try (DnsmosEngine dnsmos = new DnsmosEngine(env, models.dnsmos(), settings.acceleratorEnabled(), settings.cpuThreads())) {
                        quality = dnsmos.computeFromSpeech(concatSpeech(decoded.samples, vad.segments));
                    } catch (Exception error) {
                        listener.onProgress("DNSMOS quality", 32, "Skipped: " + error.getMessage());
                    }
                    if (quality != null) listener.onProgress("DNSMOS quality", 34, quality.ovrl + " (" + quality.provider + "), " + ms(qualityStart));
                }

                stageStart = SystemClock.elapsedRealtime();
                listener.onProgress("ASR", 35, "Loading PureORT encoder/decoder/joiner");
                PureOrtRecognizer.DecodeResult asr;
                try (PureOrtRecognizer recognizer = new PureOrtRecognizer(
                        env,
                        models.asr68FilePaths(),
                        settings.hotwordsText(),
                        1.5,
                        8,
                        settings.acceleratorEnabled(),
                        settings.cpuThreads()
                )) {
                    asr = runAsrChunks(recognizer, decoded.samples, vad.segments, listener);
                }
                listener.onProgress("ASR", 78, asr.words.size() + " word(s), " + ms(stageStart));
                checkpoints.writeStage(item.id, "asr", "{\"chars\":" + asr.text.length() + ",\"words\":" + asr.words.size() + "}");

                DiarizationResult diarization = new DiarizationResult(new ArrayList<>(), "off", "off");
                if (settings.diarizationEnabled()) {
                    stageStart = SystemClock.elapsedRealtime();
                    listener.onProgress("Diarization", 80, "Loading " + settings.speakerModel());
                    try (DiarizationEngine diar = new DiarizationEngine(env, models)) {
                        if (NativeSettings.SPEAKER_PYANNOTE.equals(settings.speakerModel())) {
                            diarization = diar.runPyannote(decoded.samples, settings.acceleratorEnabled(), settings.cpuThreads());
                        } else {
                            diarization = diar.runCampp(decoded.samples, vad.segments, settings.acceleratorEnabled(), settings.cpuThreads());
                        }
                        diarization = DiarizationEngine.postProcess(diarization, asr.words);
                    }
                    listener.onProgress("Diarization", 92, diarization.backend + ", turns=" + diarization.turns.size() + ", " + ms(stageStart));
                    checkpoints.writeStage(item.id, "diarization", "{\"backend\":\"" + diarization.backend + "\",\"turns\":" + diarization.turns.size() + "}");
                }

                PunctuationRestorer.Result punctuation = null;
                if (settings.punctuationEnabled() && !settings.bypassPunctuation() && models.hasPunctuationModel() && asr.text != null && !asr.text.trim().isEmpty()) {
                    stageStart = SystemClock.elapsedRealtime();
                    listener.onProgress("Punctuation", 93, "Loading ViBERT FP32 punctuation");
                    try (PunctuationRestorer restorer = new PunctuationRestorer(env, models, settings.acceleratorEnabled(), settings.cpuThreads())) {
                        double[] pauseHints = buildPunctuationPauseHints(asr.words, diarization.turns);
                        punctuation = restorer.restore(asr, pauseHints, settings.punctuationConfidence(), settings.caseConfidence(), (iteration, done, total) -> {
                            int pct = 93 + (int) Math.round(((iteration / 3.0) + (done / Math.max(1.0, (double) total)) / 3.0) * 5.0);
                            listener.onProgress("Punctuation", Math.min(98, pct), "ViBERT FP32 " + restorer.provider());
                        });
                        asr = punctuation.asr;
                        listener.onProgress("Punctuation", 98, punctuation.executionProvider + ", chunks=" + punctuation.chunks + ", " + ms(stageStart));
                        checkpoints.writeStage(item.id, "punctuation", "{\"provider\":\"" + escape(punctuation.executionProvider) + "\",\"chunks\":" + punctuation.chunks + "}");
                    } catch (Throwable error) {
                        Log.w(TAG, "Punctuation skipped", error);
                        listener.onProgress("Punctuation", 98, "Skipped: " + error.getMessage());
                    }
                }

                String resultJson = resultJson(item, decoded, asr, diarization, quality, punctuation);
                library.saveResult(item.id, resultJson);
                checkpoints.clear(item.id);
                listener.onComplete(item, resultJson, asr, diarization);
                listener.onProgress("Done", 100, "Saved result.asr.json, total " + ms(pipelineStart));
            } catch (Throwable error) {
                if (item != null && settings.resumeAfterKill()) {
                    try {
                        checkpoints.writeStage(item.id, "error", "{\"message\":\"" + escape(error.getMessage()) + "\"}");
                    } catch (Exception ignored) {
                    }
                }
                listener.onError(error);
            } finally {
                wakeLock.release();
            }
        });
    }

    public void debugDumpEncoders(Uri uri, boolean accelerator, ProgressListener listener) {
        executor.execute(() -> {
            long started = SystemClock.elapsedRealtime();
            try {
                wakeLock.acquire();
                listener.onProgress("Audio decode", 5, "Decoding source only");
                java.io.File source = new java.io.File(uri.getPath());
                DecodedAudio decoded = audioDecoder.decode(source);
                listener.onProgress("Audio decode", 35, decoded.decoder + ", samples=" + decoded.samples.length);
                OrtEnvironment env = OrtEnvironment.getEnvironment();
                listener.onProgress("Encoder dump", 45, "Running CAM++ and Pyannote speaker encoders on " + (accelerator ? "accelerator" : "CPU"));
                try (DiarizationEngine diar = new DiarizationEngine(env, models)) {
                    diar.writeEncoderDebugDump(decoded.samples, settings.cpuThreads(), accelerator);
                }
                listener.onProgress("Done", 100, "Saved debug/encoder_dump_" + (accelerator ? "accelerated" : "cpu") + ".json, total " + ms(started));
            } catch (Throwable error) {
                listener.onError(error);
            } finally {
                wakeLock.release();
            }
        });
    }

    public void debugDumpDiarization(Uri uri, boolean accelerator, ProgressListener listener) {
        executor.execute(() -> {
            long started = SystemClock.elapsedRealtime();
            try {
                wakeLock.acquire();
                listener.onProgress("Audio decode", 5, "Decoding source only");
                java.io.File source = new java.io.File(uri.getPath());
                DecodedAudio decoded = audioDecoder.decode(source);
                listener.onProgress("Audio decode", 20, decoded.decoder + ", samples=" + decoded.samples.length);

                OrtEnvironment env = OrtEnvironment.getEnvironment();
                DiarizationResult campp;
                DiarizationResult pyannote;
                try (DiarizationEngine diar = new DiarizationEngine(env, models)) {
                    long stage = SystemClock.elapsedRealtime();
                    listener.onProgress("CAM++ diarization", 30, "Running");
                    campp = DiarizationEngine.postProcess(
                            diar.runCampp(decoded.samples, null, accelerator, settings.cpuThreads()),
                            null);
                    listener.onProgress("CAM++ diarization", 58, "turns=" + campp.turns.size() + ", " + ms(stage));

                    stage = SystemClock.elapsedRealtime();
                    listener.onProgress("Pyannote diarization", 62, "Running");
                    pyannote = DiarizationEngine.postProcess(
                            diar.runPyannote(decoded.samples, accelerator, settings.cpuThreads()),
                            null);
                    listener.onProgress("Pyannote diarization", 92, "turns=" + pyannote.turns.size() + ", " + ms(stage));
                }
                java.io.File debugDir = new java.io.File(context.getExternalFilesDir(null), "debug");
                if (!debugDir.exists() && !debugDir.mkdirs()) throw new IllegalStateException("Cannot create debug dir");
                String json = "{"
                        + "\"sample_count\":" + decoded.samples.length + ","
                        + "\"duration_sec\":" + String.format(java.util.Locale.US, "%.6f", decoded.samples.length / (double) SAMPLE_RATE) + ","
                        + "\"campp\":" + diarizationJson(campp) + ","
                        + "\"pyannote\":" + diarizationJson(pyannote)
                        + "}";
                Files.write(new java.io.File(debugDir, "diarization_result_accelerated.json").toPath(), json.getBytes(StandardCharsets.UTF_8));
                listener.onProgress("Done", 100, "Saved debug/diarization_result_accelerated.json, total " + ms(started));
            } catch (Throwable error) {
                listener.onError(error);
            } finally {
                wakeLock.release();
            }
        });
    }

    private PureOrtRecognizer.DecodeResult runAsrChunks(
            PureOrtRecognizer recognizer,
            float[] samples,
            List<SpeechSegment> segments,
            ProgressListener listener
    ) throws Exception {
        StringBuilder text = new StringBuilder();
        List<PureOrtRecognizer.Word> words = new ArrayList<>();
        List<SpeechSegment> chunks = asrChunks(segments);
        int total = Math.max(1, chunks.size());
        int done = 0;
        for (SpeechSegment segment : chunks) {
            int length = Math.max(0, segment.end - segment.start);
            if (length == 0) continue;
            float[] chunk = new float[length];
            System.arraycopy(samples, segment.start, chunk, 0, length);
            PureOrtRecognizer.DecodeResult part = recognizer.decode(chunk, segment.start / (double) SAMPLE_RATE);
            if (text.length() > 0 && !part.text.isEmpty()) text.append(' ');
            text.append(part.text);
            words.addAll(part.words);
            done += 1;
            int pct = 35 + (int) Math.round((done / (double) total) * 40.0);
            listener.onProgress("ASR", pct, "Decoded chunk " + done + "/" + total);
        }
        return new PureOrtRecognizer.DecodeResult(text.toString().trim(), words, 0, words.size());
    }

    private List<SpeechSegment> asrChunks(List<SpeechSegment> segments) {
        List<SpeechSegment> chunks = new ArrayList<>();
        for (SpeechSegment segment : segments) {
            int start = segment.start;
            int end = segment.end;
            if (end <= start) continue;
            int cursor = start;
            boolean first = true;
            while (cursor < end) {
                int chunkStart = first ? cursor : Math.max(start, cursor - ASR_OVERLAP_SAMPLES);
                int chunkEnd = Math.min(end, cursor + ASR_CHUNK_SAMPLES);
                chunks.add(new SpeechSegment(chunkStart, chunkEnd));
                if (chunkEnd >= end) break;
                cursor = chunkEnd;
                first = false;
            }
        }
        return chunks;
    }

    private String resultJson(
            NativeFileLibrary.LibraryItem item,
            DecodedAudio audio,
            PureOrtRecognizer.DecodeResult asr,
            DiarizationResult diarization,
            DnsmosEngine.Score quality,
            PunctuationRestorer.Result punctuation
    ) {
        return "{"
                + "\"schema\":\"asr-vn-native-0.1\","
                + "\"source\":\"" + escape(item.displayName) + "\","
                + "\"duration_sec\":" + (audio.samples.length / 16000.0) + ","
                + "\"text\":\"" + escape(asr.text) + "\","
                + "\"words\":" + wordsJson(asr.words) + ","
                + "\"segments\":" + segmentsJson(asr.words, diarization.turns) + ","
                + "\"speaker_names\":" + speakerNamesJson(diarization.turns) + ","
                + "\"speaker_colors\":" + speakerColorsJson(diarization.turns) + ","
                + "\"diarization\":" + diarizationJson(diarization) + ","
                + "\"punctuation\":" + punctuationJson(punctuation) + ","
                + "\"quality_info\":" + qualityJson(quality)
                + "}";
    }

    private double[] buildPunctuationPauseHints(List<PureOrtRecognizer.Word> words, List<DiarizationResult.Turn> turns) {
        if (words == null || words.size() < 2) return null;
        double[] hints = new double[words.size()];
        for (int i = 0; i < words.size(); i++) {
            PureOrtRecognizer.Word current = words.get(i);
            PureOrtRecognizer.Word next = i + 1 < words.size() ? words.get(i + 1) : null;
            double currentEnd = wordIntervalEnd(current);
            double gap = next == null ? 1.0 : Math.max(0.0, finiteOr(next.start, currentEnd) - currentEnd);
            if (next != null && turns != null && !turns.isEmpty()) {
                int currentSpeaker = speakerForWord(current, turns);
                int nextSpeaker = speakerForWord(next, turns);
                if (currentSpeaker != nextSpeaker) gap = Math.max(gap, 1.0);
            }
            hints[i] = gap;
        }
        return hints;
    }

    private float[] concatSpeech(float[] samples, List<SpeechSegment> segments) {
        if (segments == null || segments.isEmpty()) return samples;
        int total = 0;
        for (SpeechSegment segment : segments) total += Math.max(0, Math.min(samples.length, segment.end) - Math.max(0, segment.start));
        float[] out = new float[total];
        int offset = 0;
        for (SpeechSegment segment : segments) {
            int start = Math.max(0, segment.start);
            int end = Math.min(samples.length, segment.end);
            int length = Math.max(0, end - start);
            if (length > 0) {
                System.arraycopy(samples, start, out, offset, length);
                offset += length;
            }
        }
        return out;
    }

    private String qualityJson(DnsmosEngine.Score quality) {
        if (quality == null) return "null";
        return "{"
                + "\"dnsmos_sig\":" + String.format(java.util.Locale.US, "%.2f", quality.sig) + ","
                + "\"dnsmos_bak\":" + String.format(java.util.Locale.US, "%.2f", quality.bak) + ","
                + "\"dnsmos_ovrl\":" + String.format(java.util.Locale.US, "%.2f", quality.ovrl) + ","
                + "\"provider\":\"" + escape(quality.provider) + "\""
                + "}";
    }

    private String punctuationJson(PunctuationRestorer.Result punctuation) {
        if (punctuation == null) return "null";
        return "{"
                + "\"model\":\"vibert-capu-fp32\","
                + "\"executionProvider\":\"" + escape(punctuation.executionProvider) + "\","
                + "\"punctuationLevel\":" + settings.punctuationLevel() + ","
                + "\"caseLevel\":" + settings.caseLevel() + ","
                + "\"punctuationConfidence\":" + String.format(java.util.Locale.US, "%.6f", settings.punctuationConfidence()) + ","
                + "\"caseConfidence\":" + String.format(java.util.Locale.US, "%.6f", settings.caseConfidence()) + ","
                + "\"elapsed\":" + String.format(java.util.Locale.US, "%.3f", punctuation.elapsedSeconds) + ","
                + "\"chunks\":" + punctuation.chunks
                + "}";
    }

    private String segmentsJson(List<PureOrtRecognizer.Word> words, List<DiarizationResult.Turn> turns) {
        StringBuilder json = new StringBuilder("[");
        if (words == null || words.isEmpty()) {
            json.append(']');
            return json.toString();
        }
        int currentSpeaker = speakerForWord(words.get(0), turns);
        int segmentSpeaker = -1;
        List<PureOrtRecognizer.Word> group = new ArrayList<>();
        for (PureOrtRecognizer.Word word : words) {
            int speaker = speakerForWord(word, turns);
            if (!group.isEmpty() && (speaker != currentSpeaker || shouldFlushSentence(group))) {
                appendTextSegment(json, group, currentSpeaker, segmentSpeaker);
                segmentSpeaker = currentSpeaker;
                group.clear();
            }
            group.add(word);
            currentSpeaker = speaker;
        }
        if (!group.isEmpty()) appendTextSegment(json, group, currentSpeaker, segmentSpeaker);
        json.append(']');
        return json.toString();
    }

    private void appendTextSegment(StringBuilder json, List<PureOrtRecognizer.Word> group, int speaker, int previousSpeaker) {
        if (speaker != previousSpeaker) {
            if (json.length() > 1) json.append(',');
            json.append('{')
                    .append("\"type\":\"speaker\",")
                    .append("\"speaker_id\":").append(speaker).append(',')
                    .append("\"speaker\":\"Người nói ").append(speaker + 1).append("\"")
                    .append('}');
        }
        PureOrtRecognizer.Word first = group.get(0);
        PureOrtRecognizer.Word last = group.get(group.size() - 1);
        StringBuilder text = new StringBuilder();
        for (PureOrtRecognizer.Word word : group) {
            if (text.length() > 0) text.append(' ');
            text.append(word.text);
        }
        if (json.length() > 1) json.append(',');
        json.append('{')
                .append("\"type\":\"text\",")
                .append("\"speaker_id\":").append(speaker).append(',')
                .append("\"start_time\":").append(String.format(java.util.Locale.US, "%.3f", first.start)).append(',')
                .append("\"end_time\":").append(String.format(java.util.Locale.US, "%.3f", Math.max(last.end, first.start + 0.01))).append(',')
                .append("\"text\":\"").append(escape(text.toString())).append("\",")
                .append("\"raw_words\":").append(wordsJson(group))
                .append('}');
    }

    private boolean shouldFlushSentence(List<PureOrtRecognizer.Word> group) {
        if (group.size() >= 34) return true;
        if (group.size() < 10) return false;
        String text = group.get(group.size() - 1).text;
        return text.endsWith(".") || text.endsWith("?") || text.endsWith("!") || text.endsWith(":");
    }

    private int speakerForWord(PureOrtRecognizer.Word word, List<DiarizationResult.Turn> turns) {
        if (turns == null || turns.isEmpty()) return 0;
        double start = wordIntervalStart(word);
        double end = wordIntervalEnd(word);
        double center = (start + end) * 0.5;
        DiarizationResult.Turn best = null;
        double bestOverlap = 0.0;
        double bestCenterDistance = Double.POSITIVE_INFINITY;
        for (DiarizationResult.Turn turn : turns) {
            double overlap = intervalOverlap(start, end, turn.start, turn.end);
            if (overlap <= 0.0) continue;
            double centerDistance = Math.abs(((turn.start + turn.end) * 0.5) - center);
            if (overlap > bestOverlap || (overlap == bestOverlap && centerDistance < bestCenterDistance)) {
                best = turn;
                bestOverlap = overlap;
                bestCenterDistance = centerDistance;
            }
        }
        if (best != null) return Math.max(0, best.speaker);

        DiarizationResult.Turn previous = null;
        DiarizationResult.Turn next = null;
        for (DiarizationResult.Turn turn : turns) {
            if (turn.end <= center) {
                if (previous == null || turn.end > previous.end) previous = turn;
            } else if (turn.start >= center) {
                if (next == null || turn.start < next.start) next = turn;
            }
        }
        if (previous != null && next != null) {
            double prevDistance = center - previous.end;
            double nextDistance = next.start - center;
            return Math.max(0, prevDistance <= nextDistance ? previous.speaker : next.speaker);
        }
        if (previous != null) return Math.max(0, previous.speaker);
        if (next != null) return Math.max(0, next.speaker);
        return 0;
    }

    private double wordIntervalStart(PureOrtRecognizer.Word word) {
        double start = finiteOr(word.start, 0.0);
        double end = wordEndOrStart(word.end, start);
        return end < start ? end : start;
    }

    private double wordIntervalEnd(PureOrtRecognizer.Word word) {
        double start = finiteOr(word.start, 0.0);
        double end = wordEndOrStart(word.end, start);
        if (end < start) {
            double tmp = start;
            start = end;
            end = tmp;
        }
        end = Math.min(end, start + WORD_ASSIGN_MAX_DURATION_SECONDS);
        return end <= start ? start + WORD_ASSIGN_MAX_DURATION_SECONDS : end;
    }

    private double wordEndOrStart(double value, double start) {
        double end = finiteOr(value, start);
        return end == 0.0 ? start : end;
    }

    private double finiteOr(double value, double fallback) {
        return Double.isNaN(value) || Double.isInfinite(value) ? fallback : value;
    }

    private double intervalOverlap(double startA, double endA, double startB, double endB) {
        return Math.max(0.0, Math.min(endA, endB) - Math.max(startA, startB));
    }

    private String speakerNamesJson(List<DiarizationResult.Turn> turns) {
        StringBuilder json = new StringBuilder("{");
        if (turns != null) {
            boolean first = true;
            java.util.HashSet<Integer> seen = new java.util.HashSet<>();
            for (DiarizationResult.Turn turn : turns) {
                int speaker = Math.max(0, turn.speaker);
                if (!seen.add(speaker)) continue;
                if (!first) json.append(',');
                json.append('"').append(speaker).append("\":\"Người nói ").append(speaker + 1).append('"');
                first = false;
            }
        }
        json.append('}');
        return json.toString();
    }

    private String speakerColorsJson(List<DiarizationResult.Turn> turns) {
        int[] colors = new int[]{
                0x007BFF, 0x28A745, 0xFFC107, 0xDC3545,
                0x6F42C1, 0x17A2B8, 0xFD7E14, 0xE83E8C
        };
        StringBuilder json = new StringBuilder("{");
        if (turns != null) {
            boolean first = true;
            java.util.HashSet<Integer> seen = new java.util.HashSet<>();
            for (DiarizationResult.Turn turn : turns) {
                int speaker = Math.max(0, turn.speaker);
                if (!seen.add(speaker)) continue;
                if (!first) json.append(',');
                json.append('"').append(speaker).append("\":\"")
                        .append(String.format(java.util.Locale.US, "#%06X", colors[speaker % colors.length]))
                        .append('"');
                first = false;
            }
        }
        json.append('}');
        return json.toString();
    }

    private String diarizationJson(DiarizationResult diarization) {
        StringBuilder json = new StringBuilder("{");
        json.append("\"backend\":\"").append(escape(diarization.backend)).append("\",")
                .append("\"provider\":\"").append(escape(diarization.provider)).append("\",")
                .append("\"turn_count\":").append(diarization.turns.size()).append(',')
                .append("\"turns\":[");
        for (int i = 0; i < diarization.turns.size(); i++) {
            DiarizationResult.Turn turn = diarization.turns.get(i);
            if (i > 0) json.append(',');
            json.append('{')
                    .append("\"start\":").append(String.format(java.util.Locale.US, "%.3f", turn.start)).append(',')
                    .append("\"end\":").append(String.format(java.util.Locale.US, "%.3f", turn.end)).append(',')
                    .append("\"speaker\":").append(Math.max(0, turn.speaker))
                    .append('}');
        }
        json.append("]}");
        return json.toString();
    }

    private String wordsJson(List<PureOrtRecognizer.Word> words) {
        StringBuilder json = new StringBuilder("[");
        for (int i = 0; i < words.size(); i++) {
            PureOrtRecognizer.Word word = words.get(i);
            if (i > 0) json.append(',');
            json.append('{')
                    .append("\"text\":\"").append(escape(word.text)).append("\",")
                    .append("\"start\":").append(String.format(java.util.Locale.US, "%.3f", word.start)).append(',')
                    .append("\"end\":").append(String.format(java.util.Locale.US, "%.3f", word.end)).append(',')
                    .append("\"probability\":").append(String.format(java.util.Locale.US, "%.4f", word.probability))
                    .append('}');
        }
        json.append(']');
        return json.toString();
    }

    private String escape(String value) {
        if (value == null) return "";
        return value.replace("\\", "\\\\").replace("\"", "\\\"").replace("\n", "\\n").replace("\r", "");
    }

    private String ms(long startMs) {
        long elapsed = Math.max(0, SystemClock.elapsedRealtime() - startMs);
        return String.format(java.util.Locale.US, "%.2fs", elapsed / 1000.0);
    }
}
