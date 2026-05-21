package com.asrvn.offline.pipeline;

import android.content.Context;
import android.net.Uri;
import android.os.SystemClock;
import android.util.Log;

import ai.onnxruntime.OrtEnvironment;

import com.asrvn.offline.asr.PureOrtRecognizer;
import com.asrvn.offline.audio.CanonicalAudioDecoder;
import com.asrvn.offline.audio.DecodedAudio;
import com.asrvn.offline.audio.PcmFloatFile;
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
import java.util.concurrent.CancellationException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;

public final class NativeOfflinePipeline {
    private static final String TAG = "ASRVN";
    private static final int SAMPLE_RATE = 16000;
    private static final int ASR_CHUNK_SAMPLES = 30 * SAMPLE_RATE;
    private static final int ASR_OVERLAP_SAMPLES = 3 * SAMPLE_RATE;
    private static final double ASR_OVERLAP_SEC = ASR_OVERLAP_SAMPLES / (double) SAMPLE_RATE;
    private static final int MAX_OVERLAP_WORDS = 100;
    private static final double FUZZY_MATCH_THRESHOLD = 0.8;
    private static final double MIN_MATCH_RATIO = 0.5;
    private static final int QUALITY_SAMPLE_LENGTH = 144160;
    private static final int STREAM_ANALYZE_CHUNK_SAMPLES = 1_000_000;
    private static final int LONG_DIARIZATION_CHUNK_SAMPLES = 10 * 60 * SAMPLE_RATE;
    private static final double WORD_ASSIGN_MAX_DURATION_SECONDS = 0.40;

    private static final class AudioStats {
        final double peak;
        final double rms;
        final double speechRatio;

        AudioStats(double peak, double rms, double speechRatio) {
            this.peak = peak;
            this.rms = rms;
            this.speechRatio = speechRatio;
        }
    }

    private static final class ProbabilityStats {
        final double max;
        final double avg;

        ProbabilityStats(double max, double avg) {
            this.max = max;
            this.avg = avg;
        }
    }

    private static final class AsrChunkResult {
        final List<PureOrtRecognizer.Word> words;
        final double audioStartSec;
        final double audioEndSec;

        AsrChunkResult(List<PureOrtRecognizer.Word> words, double audioStartSec, double audioEndSec) {
            this.words = words == null ? new ArrayList<>() : words;
            this.audioStartSec = audioStartSec;
            this.audioEndSec = audioEndSec;
        }
    }

    private static final class OverlapDecision {
        final int cutIndex;
        final int popCount;

        OverlapDecision(int cutIndex, int popCount) {
            this.cutIndex = Math.max(0, cutIndex);
            this.popCount = Math.max(0, popCount);
        }
    }

    public interface ProgressListener {
        default void onItemCreated(NativeFileLibrary.LibraryItem item) {
        }
        void onProgress(String phase, int percent, String message);
        default void onComplete(
                NativeFileLibrary.LibraryItem item,
                String resultJson,
                PureOrtRecognizer.DecodeResult asr,
                DiarizationResult diarization
        ) {
        }
        default void onCancelled(NativeFileLibrary.LibraryItem item) {
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
    private volatile boolean cancelRequested;

    public NativeOfflinePipeline(Context context) {
        this.context = context.getApplicationContext();
        this.library = new NativeFileLibrary(this.context);
        this.settings = new NativeSettings(this.context);
        this.models = new ModelFileRegistry(this.context);
        this.checkpoints = new PipelineCheckpointStore(this.context);
        this.wakeLock = new ScreenWakeLock(this.context);
    }

    public void importFile(Uri uri, ProgressListener listener) {
        importFile(uri, null, listener);
    }

    public void importFile(Uri uri, String requestedDisplayName, ProgressListener listener) {
        cancelRequested = false;
        executor.execute(() -> {
            NativeFileLibrary.LibraryItem item = null;
            long pipelineStart = SystemClock.elapsedRealtime();
            try {
                wakeLock.acquire();
                listener.onProgress("Reading input", 2, "Importing source file");
                item = library.importSource(uri, requestedDisplayName);
                listener.onItemCreated(item);
                checkCancelled();
                listener.onProgress("Library", 5, "Stored " + item.displayName);
                runPipelineForItem(item, pipelineStart, listener);
            } catch (Throwable error) {
                handlePipelineError(item, error, listener);
            } finally {
                wakeLock.release();
            }
        });
    }

    public void resumeLibraryItem(String itemId, ProgressListener listener) {
        cancelRequested = false;
        executor.execute(() -> {
            NativeFileLibrary.LibraryItem item = null;
            long pipelineStart = SystemClock.elapsedRealtime();
            try {
                wakeLock.acquire();
                item = library.getItem(itemId);
                if (item == null) throw new IllegalStateException("Interrupted item not found.");
                library.markProcessing(item.id);
                listener.onItemCreated(item);
                checkCancelled();
                listener.onProgress("Resume", 3, "Resuming " + item.displayName);
                runPipelineForItem(item, pipelineStart, listener);
            } catch (Throwable error) {
                handlePipelineError(item, error, listener);
            } finally {
                wakeLock.release();
            }
        });
    }

    public void cancel() {
        cancelRequested = true;
    }

    private void checkCancelled() {
        if (cancelRequested || Thread.currentThread().isInterrupted()) {
            throw new CancellationException("Pipeline cancelled.");
        }
    }

    private void runPipelineForItem(
            NativeFileLibrary.LibraryItem item,
            long pipelineStart,
            ProgressListener listener
    ) throws Exception {
        checkCancelled();
        if (!models.hasRequiredCoreModels()) {
            library.markError(item.id);
            listener.onProgress("Models", 8, models.missingSummary());
            return;
        }

        checkpoints.writeStage(item.id, "import", "{\"status\":\"source_ready\"}");
        long stageStart = SystemClock.elapsedRealtime();
        listener.onProgress("Audio decode", 10, "Decoding audio");
        checkCancelled();
        DecodedAudio decoded = audioDecoder.decode(item.sourceFile);
        checkCancelled();
        double audioDecodeElapsed = elapsedSeconds(stageStart);
        listener.onProgress("Audio decode", 14, decoded.decoder + " in " + formatSeconds(audioDecodeElapsed, 2) + "s");
        checkpoints.writeStage(item.id, "audio", "{\"samples\":" + decoded.sampleCount + "}");
        checkCancelled();
        AudioStats audioStats = analyzeAudio(decoded);
        checkCancelled();

        OrtEnvironment env = OrtEnvironment.getEnvironment();
        stageStart = SystemClock.elapsedRealtime();
        listener.onProgress("Silero VAD", 15, settings.bypassVad() ? "Bypassed" : "Running exact recurrent VAD");
        VadResult vad;
        try (SileroVadEngine vadEngine = new SileroVadEngine(env, models.sileroVad(), settings.cpuThreads())) {
            if (decoded.inMemory()) {
                vad = vadEngine.run(decoded.samples, settings.bypassVad());
            } else {
                vad = vadEngine.run(decoded.pcmFile, decoded.sampleCount, settings.bypassVad());
            }
        }
        checkCancelled();
        double vadElapsed = elapsedSeconds(stageStart);
        double vadSpeechSeconds = speechSeconds(vad.segments);
        ProbabilityStats vadProb = probabilityStats(vad.probabilities);
        listener.onProgress("Silero VAD", 30, vad.segments.size() + " segment(s), " + formatSeconds(vadElapsed, 2) + "s");
        checkpoints.writeStage(item.id, "vad", "{\"segments\":" + vad.segments.size() + "}");
        DnsmosEngine.Score quality = null;
        double qualityElapsed = 0.0;
        if (models.hasDnsmosModel()) {
            long qualityStart = SystemClock.elapsedRealtime();
            listener.onProgress("DNSMOS quality", 32, "Running");
            checkCancelled();
            try (DnsmosEngine dnsmos = new DnsmosEngine(env, models.dnsmos(), settings.acceleratorEnabled(), settings.cpuThreads())) {
                if (decoded.inMemory()) {
                    quality = dnsmos.computeFromSpeech(concatSpeech(decoded.samples, vad.segments));
                } else {
                    quality = dnsmos.computeWindows(extractQualityWindows(decoded, vad.segments));
                }
            } catch (Exception error) {
                listener.onProgress("DNSMOS quality", 32, "Skipped: " + error.getMessage());
            }
            checkCancelled();
            qualityElapsed = elapsedSeconds(qualityStart);
            if (quality != null) listener.onProgress("DNSMOS quality", 34, quality.ovrl + " (" + quality.provider + "), " + formatSeconds(qualityElapsed, 2) + "s");
        }

        stageStart = SystemClock.elapsedRealtime();
        listener.onProgress("ASR", 35, "Loading PureORT encoder/decoder/joiner");
        checkCancelled();
        int asrChunkCount = asrChunks(vad.segments).size();
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
            if (decoded.inMemory()) {
                asr = runAsrChunks(recognizer, decoded.samples, vad.segments, listener);
            } else {
                asr = runAsrChunks(recognizer, decoded.pcmFile, decoded.sampleCount, vad.segments, listener);
            }
        }
        checkCancelled();
        double asrElapsed = elapsedSeconds(stageStart);
        listener.onProgress("ASR", 78, asr.words.size() + " word(s), " + formatSeconds(asrElapsed, 2) + "s");
        checkpoints.writeStage(item.id, "asr", "{\"chars\":" + asr.text.length() + ",\"words\":" + asr.words.size() + "}");

        DiarizationResult diarization = new DiarizationResult(new ArrayList<>(), "off", "off");
        double diarizationElapsed = 0.0;
        if (settings.diarizationEnabled()) {
            stageStart = SystemClock.elapsedRealtime();
            listener.onProgress("Diarization", 80, "Loading " + settings.speakerModel());
            checkCancelled();
            try (DiarizationEngine diar = new DiarizationEngine(env, models)) {
                if (!decoded.inMemory()) {
                    diarization = runChunkedLongDiarization(diar, decoded, vad.segments, settings.acceleratorEnabled(), settings.cpuThreads(), listener);
                } else if (NativeSettings.SPEAKER_PYANNOTE.equals(settings.speakerModel())) {
                    diarization = diar.runPyannote(decoded.samples, settings.acceleratorEnabled(), settings.cpuThreads());
                } else {
                    diarization = diar.runCampp(decoded.samples, vad.segments, settings.acceleratorEnabled(), settings.cpuThreads());
                }
                diarization = DiarizationEngine.postProcess(diarization, asr.words);
            }
            checkCancelled();
            diarizationElapsed = elapsedSeconds(stageStart);
            listener.onProgress("Diarization", 92, diarization.backend + ", turns=" + diarization.turns.size() + ", " + formatSeconds(diarizationElapsed, 2) + "s");
            checkpoints.writeStage(item.id, "diarization", "{\"backend\":\"" + diarization.backend + "\",\"turns\":" + diarization.turns.size() + "}");
        }

        PunctuationRestorer.Result punctuation = null;
        double punctuationElapsed = 0.0;
        if (settings.punctuationEnabled() && !settings.bypassPunctuation() && models.hasPunctuationModel() && asr.text != null && !asr.text.trim().isEmpty()) {
            stageStart = SystemClock.elapsedRealtime();
            listener.onProgress("Punctuation", 93, "Loading ViBERT FP32 punctuation");
            checkCancelled();
            try (PunctuationRestorer restorer = new PunctuationRestorer(env, models, settings.acceleratorEnabled(), settings.cpuThreads())) {
                double[] pauseHints = buildPunctuationPauseHints(asr.words, diarization.turns);
                punctuation = restorer.restore(asr, pauseHints, settings.punctuationConfidence(), settings.caseConfidence(), (iteration, done, total) -> {
                    int pct = 93 + (int) Math.round(((iteration / 3.0) + (done / Math.max(1.0, (double) total)) / 3.0) * 5.0);
                    listener.onProgress("Punctuation", Math.min(98, pct), "ViBERT FP32 " + restorer.provider());
                });
                asr = punctuation.asr;
                punctuationElapsed = punctuation.elapsedSeconds > 0 ? punctuation.elapsedSeconds : elapsedSeconds(stageStart);
                listener.onProgress("Punctuation", 98, punctuation.executionProvider + ", chunks=" + punctuation.chunks + ", " + formatSeconds(punctuationElapsed, 2) + "s");
                checkpoints.writeStage(item.id, "punctuation", "{\"provider\":\"" + escape(punctuation.executionProvider) + "\",\"chunks\":" + punctuation.chunks + "}");
            } catch (Throwable error) {
                if (error instanceof CancellationException) throw error;
                Log.w(TAG, "Punctuation skipped", error);
                listener.onProgress("Punctuation", 98, "Skipped: " + error.getMessage());
            }
        }
        checkCancelled();

        String resultJson = resultJson(
                item,
                decoded,
                asr,
                diarization,
                quality,
                punctuation,
                audioStats,
                vad,
                vadSpeechSeconds,
                vadProb,
                audioDecodeElapsed,
                vadElapsed,
                qualityElapsed,
                asrElapsed,
                asrChunkCount,
                diarizationElapsed,
                punctuationElapsed,
                elapsedSeconds(pipelineStart)
        );
        library.saveResult(item.id, resultJson);
        cleanupDecodedAudio(decoded);
        checkpoints.clear(item.id);
        listener.onComplete(item, resultJson, asr, diarization);
        listener.onProgress("Done", 100, "Saved result.asr.json, total " + ms(pipelineStart));
    }

    private void handlePipelineError(NativeFileLibrary.LibraryItem item, Throwable error, ProgressListener listener) {
        if (error instanceof CancellationException) {
            if (item != null) {
                try {
                    library.markCancelled(item.id);
                } catch (Exception ignored) {
                }
            }
            listener.onCancelled(item);
            return;
        }
        if (item != null) {
            try {
                checkpoints.writeStage(item.id, "error", "{\"message\":\"" + escape(error.getMessage()) + "\"}");
                library.markError(item.id);
            } catch (Exception ignored) {
            }
        }
        listener.onError(error);
    }

    private void cleanupDecodedAudio(DecodedAudio decoded) {
        if (decoded == null || decoded.pcmFile == null) return;
        try {
            Files.deleteIfExists(decoded.pcmFile.toPath());
        } catch (Exception ignored) {
        }
    }

    public void debugDumpEncoders(Uri uri, boolean accelerator, ProgressListener listener) {
        executor.execute(() -> {
            long started = SystemClock.elapsedRealtime();
            try {
                wakeLock.acquire();
                listener.onProgress("Audio decode", 5, "Decoding source only");
                java.io.File source = new java.io.File(uri.getPath());
                DecodedAudio decoded = audioDecoder.decode(source);
                listener.onProgress("Audio decode", 35, decoded.decoder + ", samples=" + decoded.sampleCount);
                OrtEnvironment env = OrtEnvironment.getEnvironment();
                listener.onProgress("Encoder dump", 45, "Running CAM++ and Pyannote speaker encoders on " + (accelerator ? "accelerator" : "CPU"));
                try (DiarizationEngine diar = new DiarizationEngine(env, models)) {
                    float[] preview = decoded.inMemory()
                            ? decoded.samples
                            : PcmFloatFile.read(decoded.pcmFile, 0, Math.min(decoded.sampleCount, 10 * SAMPLE_RATE));
                    diar.writeEncoderDebugDump(preview, settings.cpuThreads(), accelerator);
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
                listener.onProgress("Audio decode", 20, decoded.decoder + ", samples=" + decoded.sampleCount);

                OrtEnvironment env = OrtEnvironment.getEnvironment();
                DiarizationResult campp;
                DiarizationResult pyannote;
                try (DiarizationEngine diar = new DiarizationEngine(env, models)) {
                    float[] diarAudio = decoded.inMemory()
                            ? decoded.samples
                            : PcmFloatFile.read(decoded.pcmFile, 0, Math.min(decoded.sampleCount, LONG_DIARIZATION_CHUNK_SAMPLES));
                    long stage = SystemClock.elapsedRealtime();
                    listener.onProgress("CAM++ diarization", 30, "Running");
                    campp = DiarizationEngine.postProcess(
                            diar.runCampp(diarAudio, null, accelerator, settings.cpuThreads()),
                            null);
                    listener.onProgress("CAM++ diarization", 58, "turns=" + campp.turns.size() + ", " + ms(stage));

                    stage = SystemClock.elapsedRealtime();
                    listener.onProgress("Pyannote diarization", 62, "Running");
                    pyannote = DiarizationEngine.postProcess(
                            diar.runPyannote(diarAudio, accelerator, settings.cpuThreads()),
                            null);
                    listener.onProgress("Pyannote diarization", 92, "turns=" + pyannote.turns.size() + ", " + ms(stage));
                }
                java.io.File debugDir = new java.io.File(context.getExternalFilesDir(null), "debug");
                if (!debugDir.exists() && !debugDir.mkdirs()) throw new IllegalStateException("Cannot create debug dir");
                String json = "{"
                        + "\"sample_count\":" + decoded.sampleCount + ","
                        + "\"duration_sec\":" + String.format(java.util.Locale.US, "%.6f", decoded.sampleCount / (double) SAMPLE_RATE) + ","
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
        List<AsrChunkResult> chunkResults = new ArrayList<>();
        List<SpeechSegment> chunks = asrChunks(segments);
        int total = Math.max(1, chunks.size());
        int done = 0;
        for (SpeechSegment segment : chunks) {
            checkCancelled();
            int length = Math.max(0, segment.end - segment.start);
            if (length == 0) continue;
            float[] chunk = new float[length];
            System.arraycopy(samples, segment.start, chunk, 0, length);
            PureOrtRecognizer.DecodeResult part = recognizer.decode(chunk, segment.start / (double) SAMPLE_RATE);
            checkCancelled();
            chunkResults.add(new AsrChunkResult(
                    part.words,
                    segment.start / (double) SAMPLE_RATE,
                    segment.end / (double) SAMPLE_RATE));
            done += 1;
            int pct = 35 + (int) Math.round((done / (double) total) * 40.0);
            listener.onProgress("ASR", pct, "Decoded chunk " + done + "/" + total);
        }
        return mergedAsrResult(chunkResults);
    }

    private PureOrtRecognizer.DecodeResult runAsrChunks(
            PureOrtRecognizer recognizer,
            java.io.File pcmFile,
            int sampleCount,
            List<SpeechSegment> segments,
            ProgressListener listener
    ) throws Exception {
        List<AsrChunkResult> chunkResults = new ArrayList<>();
        List<SpeechSegment> chunks = asrChunks(segments);
        int total = Math.max(1, chunks.size());
        int done = 0;
        for (SpeechSegment segment : chunks) {
            checkCancelled();
            int start = Math.max(0, Math.min(sampleCount, segment.start));
            int end = Math.max(start, Math.min(sampleCount, segment.end));
            int length = end - start;
            if (length == 0) continue;
            float[] chunk = PcmFloatFile.read(pcmFile, start, length);
            checkCancelled();
            PureOrtRecognizer.DecodeResult part = recognizer.decode(chunk, start / (double) SAMPLE_RATE);
            checkCancelled();
            chunkResults.add(new AsrChunkResult(
                    part.words,
                    start / (double) SAMPLE_RATE,
                    end / (double) SAMPLE_RATE));
            done += 1;
            int pct = 35 + (int) Math.round((done / (double) total) * 40.0);
            listener.onProgress("ASR", pct, "Decoded chunk " + done + "/" + total);
        }
        return mergedAsrResult(chunkResults);
    }

    private PureOrtRecognizer.DecodeResult mergedAsrResult(List<AsrChunkResult> chunkResults) {
        List<PureOrtRecognizer.Word> mergedWords = mergeChunksWithOverlap(chunkResults);
        StringBuilder text = new StringBuilder();
        for (PureOrtRecognizer.Word word : mergedWords) {
            if (text.length() > 0) text.append(' ');
            text.append(word.text);
        }
        return new PureOrtRecognizer.DecodeResult(text.toString().trim(), mergedWords, 0, mergedWords.size());
    }

    private List<PureOrtRecognizer.Word> mergeChunksWithOverlap(List<AsrChunkResult> chunkResults) {
        List<PureOrtRecognizer.Word> merged = new ArrayList<>();
        if (chunkResults == null || chunkResults.isEmpty()) return merged;
        for (int chunkIndex = 0; chunkIndex < chunkResults.size(); chunkIndex++) {
            AsrChunkResult chunk = chunkResults.get(chunkIndex);
            if (chunk.words.isEmpty()) continue;
            if (chunkIndex == 0) {
                merged.addAll(chunk.words);
                continue;
            }

            AsrChunkResult previous = chunkResults.get(chunkIndex - 1);
            double actualOverlapSec = previous.audioEndSec - chunk.audioStartSec;
            if (actualOverlapSec <= 0.05) {
                merged.addAll(chunk.words);
                continue;
            }
            double overlapSec = Math.min(ASR_OVERLAP_SEC, actualOverlapSec);
            double previousDuration = Math.max(0.0, previous.audioEndSec - previous.audioStartSec);
            double overlapStartLocal = Math.max(0.0, previousDuration - overlapSec);
            List<PureOrtRecognizer.Word> tailWords = new ArrayList<>();
            for (PureOrtRecognizer.Word word : previous.words) {
                if (localStart(word, previous) >= overlapStartLocal) tailWords.add(word);
            }
            List<PureOrtRecognizer.Word> headWords = new ArrayList<>();
            for (PureOrtRecognizer.Word word : chunk.words) {
                if (localStart(word, chunk) < overlapSec) headWords.add(word);
            }

            OverlapDecision decision = findOverlapAlignment(tailWords, headWords);
            for (int i = 0; i < decision.popCount && !merged.isEmpty(); i++) {
                merged.remove(merged.size() - 1);
            }
            if (decision.cutIndex < chunk.words.size()) {
                merged.addAll(chunk.words.subList(decision.cutIndex, chunk.words.size()));
            }
        }
        return merged;
    }

    private double localStart(PureOrtRecognizer.Word word, AsrChunkResult chunk) {
        return Math.max(0.0, word.start - chunk.audioStartSec);
    }

    private OverlapDecision findOverlapAlignment(
            List<PureOrtRecognizer.Word> tailWords,
            List<PureOrtRecognizer.Word> headWords
    ) {
        if (tailWords == null || tailWords.isEmpty() || headWords == null || headWords.isEmpty()) {
            return new OverlapDecision(0, 0);
        }
        int originalTailLen = tailWords.size();
        int tailFrom = Math.max(0, tailWords.size() - MAX_OVERLAP_WORDS);
        int headTo = Math.min(headWords.size(), MAX_OVERLAP_WORDS);
        List<PureOrtRecognizer.Word> tail = tailWords.subList(tailFrom, tailWords.size());
        List<PureOrtRecognizer.Word> head = headWords.subList(0, headTo);
        List<String> tailNorm = new ArrayList<>();
        List<String> headNorm = new ArrayList<>();
        for (PureOrtRecognizer.Word word : tail) tailNorm.add(normalizeWordForOverlap(word.text));
        for (PureOrtRecognizer.Word word : head) headNorm.add(normalizeWordForOverlap(word.text));

        int bestScore = 0;
        int bestCutIndex = 0;
        int bestPopCount = 0;
        int minOffset = -tailNorm.size() + 1;
        int maxOffset = headNorm.size();
        for (int offset = minOffset; offset < maxOffset; offset++) {
            int score = 0;
            int lastTailMatch = -1;
            int lastHeadMatch = -1;
            for (int i = 0; i < tailNorm.size(); i++) {
                int headIndex = i + offset;
                if (headIndex >= 0 && headIndex < headNorm.size()
                        && wordsMatch(tailNorm.get(i), headNorm.get(headIndex))) {
                    score++;
                    lastTailMatch = i;
                    lastHeadMatch = headIndex;
                }
            }
            int overlapWindow = Math.min(headNorm.size(), tailNorm.size() + offset) - Math.max(0, offset);
            double matchRatio = score / (double) Math.max(1, overlapWindow);
            if (score > bestScore && matchRatio >= MIN_MATCH_RATIO) {
                bestScore = score;
                bestCutIndex = lastHeadMatch + 1;
                bestPopCount = tailNorm.size() - 1 - lastTailMatch;
            }
        }

        int minLen = Math.min(tailNorm.size(), headNorm.size());
        boolean diverged = bestScore < minLen && bestPopCount > 0;
        if (bestScore == 0 || diverged) {
            List<PureOrtRecognizer.Word> divTail = bestScore == 0
                    ? tailWords
                    : tailWords.subList(Math.max(0, tailWords.size() - bestPopCount), tailWords.size());
            List<PureOrtRecognizer.Word> divHead = bestScore == 0
                    ? headWords
                    : headWords.subList(Math.min(bestCutIndex, headWords.size()), headWords.size());
            double tailProb = averageProbability(divTail);
            double headProb = averageProbability(divHead);
            if (tailProb > headProb) return new OverlapDecision(headWords.size(), 0);
            return new OverlapDecision(0, originalTailLen);
        }

        return new OverlapDecision(bestCutIndex, bestPopCount);
    }

    private String normalizeWordForOverlap(String word) {
        String value = word == null ? "" : word.toLowerCase(java.util.Locale.ROOT).trim();
        value = java.text.Normalizer.normalize(value, java.text.Normalizer.Form.NFC);
        return value.replaceAll("[^\\p{L}\\p{Nd}_]", "");
    }

    private boolean wordsMatch(String left, String right) {
        if (left.equals(right)) return true;
        if (left.isEmpty() || right.isEmpty()) return false;
        if (left.length() > 2 && right.length() > 2 && (left.contains(right) || right.contains(left))) return true;
        return similarity(left, right) >= FUZZY_MATCH_THRESHOLD;
    }

    private double similarity(String left, String right) {
        int max = Math.max(left.length(), right.length());
        if (max == 0) return 1.0;
        return 1.0 - (levenshtein(left, right) / (double) max);
    }

    private int levenshtein(String left, String right) {
        int[] prev = new int[right.length() + 1];
        int[] curr = new int[right.length() + 1];
        for (int j = 0; j <= right.length(); j++) prev[j] = j;
        for (int i = 1; i <= left.length(); i++) {
            curr[0] = i;
            for (int j = 1; j <= right.length(); j++) {
                int cost = left.charAt(i - 1) == right.charAt(j - 1) ? 0 : 1;
                curr[j] = Math.min(Math.min(curr[j - 1] + 1, prev[j] + 1), prev[j - 1] + cost);
            }
            int[] tmp = prev;
            prev = curr;
            curr = tmp;
        }
        return prev[right.length()];
    }

    private double averageProbability(List<PureOrtRecognizer.Word> words) {
        if (words == null || words.isEmpty()) return 1.0;
        double sum = 0.0;
        for (PureOrtRecognizer.Word word : words) sum += word.probability;
        return sum / words.size();
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

    private DiarizationResult runChunkedLongDiarization(
            DiarizationEngine diar,
            DecodedAudio audio,
            List<SpeechSegment> vadSegments,
            boolean accelerator,
            int threads,
            ProgressListener listener
    ) throws Exception {
        List<DiarizationResult.Turn> allTurns = new ArrayList<>();
        String backend = NativeSettings.SPEAKER_PYANNOTE.equals(settings.speakerModel())
                ? "pyannote_community1_vbx"
                : "senko_campp_optimized";
        String provider = "chunked-disk";
        int totalChunks = Math.max(1, (int) Math.ceil(audio.sampleCount / (double) LONG_DIARIZATION_CHUNK_SAMPLES));
        int chunkIndex = 0;
        for (int start = 0; start < audio.sampleCount; start += LONG_DIARIZATION_CHUNK_SAMPLES) {
            checkCancelled();
            int length = Math.min(LONG_DIARIZATION_CHUNK_SAMPLES, audio.sampleCount - start);
            float[] chunk = PcmFloatFile.read(audio.pcmFile, start, length);
            checkCancelled();
            DiarizationResult part;
            if (NativeSettings.SPEAKER_PYANNOTE.equals(settings.speakerModel())) {
                part = diar.runPyannote(chunk, accelerator, threads);
            } else {
                part = diar.runCampp(chunk, shiftedSegmentsForChunk(vadSegments, start, start + length), accelerator, threads);
            }
            checkCancelled();
            backend = part.backend + ":chunked-disk";
            provider = part.provider;
            double offsetSeconds = start / (double) SAMPLE_RATE;
            for (DiarizationResult.Turn turn : part.turns) {
                allTurns.add(new DiarizationResult.Turn(turn.start + offsetSeconds, turn.end + offsetSeconds, turn.speaker));
            }
            chunkIndex++;
            int pct = 80 + (int) Math.round((chunkIndex / (double) totalChunks) * 10.0);
            listener.onProgress("Diarization", Math.min(90, pct), "Long-file chunk " + chunkIndex + "/" + totalChunks);
            System.gc();
        }
        return new DiarizationResult(allTurns, backend, provider);
    }

    private List<SpeechSegment> shiftedSegmentsForChunk(List<SpeechSegment> segments, int chunkStart, int chunkEnd) {
        List<SpeechSegment> out = new ArrayList<>();
        if (segments == null) return out;
        for (SpeechSegment segment : segments) {
            int start = Math.max(chunkStart, segment.start);
            int end = Math.min(chunkEnd, segment.end);
            if (end > start) out.add(new SpeechSegment(start - chunkStart, end - chunkStart));
        }
        return out;
    }

    private String resultJson(
            NativeFileLibrary.LibraryItem item,
            DecodedAudio audio,
            PureOrtRecognizer.DecodeResult asr,
            DiarizationResult diarization,
            DnsmosEngine.Score quality,
            PunctuationRestorer.Result punctuation,
            AudioStats audioStats,
            VadResult vad,
            double vadSpeechSeconds,
            ProbabilityStats vadProb,
            double audioDecodeElapsed,
            double vadElapsed,
            double qualityElapsed,
            double asrElapsed,
            int asrChunkCount,
            double diarizationElapsed,
            double punctuationElapsed,
            double totalElapsed
    ) {
        double preprocessingElapsed = audioDecodeElapsed + vadElapsed;
        return "{"
                + "\"schema\":\"asr-vn-native-0.1\","
                + "\"source\":\"" + escape(item.originalName) + "\","
                + "\"display_name\":\"" + escape(item.displayName) + "\","
                + "\"duration_sec\":" + (audio.sampleCount / 16000.0) + ","
                + "\"text\":\"" + escape(asr.text) + "\","
                + "\"words\":" + wordsJson(asr.words) + ","
                + "\"segments\":" + segmentsJson(asr.words, diarization.turns) + ","
                + "\"speaker_names\":" + speakerNamesJson(diarization.turns) + ","
                + "\"speaker_colors\":" + speakerColorsJson(diarization.turns) + ","
                + "\"diarization\":" + diarizationJson(diarization) + ","
                + "\"punctuation\":" + punctuationJson(punctuation) + ","
                + "\"quality_info\":" + qualityJson(quality, asr) + ","
                + "\"timing\":" + timingJson(preprocessingElapsed, asrElapsed, diarizationElapsed, punctuationElapsed, totalElapsed) + ","
                + "\"audio_summary\":" + audioSummaryJson(
                        audio,
                        asr,
                        diarization,
                        punctuation,
                        audioStats,
                        vad,
                        vadSpeechSeconds,
                        vadProb,
                        vadElapsed,
                        qualityElapsed,
                        asrElapsed,
                        asrChunkCount,
                        diarizationElapsed,
                        punctuationElapsed)
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

    private List<float[]> extractQualityWindows(DecodedAudio audio, List<SpeechSegment> segments) throws Exception {
        List<SpeechSegment> sourceSegments = normalizedSpeechSegments(segments, audio.sampleCount);
        long totalSpeech = 0L;
        for (SpeechSegment segment : sourceSegments) totalSpeech += Math.max(0, segment.end - segment.start);
        List<float[]> windows = new ArrayList<>();
        if (totalSpeech < 8000) return windows;
        double[] positions = new double[]{0.15, 0.50, 0.85};
        for (double position : positions) {
            long center = (long) Math.floor(totalSpeech * position);
            long start = Math.max(0L, Math.min(Math.max(0L, totalSpeech - QUALITY_SAMPLE_LENGTH), center - QUALITY_SAMPLE_LENGTH / 2L));
            windows.add(readSpeechTimelineWindow(audio.pcmFile, sourceSegments, start, QUALITY_SAMPLE_LENGTH));
        }
        return windows;
    }

    private List<SpeechSegment> normalizedSpeechSegments(List<SpeechSegment> segments, int sampleCount) {
        List<SpeechSegment> normalized = new ArrayList<>();
        if (segments != null) {
            for (SpeechSegment segment : segments) {
                int start = Math.max(0, Math.min(sampleCount, segment.start));
                int end = Math.max(start, Math.min(sampleCount, segment.end));
                if (end > start) normalized.add(new SpeechSegment(start, end));
            }
        }
        if (normalized.isEmpty() && sampleCount > 0) normalized.add(new SpeechSegment(0, sampleCount));
        return normalized;
    }

    private float[] readSpeechTimelineWindow(
            java.io.File pcmFile,
            List<SpeechSegment> segments,
            long timelineStart,
            int length
    ) throws Exception {
        float[] out = new float[length];
        long cursor = 0L;
        int written = 0;
        long timelineEnd = timelineStart + length;
        for (SpeechSegment segment : segments) {
            int segmentLength = Math.max(0, segment.end - segment.start);
            long next = cursor + segmentLength;
            if (next <= timelineStart) {
                cursor = next;
                continue;
            }
            if (cursor >= timelineEnd) break;
            long localStart = Math.max(timelineStart, cursor);
            long localEnd = Math.min(timelineEnd, next);
            int sourceStart = segment.start + (int) (localStart - cursor);
            int count = (int) Math.max(0, localEnd - localStart);
            if (count > 0) {
                PcmFloatFile.readInto(pcmFile, sourceStart, out, written, count);
                written += count;
            }
            cursor = next;
        }
        return out;
    }

    private AudioStats analyzeAudio(DecodedAudio audio) throws Exception {
        if (audio.inMemory()) return analyzeAudio(audio.samples, SAMPLE_RATE);
        double peak = 0.0;
        double sumSq = 0.0;
        int total = Math.max(0, audio.sampleCount);
        for (int offset = 0; offset < total; offset += STREAM_ANALYZE_CHUNK_SAMPLES) {
            checkCancelled();
            int length = Math.min(STREAM_ANALYZE_CHUNK_SAMPLES, total - offset);
            float[] chunk = PcmFloatFile.read(audio.pcmFile, offset, length);
            for (float sample : chunk) {
                double value = sample;
                double abs = Math.abs(value);
                if (abs > peak) peak = abs;
                sumSq += value * value;
            }
        }
        double rms = Math.sqrt(sumSq / Math.max(1, total));
        int frameSize = Math.max(1, Math.round(SAMPLE_RATE * 0.03f));
        double speechThreshold = Math.max(0.004, rms * 0.6);
        int speechFrames = 0;
        double frameSq = 0.0;
        int frameSamples = 0;
        for (int offset = 0; offset < total; offset += STREAM_ANALYZE_CHUNK_SAMPLES) {
            checkCancelled();
            int length = Math.min(STREAM_ANALYZE_CHUNK_SAMPLES, total - offset);
            float[] chunk = PcmFloatFile.read(audio.pcmFile, offset, length);
            for (float sample : chunk) {
                frameSq += sample * sample;
                frameSamples++;
                if (frameSamples == frameSize) {
                    double frameRms = Math.sqrt(frameSq / frameSamples);
                    if (frameRms >= speechThreshold) speechFrames++;
                    frameSq = 0.0;
                    frameSamples = 0;
                }
            }
        }
        if (frameSamples > 0) {
            double frameRms = Math.sqrt(frameSq / frameSamples);
            if (frameRms >= speechThreshold) speechFrames++;
        }
        int totalFrames = (int) Math.ceil(total / (double) frameSize);
        double speechRatio = totalFrames > 0 ? speechFrames / (double) totalFrames : 0.0;
        return new AudioStats(peak, rms, speechRatio);
    }

    private AudioStats analyzeAudio(float[] samples, int sampleRate) {
        double peak = 0.0;
        double sumSq = 0.0;
        for (float sample : samples) {
            double value = sample;
            double abs = Math.abs(value);
            if (abs > peak) peak = abs;
            sumSq += value * value;
        }
        double rms = Math.sqrt(sumSq / Math.max(1, samples.length));
        int frameSize = Math.max(1, Math.round(sampleRate * 0.03f));
        double speechThreshold = Math.max(0.004, rms * 0.6);
        int speechFrames = 0;
        for (int offset = 0; offset < samples.length; offset += frameSize) {
            double frameSq = 0.0;
            int end = Math.min(samples.length, offset + frameSize);
            for (int i = offset; i < end; i++) {
                frameSq += samples[i] * samples[i];
            }
            double frameRms = Math.sqrt(frameSq / Math.max(1, end - offset));
            if (frameRms >= speechThreshold) speechFrames++;
        }
        int totalFrames = (int) Math.ceil(samples.length / (double) frameSize);
        double speechRatio = totalFrames > 0 ? speechFrames / (double) totalFrames : 0.0;
        return new AudioStats(peak, rms, speechRatio);
    }

    private ProbabilityStats probabilityStats(float[] probabilities) {
        if (probabilities == null || probabilities.length == 0) return new ProbabilityStats(0.0, 0.0);
        double max = 0.0;
        double sum = 0.0;
        for (float probability : probabilities) {
            if (probability > max) max = probability;
            sum += probability;
        }
        return new ProbabilityStats(max, sum / probabilities.length);
    }

    private double speechSeconds(List<SpeechSegment> segments) {
        if (segments == null || segments.isEmpty()) return 0.0;
        long samples = 0L;
        for (SpeechSegment segment : segments) {
            samples += Math.max(0, segment.end - segment.start);
        }
        return samples / (double) SAMPLE_RATE;
    }

    private double asrConfidence(List<PureOrtRecognizer.Word> words) {
        if (words == null || words.isEmpty()) return Double.NaN;
        double sum = 0.0;
        int count = 0;
        for (PureOrtRecognizer.Word word : words) {
            if (!Double.isNaN(word.probability) && !Double.isInfinite(word.probability)) {
                sum += word.probability;
                count++;
            }
        }
        return count == 0 ? Double.NaN : sum / count;
    }

    private int speakerCount(List<DiarizationResult.Turn> turns) {
        if (turns == null || turns.isEmpty()) return 0;
        java.util.HashSet<Integer> speakers = new java.util.HashSet<>();
        for (DiarizationResult.Turn turn : turns) {
            speakers.add(Math.max(0, turn.speaker));
        }
        return speakers.size();
    }

    private String formatBytes(long bytes) {
        String[] units = new String[]{"B", "KB", "MB", "GB"};
        double value = bytes;
        int unit = 0;
        while (value >= 1024.0 && unit < units.length - 1) {
            value /= 1024.0;
            unit++;
        }
        return unit == 0
                ? String.format(java.util.Locale.US, "%.0f %s", value, units[unit])
                : String.format(java.util.Locale.US, "%.1f %s", value, units[unit]);
    }

    private String qualityJson(DnsmosEngine.Score quality, PureOrtRecognizer.DecodeResult asr) {
        double confidence = asrConfidence(asr == null ? null : asr.words);
        if (quality == null && !Double.isFinite(confidence)) return "null";
        StringBuilder json = new StringBuilder("{");
        boolean first = true;
        if (quality != null) {
            json.append("\"dnsmos_sig\":").append(String.format(java.util.Locale.US, "%.2f", quality.sig)).append(',')
                    .append("\"dnsmos_bak\":").append(String.format(java.util.Locale.US, "%.2f", quality.bak)).append(',')
                    .append("\"dnsmos_ovrl\":").append(String.format(java.util.Locale.US, "%.2f", quality.ovrl)).append(',')
                    .append("\"provider\":\"").append(escape(quality.provider)).append('"');
            first = false;
        }
        if (Double.isFinite(confidence)) {
            if (!first) json.append(',');
            json.append("\"asr_confidence\":")
                    .append(String.format(java.util.Locale.US, "%.6f", confidence));
        }
        json.append('}');
        return json.toString();
    }

    private String timingJson(
            double preprocessingElapsed,
            double asrElapsed,
            double diarizationElapsed,
            double punctuationElapsed,
            double totalElapsed
    ) {
        return "{"
                + "\"preprocessing\":" + formatSeconds(preprocessingElapsed, 3) + ","
                + "\"transcription_detail\":" + formatSeconds(asrElapsed, 3) + ","
                + "\"diarization\":" + formatSeconds(diarizationElapsed, 3) + ","
                + "\"punctuation\":" + formatSeconds(punctuationElapsed, 3) + ","
                + "\"overlap_separation\":0,"
                + "\"total\":" + formatSeconds(totalElapsed, 3)
                + "}";
    }

    private String audioSummaryJson(
            DecodedAudio audio,
            PureOrtRecognizer.DecodeResult asr,
            DiarizationResult diarization,
            PunctuationRestorer.Result punctuation,
            AudioStats stats,
            VadResult vad,
            double vadSpeechSeconds,
            ProbabilityStats vadProb,
            double vadElapsed,
            double qualityElapsed,
            double asrElapsed,
            int asrChunkCount,
            double diarizationElapsed,
            double punctuationElapsed
    ) {
        StringBuilder json = new StringBuilder("[");
        appendSummaryItem(json, "Duration", formatSeconds(audio.sampleCount / (double) SAMPLE_RATE, 1) + " s");
        appendSummaryItem(json, "Audio decoder", audio.decoder);
        appendSummaryItem(json, "Original rate", "unknown");
        appendSummaryItem(json, "Offline rate", "16000 Hz mono");
        appendSummaryItem(json, "ASR model", "Zipformer-Vi 2025 (68M)");
        appendSummaryItem(json, "CPU threads", String.valueOf(settings.cpuThreads()));
        appendSummaryItem(json, "Peak / RMS", formatSeconds(stats.peak, 3) + " / " + formatSeconds(stats.rms, 3));
        appendSummaryItem(json, "Energy speech preview", Math.round(stats.speechRatio * 100.0) + "%");
        appendSummaryItem(json, "Silero VAD segments", vad.bypassed ? "bypassed" : String.valueOf(vad.segments.size()));
        appendSummaryItem(json, "VAD speech", formatSeconds(vadSpeechSeconds, 1) + " s");
        appendSummaryItem(json, "VAD prob max / avg", formatSeconds(vadProb.max, 3) + " / " + formatSeconds(vadProb.avg, 3));
        appendSummaryItem(json, "VAD runtime", formatSeconds(vadElapsed, 2) + " s");
        appendSummaryItem(json, "RMS normalize", "off");
        appendSummaryItem(json, "PCM size", formatBytes(audio.sampleCount * 4L));
        appendSummaryItem(json, "ASR chunks", String.valueOf(asrChunkCount));
        appendSummaryItem(json, "ASR speech", formatSeconds(vadSpeechSeconds, 1) + " s");
        appendSummaryItem(json, "ASR runtime", formatSeconds(asrElapsed, 2) + " s");
        double confidence = asrConfidence(asr == null ? null : asr.words);
        appendSummaryItem(json, "ASR confidence", Double.isFinite(confidence) ? formatSeconds(confidence * 100.0, 1) + "%" : "n/a");
        appendSummaryItem(json, "Quality runtime", qualityElapsed > 0 ? formatSeconds(qualityElapsed, 2) + " s" : "n/a");
        appendSummaryItem(json, "Punctuation", punctuation == null ? "no text" : formatSeconds(punctuationElapsed, 2) + " s");
        appendSummaryItem(json, "Punct chunks", punctuation == null ? "0" : String.valueOf(punctuation.chunks));
        appendSummaryItem(json, "Speaker turns", String.valueOf(diarization.turns.size()));
        appendSummaryItem(json, "Speakers", String.valueOf(speakerCount(diarization.turns)));
        appendSummaryItem(json, "Diar backend", diarization.backend);
        appendSummaryItem(json, "Speaker embeds", "0");
        appendSummaryItem(json, "Diar runtime", formatSeconds(diarizationElapsed, 2) + " s");
        appendSummaryItem(json, "Overlap regions", "off");
        appendSummaryItem(json, "Overlap lines", "0");
        appendSummaryItem(json, "Overlap runtime", "0.00 s");
        appendSummaryItem(json, "Transcript chars", String.valueOf(asr == null || asr.text == null ? 0 : asr.text.length()));
        json.append(']');
        return json.toString();
    }

    private void appendSummaryItem(StringBuilder json, String label, String value) {
        if (json.length() > 1) json.append(',');
        json.append('{')
                .append("\"label\":\"").append(escape(label)).append("\",")
                .append("\"value\":\"").append(escape(value)).append("\"")
                .append('}');
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

    private double elapsedSeconds(long startMs) {
        long elapsed = Math.max(0, SystemClock.elapsedRealtime() - startMs);
        return elapsed / 1000.0;
    }

    private String formatSeconds(double seconds, int decimals) {
        return String.format(java.util.Locale.US, "%." + decimals + "f", Math.max(0.0, seconds));
    }

    private String ms(long startMs) {
        long elapsed = Math.max(0, SystemClock.elapsedRealtime() - startMs);
        return String.format(java.util.Locale.US, "%.2fs", elapsed / 1000.0);
    }
}
