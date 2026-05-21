package com.asrvn.offline.models;

import android.content.Context;
import android.content.res.AssetManager;

import com.asrvn.offline.asr.PureOrtRecognizer;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.InputStream;
import java.nio.file.Files;
import java.util.ArrayList;
import java.util.List;

public final class ModelFileRegistry {
    public static final class RemoteModelFile {
        public final String id;
        public final String repo;
        public final String path;
        public final String relativePath;
        public final long bytes;
        public final String directUrl;

        public RemoteModelFile(String id, String repo, String path, String relativePath, long bytes) {
            this(id, repo, path, relativePath, bytes, null);
        }

        public RemoteModelFile(String id, String repo, String path, String relativePath, long bytes, String directUrl) {
            this.id = id;
            this.repo = repo;
            this.path = path;
            this.relativePath = relativePath;
            this.bytes = bytes;
            this.directUrl = directUrl;
        }

        public String url() {
            if (directUrl != null) return directUrl;
            return "https://huggingface.co/" + repo + "/resolve/main/" + path;
        }
    }

    private final File root;
    private final File externalRoot;
    private final AssetManager assets;

    public ModelFileRegistry(Context context) {
        Context app = context.getApplicationContext();
        assets = app.getAssets();
        root = new File(app.getFilesDir(), "models");
        File external = app.getExternalFilesDir(null);
        externalRoot = external == null ? null : new File(external, "models");
    }

    public File root() {
        return externalRoot == null ? root : externalRoot;
    }

    public boolean hasRequiredCoreModels() {
        return file("models/sherpa-onnx-zipformer-vi-2025-04-20/encoder-epoch-12-avg-8.onnx").isFile()
                && file("models/sherpa-onnx-zipformer-vi-2025-04-20/decoder-epoch-12-avg-8.onnx").isFile()
                && file("models/sherpa-onnx-zipformer-vi-2025-04-20/joiner-epoch-12-avg-8.onnx").isFile()
                && file("models/sherpa-onnx-zipformer-vi-2025-04-20/tokens.txt").isFile()
                && file("models/silero-vad/silero_vad_16k_op15.onnx").isFile();
    }

    public boolean hasCamppModel() {
        return file("models/campp-3dspeaker/campplus_cn_en_common_200k.onnx").isFile();
    }

    public boolean hasPyannoteModels() {
        return file("models/pyannote-onnx/segmentation-community-1.onnx").isFile()
                && file("models/pyannote-onnx/embedding_encoder.onnx").isFile()
                && file("models/pyannote-onnx/resnet_seg_1_weight.npy").isFile()
                && file("models/pyannote-onnx/resnet_seg_1_bias.npy").isFile()
                && file("models/pyannote/speaker-diarization-community-1/plda/plda_prepared.npz").isFile();
    }

    public boolean hasDnsmosModel() {
        return file("models/dnsmos/sig_bak_ovr.onnx").isFile();
    }

    public boolean hasPunctuationModel() {
        return file("models/vibert-capu/vibert-capu.onnx").isFile()
                && file("models/vibert-capu/vocab.txt").isFile()
                && file("models/vibert-capu/vocabulary/labels.txt").isFile()
                && file("models/vibert-capu/vocabulary/d_tags.txt").isFile();
    }

    public List<RemoteModelFile> missingDefaultModelFiles() {
        List<RemoteModelFile> files = new ArrayList<>();
        addIfMissing(files, new RemoteModelFile("asr68.encoder", "csukuangfj/sherpa-onnx-zipformer-vi-2025-04-20", "encoder-epoch-12-avg-8.onnx", "models/sherpa-onnx-zipformer-vi-2025-04-20/encoder-epoch-12-avg-8.onnx", 261057692L));
        addIfMissing(files, new RemoteModelFile("asr68.decoder", "csukuangfj/sherpa-onnx-zipformer-vi-2025-04-20", "decoder-epoch-12-avg-8.onnx", "models/sherpa-onnx-zipformer-vi-2025-04-20/decoder-epoch-12-avg-8.onnx", 5165084L));
        addIfMissing(files, new RemoteModelFile("asr68.joiner", "csukuangfj/sherpa-onnx-zipformer-vi-2025-04-20", "joiner-epoch-12-avg-8.onnx", "models/sherpa-onnx-zipformer-vi-2025-04-20/joiner-epoch-12-avg-8.onnx", 4104465L));
        addIfMissing(files, new RemoteModelFile("asr68.tokens", "csukuangfj/sherpa-onnx-zipformer-vi-2025-04-20", "tokens.txt", "models/sherpa-onnx-zipformer-vi-2025-04-20/tokens.txt", 25847L));
        addIfMissing(files, new RemoteModelFile("vad.silero", "snakers4/silero-vad", "src/silero_vad/data/silero_vad_16k_op15.onnx", "models/silero-vad/silero_vad_16k_op15.onnx", 1289603L, "https://raw.githubusercontent.com/snakers4/silero-vad/980b17e9d56463e51393a8d92ded473f1b17896a/src/silero_vad/data/silero_vad_16k_op15.onnx"));
        addIfMissing(files, new RemoteModelFile("speaker.campp", "welcomyou/campplus-3dspeaker-200k-onnx", "campplus_cn_en_common_200k.onnx", "models/campp-3dspeaker/campplus_cn_en_common_200k.onnx", 28283928L));
        addIfMissing(files, new RemoteModelFile("quality.dnsmos", "microsoft/DNS-Challenge", "DNSMOS/DNSMOS/sig_bak_ovr.onnx", "models/dnsmos/sig_bak_ovr.onnx", 1157965L, "https://github.com/microsoft/DNS-Challenge/raw/master/DNSMOS/DNSMOS/sig_bak_ovr.onnx"));
        for (RemoteModelFile item : punctuationRemoteFiles()) addIfMissing(files, item);
        for (RemoteModelFile item : pyannoteRemoteFiles()) addIfMissing(files, item);
        return files;
    }

    public List<RemoteModelFile> missingPyannoteModelFiles() {
        List<RemoteModelFile> files = new ArrayList<>();
        for (RemoteModelFile item : pyannoteRemoteFiles()) addIfMissing(files, item);
        return files;
    }

    public File writableFile(String relative) {
        String clean = relative.startsWith("models/") ? relative.substring("models/".length()) : relative;
        return new File(root(), clean);
    }

    public String missingSummary() {
        if (hasRequiredCoreModels()) return "Models ready";
        return "Missing offline models. Build a full APK with scripts/sync-models.ps1 or use the in-app downloader where available.";
    }

    public int installBundledModelsIfPresent() {
        int copied = 0;
        for (String relative : bundledModelPaths()) {
            if (file(relative).isFile()) continue;
            try (InputStream input = assets.open(relative)) {
                File target = writableFile(relative);
                File dir = target.getParentFile();
                if (dir != null && !dir.exists() && !dir.mkdirs()) continue;
                Files.copy(input, target.toPath());
                copied++;
            } catch (FileNotFoundException ignored) {
                // No bundled asset for this model. The HF downloader remains the fallback.
            } catch (Exception ignored) {
            }
        }
        return copied;
    }

    public PureOrtRecognizer.ModelFiles asr68Files() throws Exception {
        return new PureOrtRecognizer.ModelFiles(
                read("models/sherpa-onnx-zipformer-vi-2025-04-20/encoder-epoch-12-avg-8.onnx"),
                read("models/sherpa-onnx-zipformer-vi-2025-04-20/decoder-epoch-12-avg-8.onnx"),
                read("models/sherpa-onnx-zipformer-vi-2025-04-20/joiner-epoch-12-avg-8.onnx"),
                read("models/sherpa-onnx-zipformer-vi-2025-04-20/tokens.txt"),
                optionalRead("models/sherpa-onnx-zipformer-vi-2025-04-20/bpe.vocab")
        );
    }

    public PureOrtRecognizer.ModelFilePaths asr68FilePaths() throws Exception {
        return new PureOrtRecognizer.ModelFilePaths(
                requireFile("models/sherpa-onnx-zipformer-vi-2025-04-20/encoder-epoch-12-avg-8.onnx").getAbsolutePath(),
                requireFile("models/sherpa-onnx-zipformer-vi-2025-04-20/decoder-epoch-12-avg-8.onnx").getAbsolutePath(),
                requireFile("models/sherpa-onnx-zipformer-vi-2025-04-20/joiner-epoch-12-avg-8.onnx").getAbsolutePath(),
                read("models/sherpa-onnx-zipformer-vi-2025-04-20/tokens.txt"),
                optionalRead("models/sherpa-onnx-zipformer-vi-2025-04-20/bpe.vocab")
        );
    }

    public byte[] sileroVad() throws Exception {
        return read("models/silero-vad/silero_vad_16k_op15.onnx");
    }

    public byte[] campp() throws Exception {
        return read("models/campp-3dspeaker/campplus_cn_en_common_200k.onnx");
    }

    public byte[] pyannoteSegmentation() throws Exception {
        return read("models/pyannote-onnx/segmentation-community-1.onnx");
    }

    public byte[] pyannoteEmbeddingEncoder() throws Exception {
        return read("models/pyannote-onnx/embedding_encoder.onnx");
    }

    public byte[] pyannoteResnetWeight() throws Exception {
        return read("models/pyannote-onnx/resnet_seg_1_weight.npy");
    }

    public byte[] pyannoteResnetBias() throws Exception {
        return read("models/pyannote-onnx/resnet_seg_1_bias.npy");
    }

    public byte[] pyannotePldaPrepared() throws Exception {
        return read("models/pyannote/speaker-diarization-community-1/plda/plda_prepared.npz");
    }

    public byte[] dnsmos() throws Exception {
        return read("models/dnsmos/sig_bak_ovr.onnx");
    }

    public byte[] punctuationModel() throws Exception {
        return read("models/vibert-capu/vibert-capu.onnx");
    }

    public String punctuationModelPath() {
        return requireFile("models/vibert-capu/vibert-capu.onnx").getAbsolutePath();
    }

    public byte[] punctuationVocab() throws Exception {
        return read("models/vibert-capu/vocab.txt");
    }

    public byte[] punctuationLabels() throws Exception {
        return read("models/vibert-capu/vocabulary/labels.txt");
    }

    public byte[] punctuationDetectionTags() throws Exception {
        return read("models/vibert-capu/vocabulary/d_tags.txt");
    }

    private List<RemoteModelFile> punctuationRemoteFiles() {
        List<RemoteModelFile> files = new ArrayList<>();
        files.add(new RemoteModelFile("punct.vibert_fp32", "welcomyou/vibert-capu-onnx", "vibert-capu.onnx", "models/vibert-capu/vibert-capu.onnx", 459449409L));
        files.add(new RemoteModelFile("punct.vocab", "welcomyou/vibert-capu-onnx", "vocab.txt", "models/vibert-capu/vocab.txt", 254703L));
        files.add(new RemoteModelFile("punct.labels", "welcomyou/vibert-capu-onnx", "vocabulary/labels.txt", "models/vibert-capu/vocabulary/labels.txt", 248L));
        files.add(new RemoteModelFile("punct.d_tags", "welcomyou/vibert-capu-onnx", "vocabulary/d_tags.txt", "models/vibert-capu/vocabulary/d_tags.txt", 42L));
        files.add(new RemoteModelFile("punct.non_padded_namespaces", "welcomyou/vibert-capu-onnx", "vocabulary/non_padded_namespaces.txt", "models/vibert-capu/vocabulary/non_padded_namespaces.txt", 14L));
        return files;
    }

    private List<RemoteModelFile> pyannoteRemoteFiles() {
        List<RemoteModelFile> files = new ArrayList<>();
        files.add(new RemoteModelFile("speaker.pyannote_seg", "csukuangfj/sherpa-onnx-pyannote-segmentation-3-0", "model.onnx", "models/pyannote-onnx/segmentation-community-1.onnx", 5992913L));
        files.add(new RemoteModelFile("speaker.pyannote_embedding_encoder", "welcomyou/pyannote-community-1-onnx-split", "embedding_encoder.onnx", "models/pyannote-onnx/embedding_encoder.onnx", 21306024L));
        files.add(new RemoteModelFile("speaker.pyannote_resnet_weight", "welcomyou/pyannote-community-1-onnx-split", "resnet_seg_1_weight.npy", "models/pyannote-onnx/resnet_seg_1_weight.npy", 5243008L));
        files.add(new RemoteModelFile("speaker.pyannote_resnet_bias", "welcomyou/pyannote-community-1-onnx-split", "resnet_seg_1_bias.npy", "models/pyannote-onnx/resnet_seg_1_bias.npy", 1152L));
        return files;
    }

    private String[] bundledModelPaths() {
        return new String[]{
                "models/sherpa-onnx-zipformer-vi-2025-04-20/encoder-epoch-12-avg-8.onnx",
                "models/sherpa-onnx-zipformer-vi-2025-04-20/decoder-epoch-12-avg-8.onnx",
                "models/sherpa-onnx-zipformer-vi-2025-04-20/joiner-epoch-12-avg-8.onnx",
                "models/sherpa-onnx-zipformer-vi-2025-04-20/tokens.txt",
                "models/sherpa-onnx-zipformer-vi-2025-04-20/bpe.vocab",
                "models/sherpa-onnx-zipformer-vi-2025-04-20/bpe.model",
                "models/silero-vad/silero_vad_16k_op15.onnx",
                "models/campp-3dspeaker/campplus_cn_en_common_200k.onnx",
                "models/dnsmos/sig_bak_ovr.onnx",
                "models/vibert-capu/vibert-capu.onnx",
                "models/vibert-capu/vocab.txt",
                "models/vibert-capu/vocabulary/labels.txt",
                "models/vibert-capu/vocabulary/d_tags.txt",
                "models/vibert-capu/vocabulary/non_padded_namespaces.txt",
                "models/pyannote-onnx/segmentation-community-1.onnx",
                "models/pyannote-onnx/embedding_encoder.onnx",
                "models/pyannote-onnx/resnet_seg_1_weight.npy",
                "models/pyannote-onnx/resnet_seg_1_bias.npy",
                "models/pyannote/speaker-diarization-community-1/plda/plda_prepared.npz",
                "models/pyannote/speaker-diarization-community-1/plda/plda.npz",
                "models/pyannote/speaker-diarization-community-1/plda/xvec_transform.npz"
        };
    }

    private void addIfMissing(List<RemoteModelFile> files, RemoteModelFile item) {
        if (!file(item.relativePath).isFile()) files.add(item);
    }

    private byte[] read(String relative) throws Exception {
        File file = requireFile(relative);
        return Files.readAllBytes(file.toPath());
    }

    private File requireFile(String relative) {
        File file = file(relative);
        if (!file.isFile()) throw new IllegalStateException("Missing model file: " + file.getAbsolutePath());
        return file;
    }

    private byte[] optionalRead(String relative) throws Exception {
        File file = file(relative);
        return file.isFile() ? Files.readAllBytes(file.toPath()) : null;
    }

    private File file(String relative) {
        String clean = relative.startsWith("models/") ? relative.substring("models/".length()) : relative;
        File external = externalRoot == null ? null : new File(externalRoot, clean);
        if (external != null && external.exists()) return external;
        return new File(root, clean);
    }
}
