package com.asrvn.offline.pipeline;

import android.content.Context;

import java.io.File;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;

public final class PipelineCheckpointStore {
    private final File root;

    public PipelineCheckpointStore(Context context) {
        root = new File(context.getApplicationContext().getFilesDir(), "checkpoints");
        if (!root.exists()) root.mkdirs();
    }

    public void writeStage(String itemId, String stage, String json) throws Exception {
        File dir = new File(root, itemId);
        if (!dir.exists() && !dir.mkdirs()) throw new IllegalStateException("Cannot create checkpoint directory.");
        Files.write(new File(dir, stage + ".json").toPath(), json.getBytes(StandardCharsets.UTF_8));
    }

    public String readStage(String itemId, String stage) throws Exception {
        File file = new File(new File(root, itemId), stage + ".json");
        return file.isFile() ? new String(Files.readAllBytes(file.toPath()), StandardCharsets.UTF_8) : null;
    }

    public void clear(String itemId) {
        deleteRecursively(new File(root, itemId));
    }

    private void deleteRecursively(File file) {
        if (file == null || !file.exists()) return;
        File[] children = file.listFiles();
        if (children != null) {
            for (File child : children) deleteRecursively(child);
        }
        file.delete();
    }
}
