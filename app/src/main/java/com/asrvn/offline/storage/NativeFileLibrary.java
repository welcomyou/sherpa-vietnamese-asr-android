package com.asrvn.offline.storage;

import android.content.Context;
import android.database.Cursor;
import android.net.Uri;
import android.provider.OpenableColumns;

import org.json.JSONObject;

import java.io.File;
import java.io.FileOutputStream;
import java.io.InputStream;
import java.nio.file.AtomicMoveNotSupportedException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.StandardCopyOption;
import java.util.ArrayList;
import java.util.List;
import java.util.Locale;
import java.util.UUID;

public final class NativeFileLibrary {
    public static final String STATUS_PROCESSING = "processing";
    public static final String STATUS_COMPLETE = "complete";
    public static final String STATUS_ERROR = "error";
    public static final String STATUS_CANCELLED = "cancelled";

    public static final class LibraryItem {
        public final String id;
        public final String displayName;
        public final String originalName;
        public final String status;
        public final File sourceFile;
        public final long sourceBytes;
        public final long createdAtMillis;
        public final long updatedAtMillis;

        private LibraryItem(
                String id,
                String displayName,
                String originalName,
                String status,
                File sourceFile,
                long sourceBytes,
                long createdAtMillis,
                long updatedAtMillis
        ) {
            this.id = id;
            this.displayName = displayName;
            this.originalName = originalName;
            this.status = status;
            this.sourceFile = sourceFile;
            this.sourceBytes = sourceBytes;
            this.createdAtMillis = createdAtMillis;
            this.updatedAtMillis = updatedAtMillis;
        }
    }

    private static final int SCHEMA_VERSION = 2;

    private final Context context;
    private final File root;

    public NativeFileLibrary(Context context) {
        this.context = context.getApplicationContext();
        File filesDir = this.context.getFilesDir();
        deleteRecursively(new File(filesDir, "library"));
        File libraryRoot = new File(filesDir, "library_v2");
        this.root = new File(libraryRoot, "items");
        if (!root.exists() && !root.mkdirs()) {
            throw new IllegalStateException("Cannot create native library directory.");
        }
    }

    public LibraryItem importSource(Uri uri) throws Exception {
        return importSource(uri, null);
    }

    public LibraryItem importSource(Uri uri, String requestedDisplayName) throws Exception {
        String id = UUID.randomUUID().toString();
        String originalName = displayName(uri);
        String name = cleanDisplayName(requestedDisplayName, baseName(originalName));
        File dir = new File(root, id);
        if (!dir.mkdirs()) throw new IllegalStateException("Cannot create item directory.");

        File source = new File(dir, "source" + extensionOf(originalName));
        try (InputStream input = context.getContentResolver().openInputStream(uri);
             FileOutputStream output = new FileOutputStream(source)) {
            if (input == null) throw new IllegalStateException("Cannot open input file.");
            byte[] buffer = new byte[1024 * 1024];
            int read;
            while ((read = input.read(buffer)) >= 0) {
                output.write(buffer, 0, read);
            }
        }

        long now = System.currentTimeMillis();
        LibraryItem item = new LibraryItem(
                id,
                name,
                originalName,
                STATUS_PROCESSING,
                source,
                source.length(),
                now,
                now);
        writeMetadata(item);
        return item;
    }

    public void markProcessing(String itemId) throws Exception {
        updateStatus(itemId, STATUS_PROCESSING);
    }

    public void markError(String itemId) throws Exception {
        updateStatus(itemId, STATUS_ERROR);
    }

    public void markCancelled(String itemId) throws Exception {
        updateStatus(itemId, STATUS_CANCELLED);
    }

    public void saveResult(String itemId, String json) throws Exception {
        File dir = itemDir(itemId);
        if (!dir.isDirectory()) throw new IllegalStateException("Library item not found.");
        Files.write(resultFile(itemId).toPath(), json.getBytes(StandardCharsets.UTF_8));
        updateStatus(itemId, STATUS_COMPLETE);
    }

    public String readResult(String itemId) throws Exception {
        File result = resultFile(itemId);
        if (!result.isFile()) return null;
        return new String(Files.readAllBytes(result.toPath()), StandardCharsets.UTF_8);
    }

    public boolean hasResult(String itemId) {
        return resultFile(itemId).isFile();
    }

    public void deleteItem(String itemId) throws Exception {
        File dir = itemDir(itemId);
        if (!dir.getCanonicalPath().startsWith(root.getCanonicalPath() + File.separator)) {
            throw new IllegalStateException("Invalid library item path.");
        }
        deleteRecursively(dir);
    }

    public LibraryItem getItem(String itemId) {
        return readItem(itemDir(itemId));
    }

    public LibraryItem latestProcessingItem() {
        LibraryItem latest = null;
        for (LibraryItem item : listItems()) {
            if (!STATUS_PROCESSING.equals(item.status) || hasResult(item.id)) continue;
            if (latest == null || item.updatedAtMillis > latest.updatedAtMillis) latest = item;
        }
        return latest;
    }

    public List<LibraryItem> listItems() {
        List<LibraryItem> items = new ArrayList<>();
        File[] dirs = root.listFiles(File::isDirectory);
        if (dirs == null) return items;
        for (File dir : dirs) {
            LibraryItem item = readItem(dir);
            if (item != null) items.add(item);
        }
        items.sort((left, right) -> Long.compare(right.updatedAtMillis, left.updatedAtMillis));
        return items;
    }

    private void updateStatus(String itemId, String status) throws Exception {
        LibraryItem current = getItem(itemId);
        if (current == null) throw new IllegalStateException("Library item not found.");
        LibraryItem updated = new LibraryItem(
                current.id,
                current.displayName,
                current.originalName,
                status,
                current.sourceFile,
                current.sourceFile.length(),
                current.createdAtMillis,
                System.currentTimeMillis());
        writeMetadata(updated);
    }

    private LibraryItem readItem(File dir) {
        if (dir == null || !dir.isDirectory()) return null;
        File metadata = new File(dir, "metadata.json");
        if (!metadata.isFile()) return null;
        try {
            JSONObject json = new JSONObject(new String(Files.readAllBytes(metadata.toPath()), StandardCharsets.UTF_8));
            String id = json.optString("id", dir.getName());
            String sourceFileName = json.optString("source_file", "source.bin");
            File source = new File(dir, sourceFileName);
            if (!source.isFile()) return null;
            String originalName = json.optString("original_name", source.getName());
            String displayName = json.optString("display_name", baseName(originalName));
            String status = json.optString("status", hasResult(id) ? STATUS_COMPLETE : STATUS_PROCESSING);
            long createdAt = json.optLong("created_at", source.lastModified());
            long updatedAt = json.optLong("updated_at", Math.max(source.lastModified(), resultFile(id).lastModified()));
            long sourceBytes = json.optLong("source_bytes", source.length());
            return new LibraryItem(id, displayName, originalName, status, source, sourceBytes, createdAt, updatedAt);
        } catch (Exception ignored) {
            return null;
        }
    }

    private void writeMetadata(LibraryItem item) throws Exception {
        JSONObject json = new JSONObject();
        json.put("schema_version", SCHEMA_VERSION);
        json.put("id", item.id);
        json.put("display_name", item.displayName);
        json.put("original_name", item.originalName);
        json.put("source_file", item.sourceFile.getName());
        json.put("source_bytes", item.sourceBytes);
        json.put("status", item.status);
        json.put("created_at", item.createdAtMillis);
        json.put("updated_at", item.updatedAtMillis);

        File dir = itemDir(item.id);
        if (!dir.exists() && !dir.mkdirs()) throw new IllegalStateException("Cannot create item directory.");
        File tmp = new File(dir, "metadata.json.tmp");
        File target = new File(dir, "metadata.json");
        Files.write(tmp.toPath(), json.toString(2).getBytes(StandardCharsets.UTF_8));
        try {
            Files.move(tmp.toPath(), target.toPath(), StandardCopyOption.REPLACE_EXISTING, StandardCopyOption.ATOMIC_MOVE);
        } catch (AtomicMoveNotSupportedException ignored) {
            Files.move(tmp.toPath(), target.toPath(), StandardCopyOption.REPLACE_EXISTING);
        }
    }

    private File itemDir(String itemId) {
        return new File(root, itemId);
    }

    private File resultFile(String itemId) {
        return new File(itemDir(itemId), "result.asr.json");
    }

    private String displayName(Uri uri) {
        try (Cursor cursor = context.getContentResolver().query(uri, null, null, null, null)) {
            if (cursor != null && cursor.moveToFirst()) {
                int index = cursor.getColumnIndex(OpenableColumns.DISPLAY_NAME);
                if (index >= 0) {
                    String value = cursor.getString(index);
                    if (value != null && !value.trim().isEmpty()) return value;
                }
            }
        } catch (Exception ignored) {
        }
        return uri.getLastPathSegment() == null ? "source.bin" : uri.getLastPathSegment();
    }

    private String cleanDisplayName(String requested, String fallback) {
        String value = requested == null ? "" : requested.replace('\r', ' ').replace('\n', ' ').trim();
        return value.isEmpty() ? fallback : value;
    }

    private String baseName(String name) {
        if (name == null || name.trim().isEmpty()) return "Tập tin";
        String value = name.trim();
        int slash = Math.max(value.lastIndexOf('/'), value.lastIndexOf('\\'));
        if (slash >= 0 && slash < value.length() - 1) value = value.substring(slash + 1);
        int dot = value.lastIndexOf('.');
        if (dot > 0) value = value.substring(0, dot);
        return value.isEmpty() ? "Tập tin" : value;
    }

    private String extensionOf(String name) {
        if (name == null) return ".bin";
        int slash = Math.max(name.lastIndexOf('/'), name.lastIndexOf('\\'));
        int dot = name.lastIndexOf('.');
        if (dot <= slash || dot < 0 || dot == name.length() - 1) return ".bin";
        String ext = name.substring(dot).replaceAll("[^A-Za-z0-9.]", "");
        return ext.isEmpty() ? ".bin" : ext.toLowerCase(Locale.US);
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
