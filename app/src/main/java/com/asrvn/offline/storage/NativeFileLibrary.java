package com.asrvn.offline.storage;

import android.content.Context;
import android.database.Cursor;
import android.net.Uri;
import android.provider.OpenableColumns;

import java.io.File;
import java.io.FileOutputStream;
import java.io.InputStream;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.util.ArrayList;
import java.util.List;
import java.util.UUID;

public final class NativeFileLibrary {
    public static final class LibraryItem {
        public final String id;
        public final String displayName;
        public final File sourceFile;

        private LibraryItem(String id, String displayName, File sourceFile) {
            this.id = id;
            this.displayName = displayName;
            this.sourceFile = sourceFile;
        }
    }

    private final Context context;
    private final File root;

    public NativeFileLibrary(Context context) {
        this.context = context.getApplicationContext();
        this.root = new File(context.getFilesDir(), "library");
        if (!root.exists() && !root.mkdirs()) {
            throw new IllegalStateException("Cannot create native library directory.");
        }
    }

    public LibraryItem importSource(Uri uri) throws Exception {
        String id = UUID.randomUUID().toString();
        String name = displayName(uri);
        File dir = new File(root, id);
        if (!dir.mkdirs()) throw new IllegalStateException("Cannot create item directory.");
        File source = new File(dir, "source" + extensionOf(name));
        try (InputStream input = context.getContentResolver().openInputStream(uri);
             FileOutputStream output = new FileOutputStream(source)) {
            if (input == null) throw new IllegalStateException("Cannot open input file.");
            byte[] buffer = new byte[1024 * 1024];
            int read;
            while ((read = input.read(buffer)) >= 0) {
                output.write(buffer, 0, read);
            }
        }
        Files.write(new File(dir, "name.txt").toPath(), name.getBytes(StandardCharsets.UTF_8));
        return new LibraryItem(id, name, source);
    }

    public void saveResult(String itemId, String json) throws Exception {
        File dir = new File(root, itemId);
        if (!dir.isDirectory()) throw new IllegalStateException("Library item not found.");
        Files.write(new File(dir, "result.asr.json").toPath(), json.getBytes(StandardCharsets.UTF_8));
    }

    public String readResult(String itemId) throws Exception {
        File result = resultFile(itemId);
        if (!result.isFile()) return null;
        return new String(Files.readAllBytes(result.toPath()), StandardCharsets.UTF_8);
    }

    public boolean hasResult(String itemId) {
        return resultFile(itemId).isFile();
    }

    private File resultFile(String itemId) {
        return new File(new File(root, itemId), "result.asr.json");
    }

    public List<LibraryItem> listItems() {
        List<LibraryItem> items = new ArrayList<>();
        File[] dirs = root.listFiles(File::isDirectory);
        if (dirs == null) return items;
        for (File dir : dirs) {
            File source = findSource(dir);
            if (source == null) continue;
            File nameFile = new File(dir, "name.txt");
            String name = dir.getName();
            try {
                if (nameFile.isFile()) name = new String(Files.readAllBytes(nameFile.toPath()), StandardCharsets.UTF_8);
            } catch (Exception ignored) {
            }
            items.add(new LibraryItem(dir.getName(), name, source));
        }
        return items;
    }

    private File findSource(File dir) {
        File[] files = dir.listFiles(file -> file.isFile() && file.getName().startsWith("source"));
        return files == null || files.length == 0 ? null : files[0];
    }

    private String displayName(Uri uri) {
        try (Cursor cursor = context.getContentResolver().query(uri, null, null, null, null)) {
            if (cursor != null && cursor.moveToFirst()) {
                int index = cursor.getColumnIndex(OpenableColumns.DISPLAY_NAME);
                if (index >= 0) return cursor.getString(index);
            }
        }
        return uri.getLastPathSegment() == null ? "source.bin" : uri.getLastPathSegment();
    }

    private String extensionOf(String name) {
        if (name == null) return ".bin";
        int slash = Math.max(name.lastIndexOf('/'), name.lastIndexOf('\\'));
        int dot = name.lastIndexOf('.');
        if (dot <= slash || dot < 0 || dot == name.length() - 1) return ".bin";
        String ext = name.substring(dot).replaceAll("[^A-Za-z0-9.]", "");
        return ext.isEmpty() ? ".bin" : ext.toLowerCase(java.util.Locale.US);
    }
}
