package com.asrvn.offline.diarization;

import java.io.ByteArrayOutputStream;
import java.io.InputStream;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.charset.StandardCharsets;
import java.util.HashMap;
import java.util.Map;
import java.util.zip.ZipEntry;
import java.util.zip.ZipInputStream;

final class NpyNpzReader {
    static final class ArrayData {
        final double[] data;
        final int[] shape;

        ArrayData(double[] data, int[] shape) {
            this.data = data;
            this.shape = shape;
        }
    }

    private NpyNpzReader() {
    }

    static ArrayData npy(byte[] bytes) {
        if (bytes.length < 16 || bytes[0] != (byte) 0x93 || bytes[1] != 'N' || bytes[2] != 'U' || bytes[3] != 'M') {
            throw new IllegalArgumentException("Invalid NPY file.");
        }
        int major = bytes[6] & 0xff;
        int headerLength;
        int offset;
        if (major == 1) {
            headerLength = (bytes[8] & 0xff) | ((bytes[9] & 0xff) << 8);
            offset = 10;
        } else {
            headerLength = (bytes[8] & 0xff) | ((bytes[9] & 0xff) << 8) | ((bytes[10] & 0xff) << 16) | ((bytes[11] & 0xff) << 24);
            offset = 12;
        }
        String header = new String(bytes, offset, headerLength, StandardCharsets.ISO_8859_1);
        boolean float64 = header.contains("'descr': '<f8") || header.contains("\"descr\": \"<f8");
        boolean float32 = header.contains("'descr': '<f4") || header.contains("\"descr\": \"<f4");
        if (!float32 && !float64) throw new IllegalArgumentException("Unsupported NPY dtype: " + header);
        if (header.contains("True")) throw new IllegalArgumentException("Fortran-order NPY is unsupported.");
        int open = header.indexOf('(');
        int close = header.indexOf(')', open + 1);
        if (open < 0 || close < 0) throw new IllegalArgumentException("NPY shape missing.");
        String[] parts = header.substring(open + 1, close).split(",");
        int dims = 0;
        for (String part : parts) if (!part.trim().isEmpty()) dims++;
        int[] shape = new int[dims];
        int p = 0;
        int total = 1;
        for (String part : parts) {
            String clean = part.trim();
            if (clean.isEmpty()) continue;
            int value = Integer.parseInt(clean);
            shape[p++] = value;
            total *= value;
        }
        ByteBuffer buffer = ByteBuffer.wrap(bytes, offset + headerLength, bytes.length - offset - headerLength).order(ByteOrder.LITTLE_ENDIAN);
        double[] data = new double[total];
        for (int i = 0; i < total; i++) data[i] = float64 ? buffer.getDouble() : buffer.getFloat();
        return new ArrayData(data, shape);
    }

    static Map<String, ArrayData> npz(byte[] bytes) throws Exception {
        Map<String, ArrayData> arrays = new HashMap<>();
        try (ZipInputStream zip = new ZipInputStream(new java.io.ByteArrayInputStream(bytes))) {
            ZipEntry entry;
            while ((entry = zip.getNextEntry()) != null) {
                if (entry.isDirectory() || !entry.getName().endsWith(".npy")) continue;
                String key = entry.getName();
                int slash = key.lastIndexOf('/');
                if (slash >= 0) key = key.substring(slash + 1);
                key = key.substring(0, key.length() - 4);
                arrays.put(key, npy(readAll(zip)));
            }
        }
        return arrays;
    }

    private static byte[] readAll(InputStream input) throws Exception {
        ByteArrayOutputStream output = new ByteArrayOutputStream();
        byte[] buffer = new byte[8192];
        int read;
        while ((read = input.read(buffer)) >= 0) output.write(buffer, 0, read);
        return output.toByteArray();
    }
}
