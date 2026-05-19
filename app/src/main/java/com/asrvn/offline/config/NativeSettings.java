package com.asrvn.offline.config;

import android.content.Context;
import android.content.SharedPreferences;

import java.io.ByteArrayOutputStream;
import java.io.InputStream;
import java.nio.charset.StandardCharsets;

public final class NativeSettings {
    public static final String ASR_MODEL_68M = "sherpa-onnx-zipformer-vi-2025-04-20";
    public static final String SPEAKER_CAMPP = "senko_campp_optimized";
    public static final String SPEAKER_PYANNOTE = "pyannote_community1_vbx";
    public static final int DEFAULT_PUNCTUATION_LEVEL = 6;
    public static final int DEFAULT_CASE_LEVEL = 6;

    private final SharedPreferences prefs;
    private final Context context;

    public NativeSettings(Context context) {
        this.context = context.getApplicationContext();
        prefs = this.context.getSharedPreferences("asr_vn_native_settings", Context.MODE_PRIVATE);
        if (!prefs.getBoolean("ui_compact_v2_applied", false)) {
            prefs.edit()
                    .putInt("ui_text_scale", 100)
                    .putBoolean("resume_after_kill", true)
                    .putBoolean("ui_compact_v2_applied", true)
                    .apply();
        }
        if (!prefs.getBoolean("ui_compact_v3_applied", false)) {
            prefs.edit()
                    .putInt("ui_text_scale", 100)
                    .putBoolean("resume_after_kill", true)
                    .putBoolean("ui_compact_v3_applied", true)
                    .apply();
        }
        if (!prefs.getBoolean("ui_readable_v4_applied", false)) {
            prefs.edit()
                    .putInt("ui_text_scale", Math.max(120, prefs.getInt("ui_text_scale", 110)))
                    .putBoolean("resume_after_kill", true)
                    .putBoolean("ui_readable_v4_applied", true)
                    .apply();
        }
    }

    public String asrModel() {
        return prefs.getString("asr_model", ASR_MODEL_68M);
    }

    public String speakerModel() {
        return prefs.getString("speaker_model", SPEAKER_CAMPP);
    }

    public int cpuThreads() {
        return Math.max(1, Math.min(8, prefs.getInt("cpu_threads", 4)));
    }

    public int uiTextScalePercent() {
        return Math.max(100, Math.min(140, prefs.getInt("ui_text_scale", 120)));
    }

    public void setUiTextScalePercent(int value) {
        prefs.edit().putInt("ui_text_scale", Math.max(100, Math.min(140, value))).apply();
    }

    public boolean punctuationEnabled() {
        return prefs.getBoolean("punctuation", true);
    }

    public int punctuationLevel() {
        return sliderLevel("punctuation_level", DEFAULT_PUNCTUATION_LEVEL);
    }

    public int caseLevel() {
        return sliderLevel("case_level", DEFAULT_CASE_LEVEL);
    }

    public void setPunctuationLevel(int value) {
        prefs.edit().putInt("punctuation_level", clampSliderLevel(value)).apply();
    }

    public void setCaseLevel(int value) {
        prefs.edit().putInt("case_level", clampSliderLevel(value)).apply();
    }

    public boolean bypassPunctuation() {
        return punctuationLevel() <= 1;
    }

    public double punctuationConfidence() {
        return punctuationConfidenceFromLevel(punctuationLevel());
    }

    public double caseConfidence() {
        return caseConfidenceFromLevel(caseLevel());
    }

    public static double punctuationConfidenceFromLevel(int value) {
        int level = clampSliderLevel(value);
        return 0.5 - (level - 1) * (1.3 / 9.0);
    }

    public static double caseConfidenceFromLevel(int value) {
        int level = clampSliderLevel(value);
        return -1.5 + (level - 1) * (2.0 / 9.0);
    }

    public static int clampSliderLevel(int value) {
        return Math.max(1, Math.min(10, value));
    }

    private int sliderLevel(String key, int fallback) {
        return clampSliderLevel(prefs.getInt(key, fallback));
    }

    public boolean diarizationEnabled() {
        return prefs.getBoolean("diarization", true);
    }

    public boolean bypassVad() {
        return prefs.getBoolean("bypass_vad", false);
    }

    public boolean resumeAfterKill() {
        return true;
    }

    public boolean acceleratorEnabled() {
        return prefs.getBoolean("accelerator", true);
    }

    public void set(String key, boolean value) {
        prefs.edit().putBoolean(key, value).apply();
    }

    public void setSpeakerModel(String model) {
        prefs.edit().putString("speaker_model", model).apply();
    }

    public String hotwordsText() {
        String stored = prefs.getString("hotwords_text", null);
        return stored == null ? defaultHotwordsText() : stored;
    }

    public void setHotwordsText(String value) {
        prefs.edit().putString("hotwords_text", value == null ? "" : value).apply();
    }

    public void resetHotwords() {
        prefs.edit().remove("hotwords_text").apply();
    }

    public String defaultHotwordsText() {
        try (InputStream input = context.getAssets().open("hotword.txt")) {
            ByteArrayOutputStream bytes = new ByteArrayOutputStream();
            byte[] buffer = new byte[8192];
            int read;
            while ((read = input.read(buffer)) >= 0) {
                bytes.write(buffer, 0, read);
            }
            return new String(bytes.toByteArray(), StandardCharsets.UTF_8);
        } catch (Exception ignored) {
            return "";
        }
    }
}
