package com.asrvn.offline;

import android.app.Activity;
import android.app.AlertDialog;
import android.content.ClipData;
import android.content.ClipboardManager;
import android.content.Intent;
import android.content.res.ColorStateList;
import android.database.Cursor;
import android.graphics.Color;
import android.graphics.Typeface;
import android.graphics.drawable.ColorDrawable;
import android.graphics.drawable.GradientDrawable;
import android.media.MediaPlayer;
import android.net.Uri;
import android.os.Build;
import android.os.Bundle;
import android.os.Handler;
import android.os.Looper;
import android.provider.OpenableColumns;
import android.text.SpannableStringBuilder;
import android.text.Spanned;
import android.text.Editable;
import android.text.InputType;
import android.text.TextPaint;
import android.text.TextWatcher;
import android.text.method.LinkMovementMethod;
import android.text.style.ClickableSpan;
import android.util.TypedValue;
import android.util.Log;
import android.view.Gravity;
import android.view.View;
import android.view.ViewGroup;
import android.view.Window;
import android.view.WindowManager;
import android.view.animation.DecelerateInterpolator;
import android.widget.ArrayAdapter;
import android.widget.Button;
import android.widget.CheckBox;
import android.widget.EditText;
import android.widget.FrameLayout;
import android.widget.LinearLayout;
import android.widget.ProgressBar;
import android.widget.RadioButton;
import android.widget.RadioGroup;
import android.widget.ScrollView;
import android.widget.SeekBar;
import android.widget.Spinner;
import android.widget.TextView;

import org.json.JSONArray;
import org.json.JSONObject;

import com.asrvn.offline.config.NativeSettings;
import com.asrvn.offline.asr.PureOrtRecognizer;
import com.asrvn.offline.asr.TokenParser;
import com.asrvn.offline.diarization.DiarizationResult;
import com.asrvn.offline.models.ModelFileRegistry;
import com.asrvn.offline.pipeline.NativeOfflinePipeline;
import com.asrvn.offline.storage.NativeFileLibrary;

import java.io.File;
import java.io.ByteArrayOutputStream;
import java.io.InputStream;
import java.io.OutputStream;
import java.text.SimpleDateFormat;
import java.nio.charset.StandardCharsets;
import java.util.Date;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;

public final class MainActivity extends Activity {
    private static final String TAG = "ASRVN";
    private static final int REQUEST_OPEN_MEDIA = 1001;
    private static final int REQUEST_IMPORT_HOTWORD_TXT = 1002;
    private static final int REQUEST_EXPORT_HOTWORD_TXT = 1003;

    private static final int BG = Color.rgb(43, 43, 43);
    private static final int CARD = Color.rgb(58, 58, 58);
    private static final int ELEVATED = Color.rgb(70, 70, 70);
    private static final int INPUT = Color.rgb(37, 37, 37);
    private static final int BORDER = Color.rgb(85, 85, 85);
    private static final int TEXT = Color.WHITE;
    private static final int MUTED = Color.rgb(204, 204, 204);
    private static final int ACCENT = Color.rgb(0, 123, 255);
    private static final int PRIMARY = Color.rgb(37, 99, 180);
    private static final int SUCCESS = Color.rgb(40, 167, 69);
    private static final int DANGER = Color.rgb(220, 53, 69);
    private static final int[] SPEAKER_COLORS = new int[]{
            Color.rgb(0, 123, 255),
            Color.rgb(40, 167, 69),
            Color.rgb(255, 193, 7),
            Color.rgb(220, 53, 69),
            Color.rgb(111, 66, 193),
            Color.rgb(23, 162, 184),
            Color.rgb(253, 126, 20),
            Color.rgb(232, 62, 140)
    };
    private static final String[] SPEAKER_COLOR_LABELS = new String[]{
            "Xanh dương", "Xanh lá", "Vàng", "Đỏ", "Tím", "Cyan", "Cam", "Hồng"
    };
    private static final double WORD_ASSIGN_MAX_DURATION_SECONDS = 0.40;
    private static final double DEFAULT_HOTWORD_SCORE = 1.5;
    private static final int MAX_HOTWORDS = 1000;

    private static final class EditorSegment {
        final double start;
        final double end;
        String text;
        int speaker;
        final List<TimedWord> words;

        EditorSegment(double start, double end, String text, int speaker) {
            this(start, end, text, speaker, null);
        }

        EditorSegment(double start, double end, String text, int speaker, List<TimedWord> words) {
            this.start = start;
            this.end = end;
            this.text = text == null ? "" : text;
            this.speaker = speaker;
            this.words = words == null ? new ArrayList<>() : new ArrayList<>(words);
        }
    }

    private static final class SpeakerMeta {
        String name;
        int color;

        SpeakerMeta(String name, int color) {
            this.name = name;
            this.color = color;
        }
    }

    private static final class TimedWord {
        final String text;
        final double start;
        final double end;

        TimedWord(String text, double start, double end) {
            this.text = text == null ? "" : text;
            this.start = start;
            this.end = end;
        }
    }

    private static final class HotwordItem {
        String text;
        double score;

        HotwordItem(String text, double score) {
            this.text = text == null ? "" : text;
            this.score = score;
        }
    }

    private FrameLayout rootFrame;
    private View drawer;
    private View scrim;
    private LinearLayout libraryList;
    private LinearLayout configBody;
    private TextView configHeaderText;
    private View dropZoneView;
    private ScrollView mainScrollView;
    private View dropPromptView;
    private LinearLayout selectedFileView;
    private TextView fileNameView;
    private TextView fileSizeView;
    private TextView logView;
    private LinearLayout activeHotwordList;
    private TextView activeHotwordSummary;
    private EditText activeHotwordSearch;
    private List<HotwordItem> activeHotwordItems;
    private String pendingHotwordExportText;
    private LinearLayout transcriptRoot;
    private TextView transcriptEmptyView;
    private TextView playerTimeView;
    private Button playerPlayPauseButton;
    private SeekBar playerSeekBar;
    private TextView stageView;
    private TextView progressText;
    private ProgressBar progressBar;
    private View progressContainer;
    private View resultPanelView;
    private View fixedPlayerView;
    private Spinner speakerModelSpinner;
    private MediaPlayer mediaPlayer;
    private boolean userSeeking;
    private boolean configExpanded = true;
    private int activeSegmentIndex = -1;
    private long lastActiveAutoScrollAtMs;
    private Uri selectedUri;
    private String selectedLibraryItemId;
    private String selectedDisplayName;
    private long selectedFileSizeBytes;
    private String pendingDebugSpeakerModel;
    private String currentTranscriptText = "";
    private double currentDurationSeconds;
    private final List<EditorSegment> editorSegments = new ArrayList<>();
    private final Map<Integer, SpeakerMeta> editorSpeakers = new LinkedHashMap<>();
    private final Handler uiHandler = new Handler(Looper.getMainLooper());

    private NativeOfflinePipeline pipeline;
    private NativeSettings settings;
    private ModelFileRegistry models;
    private NativeFileLibrary library;

    private interface LevelSetter {
        void setLevel(int value);
    }

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        settings = new NativeSettings(this);
        models = new ModelFileRegistry(this);
        library = new NativeFileLibrary(this);
        pipeline = new NativeOfflinePipeline(this);
        enableStandaloneChrome();
        if (Build.VERSION.SDK_INT >= 21) {
            Window window = getWindow();
            window.setStatusBarColor(BG);
            window.setNavigationBarColor(BG);
        }
        setContentView(buildContentView());
        rootFrame.postDelayed(this::showInitialModelDownloadPromptIfNeeded, 700);
        handleDebugIntent(getIntent());
    }

    @Override
    protected void onNewIntent(Intent intent) {
        super.onNewIntent(intent);
        setIntent(intent);
        handleDebugIntent(intent);
    }

    @Override
    protected void onDestroy() {
        uiHandler.removeCallbacksAndMessages(null);
        releaseMediaPlayer();
        super.onDestroy();
    }

    private View buildContentView() {
        rootFrame = new FrameLayout(this);
        rootFrame.setBackgroundColor(BG);

        LinearLayout app = vertical();
        app.setBackgroundColor(BG);
        app.setPadding(0, statusBarHeight(), 0, 0);
        app.addView(topBar());
        app.addView(mainScroll(), new LinearLayout.LayoutParams(
                LinearLayout.LayoutParams.MATCH_PARENT, 0, 1f));
        rootFrame.addView(app);

        fixedPlayerView = fixedPlayerBar();
        fixedPlayerView.setVisibility(View.GONE);
        rootFrame.addView(fixedPlayerView, new FrameLayout.LayoutParams(
                FrameLayout.LayoutParams.MATCH_PARENT,
                dp(62),
                Gravity.BOTTOM));

        scrim = new View(this);
        scrim.setBackgroundColor(Color.argb(70, 0, 0, 0));
        scrim.setVisibility(View.GONE);
        scrim.setOnClickListener(v -> closeDrawer());
        rootFrame.addView(scrim, new FrameLayout.LayoutParams(
                FrameLayout.LayoutParams.MATCH_PARENT,
                FrameLayout.LayoutParams.MATCH_PARENT));

        drawer = libraryDrawer();
        int screenWidth = getResources().getDisplayMetrics().widthPixels;
        int drawerWidth = screenWidth < dp(900)
                ? (int) (screenWidth * 0.95f)
                : Math.min(dp(520), (int) (screenWidth * 0.95f));
        FrameLayout.LayoutParams drawerParams = new FrameLayout.LayoutParams(
                drawerWidth,
                FrameLayout.LayoutParams.MATCH_PARENT,
                Gravity.RIGHT);
        drawer.setVisibility(View.GONE);
        rootFrame.addView(drawer, drawerParams);
        refreshLibraryList();
        return rootFrame;
    }

    private View topBar() {
        LinearLayout bar = new LinearLayout(this);
        bar.setOrientation(LinearLayout.HORIZONTAL);
        bar.setGravity(Gravity.CENTER_VERTICAL);
        bar.setPadding(dp(16), 0, dp(16), 0);
        bar.setBackgroundColor(CARD);

        TextView title = text("Sherpa Vietnamese\nASR", 20, true, ACCENT);
        title.setLineSpacing(0, 1.0f);
        bar.addView(title, new LinearLayout.LayoutParams(0, dp(78), 1f));

        TextView info = text("i", 18, true, Color.rgb(174, 174, 174));
        info.setGravity(Gravity.CENTER);
        bar.addView(info, new LinearLayout.LayoutParams(dp(42), dp(78)));

        Button files = button("Quản lý tập\ntin", ELEVATED, v -> openDrawer());
        files.setTextSize(TypedValue.COMPLEX_UNIT_PX, 18 * uiScale());
        files.setGravity(Gravity.CENTER);
        LinearLayout.LayoutParams params = new LinearLayout.LayoutParams(dp(156), dp(68));
        params.leftMargin = dp(8);
        bar.addView(files, params);
        return bar;
    }

    private View mainScroll() {
        ScrollView scroll = new ScrollView(this);
        mainScrollView = scroll;
        scroll.setFillViewport(false);
        scroll.setClipToPadding(false);
        scroll.setPadding(0, 0, 0, dp(72));
        scroll.setDescendantFocusability(ViewGroup.FOCUS_BEFORE_DESCENDANTS);
        LinearLayout main = vertical();
        main.setPadding(dp(14), dp(12), dp(14), dp(14));
        main.addView(configPanel());
        main.addView(filePanel());
        main.addView(resultPanel());
        scroll.addView(main);
        return scroll;
    }

    private View configPanel() {
        LinearLayout panel = panel();
        configHeaderText = text("▼  Cấu hình", 24, true, TEXT);
        View header = panelHeader(configHeaderText);
        header.setOnClickListener(v -> toggleConfigPanel());
        panel.addView(header);

        LinearLayout body = vertical();
        configBody = body;
        body.setPadding(dp(18), dp(8), dp(18), dp(10));
        body.addView(formRow("Model ASR:", spinner(new String[]{
                "Zipformer-Vi 2025 (68M)",
                "Zipformer 30M (nhanh)",
                "ROVER (chậm, chính xác)"
        })));
        body.addView(textScaleRow());
        body.addView(levelSliderRow("Thêm dấu:", settings.punctuationLevel(), settings::setPunctuationLevel));
        body.addView(levelSliderRow("Viết hoa:", settings.caseLevel(), settings::setCaseLevel));
        body.addView(fullCheck("Audio cực kỳ khó nghe (có thể làm giảm độ chính xác nếu Audio tốt)", settings.bypassVad(), "bypass_vad"));
        body.addView(fullCheck("Phân tách người nói (nếu biết và chỉ định trước số người nói, kết quả sẽ chính xác hơn)", settings.diarizationEnabled(), "diarization"));
        body.addView(formRow("Số người nói:", spinner(new String[]{
                "Không rõ (tự động)", "2", "3", "4", "5", "6", "7", "8", "9", "10"
        })));
        body.addView(formRow("Model người nói:", speakerSpinner()));
        body.addView(hotwordRow());
        panel.addView(body);
        return withMargins(panel, 0, 0, 0, 12);
    }

    private void toggleConfigPanel() {
        configExpanded = !configExpanded;
        if (configBody != null) configBody.setVisibility(configExpanded ? View.VISIBLE : View.GONE);
        if (configHeaderText != null) {
            configHeaderText.setText((configExpanded ? "▼  " : "▶  ") + "Cấu hình");
        }
    }

    private View filePanel() {
        LinearLayout panel = panel();
        LinearLayout body = vertical();
        body.setPadding(0, 0, 0, dp(6));

        FrameLayout drop = new FrameLayout(this);
        dropZoneView = drop;
        drop.setPadding(dp(22), dp(20), dp(22), dp(20));
        GradientDrawable bg = rounded(CARD, BORDER, 2, 8);
        bg.setStroke(dp(2), BORDER, dp(8), dp(4));
        drop.setBackground(bg);
        drop.setOnClickListener(v -> openMediaPicker());

        LinearLayout row = new LinearLayout(this);
        row.setOrientation(LinearLayout.HORIZONTAL);
        row.setGravity(Gravity.CENTER_VERTICAL);
        row.setPadding(dp(28), 0, dp(28), 0);
        dropPromptView = row;

        LinearLayout left = vertical();
        int promptTextMaxWidth = Math.max(dp(320), getResources().getDisplayMetrics().widthPixels - dp(260));
        TextView headline = text("Bấm để chọn file âm thanh\n(tối đa 500MB)", 22, false, MUTED);
        headline.setMaxWidth(promptTextMaxWidth);
        headline.setLineSpacing(dp(1), 1.0f);
        left.addView(headline);
        addPromptHint(left, "mp3, wav, m4a, flac, aac, wma, ogg, opus", promptTextMaxWidth);
        addPromptHint(left, "mp4, mkv, avi, mov, webm, flv, wmv", promptTextMaxWidth);
        addPromptHint(left, "Kết quả chính xác nhất với tập tin ghi âm", promptTextMaxWidth);
        addPromptHint(left, "trực tiếp từ hệ thống microphone cổ ngỗng.", promptTextMaxWidth);
        addPromptHint(left, "Nếu ghi âm trực tiếp tại phòng họp thì ưu", promptTextMaxWidth);
        addPromptHint(left, "tiên sử dụng thiết bị chuyên ghi âm như", promptTextMaxWidth);
        addPromptHint(left, "Sony và để sát người chủ trì.", promptTextMaxWidth);
        row.addView(left, new LinearLayout.LayoutParams(
                0,
                LinearLayout.LayoutParams.WRAP_CONTENT,
                1f));

        TextView icon = text("📁", 58, false, Color.rgb(255, 203, 73));
        icon.setGravity(Gravity.CENTER);
        LinearLayout.LayoutParams iconParams = new LinearLayout.LayoutParams(dp(82), dp(82));
        iconParams.leftMargin = dp(18);
        row.addView(icon, iconParams);
        drop.addView(row, new FrameLayout.LayoutParams(
                FrameLayout.LayoutParams.MATCH_PARENT,
                FrameLayout.LayoutParams.WRAP_CONTENT,
                Gravity.CENTER_VERTICAL));

        selectedFileView = new LinearLayout(this);
        selectedFileView.setOrientation(LinearLayout.HORIZONTAL);
        selectedFileView.setGravity(Gravity.CENTER);
        selectedFileView.setVisibility(View.GONE);
        TextView fileIcon = text("📄", 24, false, MUTED);
        selectedFileView.addView(fileIcon, new LinearLayout.LayoutParams(dp(34), dp(42)));
        fileNameView = text("", 17, true, TEXT);
        selectedFileView.addView(fileNameView, new LinearLayout.LayoutParams(0, dp(42), 1f));
        fileSizeView = text("", 13, false, MUTED);
        fileSizeView.setGravity(Gravity.CENTER_VERTICAL | Gravity.RIGHT);
        selectedFileView.addView(fileSizeView, new LinearLayout.LayoutParams(dp(90), dp(42)));
        TextView clear = text("×", 26, true, DANGER);
        clear.setGravity(Gravity.CENTER);
        clear.setOnClickListener(v -> {
            clearSelectedFile();
            v.getParent().requestDisallowInterceptTouchEvent(true);
        });
        selectedFileView.addView(clear, new LinearLayout.LayoutParams(dp(38), dp(42)));
        drop.addView(selectedFileView, new FrameLayout.LayoutParams(
                FrameLayout.LayoutParams.MATCH_PARENT,
                FrameLayout.LayoutParams.WRAP_CONTENT,
                Gravity.CENTER));

        LinearLayout.LayoutParams dropParams = new LinearLayout.LayoutParams(
                LinearLayout.LayoutParams.MATCH_PARENT, dp(360));
        dropParams.setMargins(dp(18), dp(14), dp(18), dp(10));
        body.addView(drop, dropParams);

        body.addView(actionRows());
        body.addView(progressBlock());
        panel.addView(body);
        return withMargins(panel, 0, 0, 0, 12);
    }

    private View actionRows() {
        LinearLayout rows = vertical();
        rows.setPadding(dp(18), 0, dp(18), 0);
        LinearLayout row1 = actionRow();
        row1.addView(button("Xử lý", PRIMARY, v -> processSelectedFile()), actionButtonParams());
        rows.addView(row1);

        LinearLayout row2 = actionRow();
        row2.addView(button("⇩ Tải JSON", INPUT, v -> appendLog("Chưa có kết quả JSON để tải.")), actionButtonParams());
        row2.addView(button("Sao chép văn bản", INPUT, v -> copyTranscript()), actionButtonParams());
        rows.addView(row2);

        return rows;
    }

    private View progressBlock() {
        LinearLayout block = vertical();
        progressContainer = block;
        block.setVisibility(View.GONE);
        block.setPadding(dp(18), dp(6), dp(18), 0);
        LinearLayout head = new LinearLayout(this);
        head.setOrientation(LinearLayout.HORIZONTAL);
        stageView = text("Đang chờ...", 13, true, MUTED);
        progressText = text("0%", 13, true, MUTED);
        progressText.setGravity(Gravity.RIGHT | Gravity.CENTER_VERTICAL);
        head.addView(stageView, new LinearLayout.LayoutParams(0, dp(20), 1f));
        head.addView(progressText, new LinearLayout.LayoutParams(dp(58), dp(20)));
        block.addView(head);

        progressBar = new ProgressBar(this, null, android.R.attr.progressBarStyleHorizontal);
        progressBar.setMax(100);
        if (Build.VERSION.SDK_INT >= 21) {
            progressBar.setProgressTintList(ColorStateList.valueOf(ACCENT));
            progressBar.setProgressBackgroundTintList(ColorStateList.valueOf(ELEVATED));
        }
        block.addView(progressBar, new LinearLayout.LayoutParams(
                LinearLayout.LayoutParams.MATCH_PARENT, dp(10)));
        return block;
    }

    private View resultPanel() {
        LinearLayout panel = panel();
        resultPanelView = panel;
        panel.setVisibility(View.GONE);

        LinearLayout player = new LinearLayout(this);
        player.setOrientation(LinearLayout.HORIZONTAL);
        player.setGravity(Gravity.CENTER_VERTICAL);
        player.setPadding(dp(18), 0, dp(18), dp(6));
        playerTimeView = text("00:00 / 00:00", 14, true, MUTED);
        player.addView(playerTimeView, new LinearLayout.LayoutParams(dp(132), dp(36)));
        playerSeekBar = new SeekBar(this);
        playerSeekBar.setMax(1000);
        if (Build.VERSION.SDK_INT >= 21) {
            playerSeekBar.setProgressTintList(ColorStateList.valueOf(ACCENT));
            playerSeekBar.setThumbTintList(ColorStateList.valueOf(ACCENT));
            playerSeekBar.setProgressBackgroundTintList(ColorStateList.valueOf(Color.rgb(235, 235, 235)));
        }
        playerSeekBar.setOnSeekBarChangeListener(new SeekBar.OnSeekBarChangeListener() {
            @Override
            public void onProgressChanged(SeekBar seekBar, int progress, boolean fromUser) {
                if (!fromUser || mediaPlayer == null) return;
                int duration = Math.max(1, mediaPlayer.getDuration());
                mediaPlayer.seekTo((int) ((progress / 1000.0) * duration));
                updatePlayerTime();
            }

            @Override
            public void onStartTrackingTouch(SeekBar seekBar) {
                userSeeking = true;
            }

            @Override
            public void onStopTrackingTouch(SeekBar seekBar) {
                userSeeking = false;
            }
        });
        player.addView(playerSeekBar, new LinearLayout.LayoutParams(0, dp(36), 1f));
        // The actual player is fixed at the bottom of the screen so transcript
        // scrolling does not hide the current audio position.

        transcriptRoot = vertical();
        transcriptRoot.setPadding(dp(18), 0, dp(18), dp(8));
        transcriptEmptyView = text("Chưa có nội dung. Chọn file và bấm Xử lý để hiển thị transcript tại đây.", 16, false, MUTED);
        transcriptEmptyView.setPadding(dp(10), dp(8), dp(10), dp(8));
        transcriptRoot.addView(transcriptEmptyView);
        panel.addView(transcriptRoot);

        logView = text("", 13, false, Color.rgb(225, 230, 235));
        logView.setTypeface(Typeface.MONOSPACE);
        logView.setTextIsSelectable(true);
        ScrollView scroll = new ScrollView(this);
        scroll.setPadding(dp(10), dp(6), dp(10), dp(8));
        scroll.addView(logView);
        scroll.setVisibility(View.GONE);
        panel.addView(scroll, new LinearLayout.LayoutParams(
                LinearLayout.LayoutParams.MATCH_PARENT, dp(1)));
        return panel;
    }

    private View fixedPlayerBar() {
        LinearLayout player = new LinearLayout(this);
        player.setOrientation(LinearLayout.HORIZONTAL);
        player.setGravity(Gravity.CENTER_VERTICAL);
        player.setPadding(dp(14), dp(6), dp(14), dp(6));
        GradientDrawable bg = rounded(Color.rgb(35, 35, 35), BORDER, 1, 0);
        player.setBackground(bg);

        playerPlayPauseButton = button("▶", INPUT, v -> togglePlayback());
        playerPlayPauseButton.setTextSize(TypedValue.COMPLEX_UNIT_PX, 20 * uiScale());
        player.addView(playerPlayPauseButton, new LinearLayout.LayoutParams(dp(52), dp(48)));

        playerTimeView = text("00:00 / 00:00", 14, true, MUTED);
        player.addView(playerTimeView, new LinearLayout.LayoutParams(dp(126), dp(48)));

        playerSeekBar = new SeekBar(this);
        playerSeekBar.setMax(1000);
        if (Build.VERSION.SDK_INT >= 21) {
            playerSeekBar.setProgressTintList(ColorStateList.valueOf(ACCENT));
            playerSeekBar.setThumbTintList(ColorStateList.valueOf(ACCENT));
            playerSeekBar.setProgressBackgroundTintList(ColorStateList.valueOf(Color.rgb(120, 120, 120)));
        }
        playerSeekBar.setOnSeekBarChangeListener(new SeekBar.OnSeekBarChangeListener() {
            @Override
            public void onProgressChanged(SeekBar seekBar, int progress, boolean fromUser) {
                if (!fromUser || mediaPlayer == null) return;
                int duration = Math.max(1, mediaPlayer.getDuration());
                int targetMs = (int) ((progress / 1000.0) * duration);
                if (Build.VERSION.SDK_INT >= 26) {
                    mediaPlayer.seekTo(targetMs, MediaPlayer.SEEK_CLOSEST);
                } else {
                    mediaPlayer.seekTo(targetMs);
                }
                updateActiveSegment(targetMs / 1000.0, true);
                updatePlayerTime();
            }

            @Override
            public void onStartTrackingTouch(SeekBar seekBar) {
                userSeeking = true;
            }

            @Override
            public void onStopTrackingTouch(SeekBar seekBar) {
                userSeeking = false;
            }
        });
        player.addView(playerSeekBar, new LinearLayout.LayoutParams(0, dp(48), 1f));
        return player;
    }

    private View libraryDrawer() {
        LinearLayout panel = vertical();
        panel.setBackgroundColor(CARD);

        LinearLayout header = new LinearLayout(this);
        header.setOrientation(LinearLayout.HORIZONTAL);
        header.setGravity(Gravity.CENTER_VERTICAL);
        header.setPadding(dp(16), dp(10) + statusBarHeight(), dp(12), dp(10));
        header.setBackgroundColor(CARD);
        header.addView(text("Quản lý tập tin", 24, true, TEXT), new LinearLayout.LayoutParams(0, dp(58), 1f));

        EditText search = new EditText(this);
        search.setHint("Tìm kiếm...");
        search.setSingleLine(true);
        search.setTextSize(TypedValue.COMPLEX_UNIT_PX, 16 * uiScale());
        search.setTextColor(TEXT);
        search.setHintTextColor(Color.rgb(135, 135, 135));
        search.setInputType(InputType.TYPE_CLASS_TEXT);
        search.setPadding(dp(14), 0, dp(14), 0);
        search.setBackground(rounded(INPUT, BORDER, 1, 4));
        header.addView(search, new LinearLayout.LayoutParams(dp(260), dp(48)));

        TextView close = text("×", 34, true, MUTED);
        close.setGravity(Gravity.CENTER);
        close.setOnClickListener(v -> closeDrawer());
        header.addView(close, new LinearLayout.LayoutParams(dp(52), dp(58)));
        panel.addView(header);

        View line1 = new View(this);
        line1.setBackgroundColor(BORDER);
        panel.addView(line1, new LinearLayout.LayoutParams(LinearLayout.LayoutParams.MATCH_PARENT, dp(1)));

        LinearLayout toolbar = new LinearLayout(this);
        toolbar.setOrientation(LinearLayout.HORIZONTAL);
        toolbar.setGravity(Gravity.CENTER_VERTICAL);
        toolbar.setPadding(dp(16), dp(10), dp(16), dp(10));
        CheckBox all = checkPlain("Chọn tất cả", false);
        all.setTextSize(TypedValue.COMPLEX_UNIT_PX, 18 * uiScale());
        toolbar.addView(all, new LinearLayout.LayoutParams(0, dp(48), 1f));
        toolbar.addView(button("Xóa đã chọn", DANGER, v -> appendLog("Chọn file cần xóa trong bước tiếp theo.")),
                new LinearLayout.LayoutParams(dp(166), dp(44)));
        panel.addView(toolbar);

        View line2 = new View(this);
        line2.setBackgroundColor(BORDER);
        panel.addView(line2, new LinearLayout.LayoutParams(LinearLayout.LayoutParams.MATCH_PARENT, dp(1)));

        ScrollView scroll = new ScrollView(this);
        scroll.setPadding(dp(12), dp(14), dp(12), dp(12));
        libraryList = vertical();
        scroll.addView(libraryList);
        panel.addView(scroll, new LinearLayout.LayoutParams(
                LinearLayout.LayoutParams.MATCH_PARENT, 0, 1f));
        return panel;
    }

    private void openDrawer() {
        refreshLibraryList();
        if (fixedPlayerView != null) fixedPlayerView.setVisibility(View.GONE);
        drawer.setVisibility(View.VISIBLE);
        scrim.setVisibility(View.VISIBLE);
        drawer.setTranslationX(drawer.getWidth() == 0 ? getResources().getDisplayMetrics().widthPixels : drawer.getWidth());
        drawer.animate().translationX(0).setDuration(190).setInterpolator(new DecelerateInterpolator()).start();
    }

    private void closeDrawer() {
        if (drawer == null || drawer.getVisibility() != View.VISIBLE) return;
        drawer.animate().translationX(drawer.getWidth()).setDuration(160).withEndAction(() -> {
            drawer.setVisibility(View.GONE);
            scrim.setVisibility(View.GONE);
            if (fixedPlayerView != null && mediaPlayer != null) fixedPlayerView.setVisibility(View.VISIBLE);
        }).start();
    }

    private void refreshLibraryList() {
        if (libraryList == null) return;
        libraryList.removeAllViews();
        List<NativeFileLibrary.LibraryItem> items = library.listItems();
        if (items.isEmpty()) {
            TextView empty = text("Chưa có tập tin.", 16, false, MUTED);
            empty.setPadding(dp(12), dp(16), dp(12), dp(16));
            libraryList.addView(empty);
            return;
        }
        SimpleDateFormat fmt = new SimpleDateFormat("d/M/yyyy HH:mm", Locale.getDefault());
        for (NativeFileLibrary.LibraryItem item : items) {
            LinearLayout row = vertical();
            row.setPadding(dp(14), dp(14), dp(14), dp(14));
            row.setBackground(rounded(CARD, BORDER, 1, 8));
            row.setOnClickListener(v -> openLibraryItem(item));

            LinearLayout top = new LinearLayout(this);
            top.setOrientation(LinearLayout.HORIZONTAL);
            top.setGravity(Gravity.CENTER_VERTICAL);
            top.addView(checkPlain("", false), new LinearLayout.LayoutParams(dp(40), dp(36)));

            TextView name = text(item.displayName, 20, true, TEXT);
            top.addView(name, new LinearLayout.LayoutParams(0, dp(44), 1f));
            top.addView(badge(library.hasResult(item.id) ? "Hoàn thành" : "Sẵn sàng"));
            row.addView(top);

            String meta = formatBytes(item.sourceFile.length()) + "    " + fmt.format(new Date(item.sourceFile.lastModified()));
            TextView details = text(meta, 16, false, MUTED);
            details.setPadding(dp(52), 0, 0, 0);
            row.addView(details);
            libraryList.addView(withMargins(row, 0, 0, 0, 10));
        }
    }

    private View panelHeader(String marker, String title) {
        return panelHeader(text(marker + "  " + title, 24, true, TEXT));
    }

    private View panelHeader(TextView text) {
        LinearLayout head = new LinearLayout(this);
        head.setOrientation(LinearLayout.HORIZONTAL);
        head.setGravity(Gravity.CENTER_VERTICAL);
        head.setMinimumHeight(dp(58));
        head.setPadding(dp(18), dp(8), dp(18), dp(8));
        text.setIncludeFontPadding(true);
        head.addView(text, new LinearLayout.LayoutParams(
                LinearLayout.LayoutParams.MATCH_PARENT,
                LinearLayout.LayoutParams.WRAP_CONTENT));
        return head;
    }

    private View formRow(String label, View control) {
        LinearLayout row = new LinearLayout(this);
        row.setOrientation(LinearLayout.HORIZONTAL);
        row.setGravity(Gravity.CENTER_VERTICAL);
        row.setPadding(0, dp(3), 0, dp(3));
        TextView labelView = text(label, 17, false, MUTED);
        labelView.setSingleLine(true);
        row.addView(labelView, new LinearLayout.LayoutParams(controlLabelWidth(), dp(42)));
        row.addView(control, new LinearLayout.LayoutParams(0, dp(44), 1f));
        return row;
    }

    private View hotwordRow() {
        return formRow("Hotword:", button("Quản lý hotword.txt", INPUT, v -> openHotwordDialog()));
    }

    private View levelSliderRow(String label, int initialValue, LevelSetter setter) {
        int initial = NativeSettings.clampSliderLevel(initialValue);
        LinearLayout row = new LinearLayout(this);
        row.setOrientation(LinearLayout.HORIZONTAL);
        row.setGravity(Gravity.CENTER_VERTICAL);
        row.setPadding(0, dp(2), 0, dp(2));
        TextView labelView = text(label, 17, false, MUTED);
        labelView.setSingleLine(true);
        row.addView(labelView, new LinearLayout.LayoutParams(controlLabelWidth(), dp(38)));
        SeekBar seek = new SeekBar(this);
        seek.setPadding(0, 0, 0, 0);
        seek.setThumbOffset(0);
        seek.setMax(9);
        seek.setProgress(initial - 1);
        if (Build.VERSION.SDK_INT >= 21) {
            seek.setProgressTintList(ColorStateList.valueOf(ACCENT));
            seek.setThumbTintList(ColorStateList.valueOf(ACCENT));
            seek.setProgressBackgroundTintList(ColorStateList.valueOf(Color.rgb(235, 235, 235)));
        }
        row.addView(seek, new LinearLayout.LayoutParams(0, dp(38), 1f));
        TextView val = text(formatConfidenceLabel(initial), 15, false, MUTED);
        val.setSingleLine(true);
        val.setGravity(Gravity.RIGHT | Gravity.CENTER_VERTICAL);
        row.addView(val, new LinearLayout.LayoutParams(sliderValueWidth(), dp(38)));
        seek.setOnSeekBarChangeListener(new SeekBar.OnSeekBarChangeListener() {
            private int pending = initial;

            @Override
            public void onProgressChanged(SeekBar seekBar, int progress, boolean fromUser) {
                pending = NativeSettings.clampSliderLevel(progress + 1);
                val.setText(formatConfidenceLabel(pending));
            }

            @Override
            public void onStartTrackingTouch(SeekBar seekBar) {
            }

            @Override
            public void onStopTrackingTouch(SeekBar seekBar) {
                setter.setLevel(pending);
            }
        });
        return row;
    }

    private View textScaleRow() {
        LinearLayout row = new LinearLayout(this);
        row.setOrientation(LinearLayout.HORIZONTAL);
        row.setGravity(Gravity.CENTER_VERTICAL);
        row.setPadding(0, dp(2), 0, dp(2));
        TextView labelView = text("Cỡ chữ:", 17, false, MUTED);
        labelView.setSingleLine(true);
        row.addView(labelView, new LinearLayout.LayoutParams(controlLabelWidth(), dp(38)));

        SeekBar seek = new SeekBar(this);
        seek.setPadding(0, 0, 0, 0);
        seek.setThumbOffset(0);
        seek.setMax(4);
        int current = settings.uiTextScalePercent();
        seek.setProgress(Math.max(0, Math.min(4, (current - 100) / 10)));
        if (Build.VERSION.SDK_INT >= 21) {
            seek.setProgressTintList(ColorStateList.valueOf(ACCENT));
            seek.setThumbTintList(ColorStateList.valueOf(ACCENT));
            seek.setProgressBackgroundTintList(ColorStateList.valueOf(Color.rgb(235, 235, 235)));
        }
        row.addView(seek, new LinearLayout.LayoutParams(0, dp(38), 1f));

        TextView val = text(formatTextScale(current), 15, false, MUTED);
        val.setSingleLine(true);
        val.setGravity(Gravity.RIGHT | Gravity.CENTER_VERTICAL);
        row.addView(val, new LinearLayout.LayoutParams(sliderValueWidth(), dp(38)));
        seek.setOnSeekBarChangeListener(new SeekBar.OnSeekBarChangeListener() {
            private int pending = current;

            @Override
            public void onProgressChanged(SeekBar seekBar, int progress, boolean fromUser) {
                pending = 100 + progress * 10;
                val.setText(formatTextScale(pending));
            }

            @Override
            public void onStartTrackingTouch(SeekBar seekBar) {
            }

            @Override
            public void onStopTrackingTouch(SeekBar seekBar) {
                if (pending != settings.uiTextScalePercent()) {
                    settings.setUiTextScalePercent(pending);
                    rebuildUiAfterTextScaleChange();
                }
            }
        });
        return row;
    }

    private String formatConfidenceLabel(int value) {
        int level = NativeSettings.clampSliderLevel(value);
        return confidenceLabel(level) + " (" + level + ")";
    }

    private String confidenceLabel(int level) {
        if (level <= 2) return "Rất ít";
        if (level <= 4) return "Ít";
        if (level <= 6) return "Vừa";
        if (level <= 8) return "Nhiều";
        return "Rất nhiều";
    }

    private String formatTextScale(int scale) {
        return scale + "%";
    }

    private void rebuildUiAfterTextScaleChange() {
        int restoreScrollY = mainScrollView == null ? 0 : mainScrollView.getScrollY();
        String restoreName = selectedDisplayName;
        long restoreSize = selectedFileSizeBytes;
        boolean hasTranscript = !editorSegments.isEmpty();
        boolean hasPlayer = mediaPlayer != null;
        setContentView(buildContentView());
        if (restoreName != null) updateSelectedFileUi(restoreName, restoreSize);
        if (hasTranscript) {
            renderEditorTranscript();
        } else if (resultPanelView != null) {
            resultPanelView.setVisibility(View.GONE);
        }
        if (fixedPlayerView != null) fixedPlayerView.setVisibility(hasPlayer ? View.VISIBLE : View.GONE);
        updatePlayerTime();
        if (mainScrollView != null) {
            mainScrollView.post(() -> mainScrollView.scrollTo(0, restoreScrollY));
        }
    }

    private int controlLabelWidth() {
        int width = getResources().getDisplayMetrics().widthPixels;
        return Math.max(dp(124), Math.min(dp(168), Math.round(width * 0.30f)));
    }

    private int sliderValueWidth() {
        int width = getResources().getDisplayMetrics().widthPixels;
        return Math.max(dp(58), Math.min(dp(78), Math.round(width * 0.14f)));
    }

    private View fullCheck(String label, boolean checked, String key) {
        CheckBox box = checkPlain(label, checked);
        box.setPadding(0, dp(2), 0, dp(2));
        box.setTextSize(TypedValue.COMPLEX_UNIT_PX, 16 * uiScale());
        box.setLineSpacing(0, 1.0f);
        box.setSingleLine(false);
        box.setOnCheckedChangeListener((button, isChecked) -> settings.set(key, isChecked));
        return box;
    }

    private Spinner speakerSpinner() {
        Spinner spinner = spinner(new String[]{
                "Senko CAM++ (Optimized - Nhanh)",
                "Pyannote Community-1 VBx"
        });
        speakerModelSpinner = spinner;
        spinner.setSelection(NativeSettings.SPEAKER_PYANNOTE.equals(settings.speakerModel()) ? 1 : 0);
        spinner.setOnItemSelectedListener(new android.widget.AdapterView.OnItemSelectedListener() {
            @Override
            public void onItemSelected(android.widget.AdapterView<?> parent, View view, int position, long id) {
                settings.setSpeakerModel(position == 1 ? NativeSettings.SPEAKER_PYANNOTE : NativeSettings.SPEAKER_CAMPP);
            }

            @Override
            public void onNothingSelected(android.widget.AdapterView<?> parent) {
            }
        });
        return spinner;
    }

    private Spinner spinner(String[] values) {
        ArrayAdapter<String> adapter = new ArrayAdapter<String>(this, android.R.layout.simple_spinner_item, values) {
            @Override
            public View getView(int position, View convertView, android.view.ViewGroup parent) {
                TextView view = (TextView) super.getView(position, convertView, parent);
                view.setTextColor(TEXT);
                view.setTextSize(TypedValue.COMPLEX_UNIT_PX, 19 * uiScale());
                view.setSingleLine(true);
                return view;
            }

            @Override
            public View getDropDownView(int position, View convertView, android.view.ViewGroup parent) {
                TextView view = (TextView) super.getDropDownView(position, convertView, parent);
                view.setTextColor(Color.BLACK);
                view.setTextSize(TypedValue.COMPLEX_UNIT_PX, 17 * uiScale());
                return view;
            }
        };
        adapter.setDropDownViewResource(android.R.layout.simple_spinner_dropdown_item);
        Spinner spinner = new Spinner(this);
        spinner.setAdapter(adapter);
        spinner.setPadding(dp(10), 0, dp(10), 0);
        spinner.setBackground(rounded(INPUT, BORDER, 1, 4));
        return spinner;
    }

    private Spinner colorSpinner(int selectedIndex) {
        ArrayAdapter<String> adapter = new ArrayAdapter<String>(this, android.R.layout.simple_spinner_item, SPEAKER_COLOR_LABELS) {
            @Override
            public View getView(int position, View convertView, android.view.ViewGroup parent) {
                TextView view = (TextView) super.getView(position, convertView, parent);
                view.setTextColor(SPEAKER_COLORS[Math.max(0, position) % SPEAKER_COLORS.length]);
                view.setTextSize(TypedValue.COMPLEX_UNIT_PX, 17 * uiScale());
                view.setTypeface(Typeface.DEFAULT, Typeface.BOLD);
                return view;
            }

            @Override
            public View getDropDownView(int position, View convertView, android.view.ViewGroup parent) {
                TextView view = (TextView) super.getDropDownView(position, convertView, parent);
                view.setTextColor(SPEAKER_COLORS[Math.max(0, position) % SPEAKER_COLORS.length]);
                view.setTextSize(TypedValue.COMPLEX_UNIT_PX, 17 * uiScale());
                view.setTypeface(Typeface.DEFAULT, Typeface.BOLD);
                return view;
            }
        };
        adapter.setDropDownViewResource(android.R.layout.simple_spinner_dropdown_item);
        Spinner spinner = new Spinner(this);
        spinner.setAdapter(adapter);
        spinner.setPadding(dp(10), 0, dp(10), 0);
        spinner.setBackground(rounded(INPUT, BORDER, 1, 4));
        spinner.setSelection(Math.max(0, Math.min(SPEAKER_COLORS.length - 1, selectedIndex)));
        return spinner;
    }

    private void renderColorDots(LinearLayout row, int[] selectedColor) {
        row.removeAllViews();
        row.setOrientation(LinearLayout.HORIZONTAL);
        row.setGravity(Gravity.CENTER_VERTICAL);
        row.setPadding(0, dp(8), 0, dp(6));
        for (int color : SPEAKER_COLORS) {
            TextView dot = new TextView(this);
            GradientDrawable bg = new GradientDrawable();
            bg.setShape(GradientDrawable.OVAL);
            bg.setColor(color);
            bg.setStroke(dp(selectedColor[0] == color ? 4 : 1), selectedColor[0] == color ? TEXT : BORDER);
            dot.setBackground(bg);
            dot.setOnClickListener(v -> {
                selectedColor[0] = color;
                renderColorDots(row, selectedColor);
            });
            LinearLayout.LayoutParams params = new LinearLayout.LayoutParams(dp(34), dp(34));
            params.setMargins(0, 0, dp(10), 0);
            row.addView(dot, params);
        }
    }

    private LinearLayout actionRow() {
        LinearLayout row = new LinearLayout(this);
        row.setOrientation(LinearLayout.HORIZONTAL);
        row.setGravity(Gravity.LEFT);
        row.setPadding(0, 0, 0, dp(8));
        return row;
    }

    private LinearLayout.LayoutParams actionButtonParams() {
        LinearLayout.LayoutParams params = new LinearLayout.LayoutParams(0, dp(46), 1f);
        params.rightMargin = dp(10);
        return params;
    }

    private TextView hint(String value) {
        TextView view = text(value, 14, false, MUTED);
        view.setGravity(Gravity.LEFT);
        return view;
    }

    private void addPromptHint(LinearLayout parent, String value, int maxWidth) {
        TextView view = hint(value);
        view.setMaxWidth(maxWidth);
        parent.addView(view);
    }

    private CheckBox checkPlain(String label, boolean checked) {
        CheckBox box = new CheckBox(this);
        box.setText(label);
        box.setTextColor(MUTED);
        box.setTextSize(TypedValue.COMPLEX_UNIT_PX, 16 * uiScale());
        box.setChecked(checked);
        box.setGravity(Gravity.CENTER_VERTICAL);
        if (Build.VERSION.SDK_INT >= 21) {
            box.setButtonTintList(ColorStateList.valueOf(ACCENT));
        }
        return box;
    }

    private Button button(String label, int color, View.OnClickListener listener) {
        Button button = new Button(this);
        button.setText(label);
        button.setTextColor(TEXT);
        button.setTextSize(TypedValue.COMPLEX_UNIT_PX, 17 * uiScale());
        button.setAllCaps(false);
        button.setGravity(Gravity.CENTER);
        button.setPadding(dp(8), 0, dp(8), 0);
        button.setBackground(rounded(color, BORDER, 1, 5));
        button.setOnClickListener(listener);
        return button;
    }

    private TextView badge(String label) {
        TextView badge = text(label, 16, true, TEXT);
        badge.setGravity(Gravity.CENTER);
        badge.setPadding(dp(12), 0, dp(12), 0);
        badge.setBackground(rounded(SUCCESS, SUCCESS, 1, 4));
        return badge;
    }

    private LinearLayout panel() {
        LinearLayout panel = vertical();
        panel.setBackground(rounded(CARD, BORDER, 1, 8));
        return panel;
    }

    private LinearLayout vertical() {
        LinearLayout view = new LinearLayout(this);
        view.setOrientation(LinearLayout.VERTICAL);
        return view;
    }

    private TextView text(String value, int sp, boolean bold, int color) {
        TextView view = new TextView(this);
        view.setText(value);
        view.setTextColor(color);
        view.setTextSize(TypedValue.COMPLEX_UNIT_PX, sp * uiScale());
        view.setGravity(Gravity.CENTER_VERTICAL);
        if (bold) view.setTypeface(Typeface.DEFAULT, Typeface.BOLD);
        return view;
    }

    private GradientDrawable rounded(int fill, int stroke, int strokeWidth, int radius) {
        GradientDrawable drawable = new GradientDrawable();
        drawable.setColor(fill);
        drawable.setCornerRadius(dp(radius));
        drawable.setStroke(dp(strokeWidth), stroke);
        return drawable;
    }

    private View withMargins(View view, int left, int top, int right, int bottom) {
        LinearLayout.LayoutParams params = new LinearLayout.LayoutParams(
                LinearLayout.LayoutParams.MATCH_PARENT,
                LinearLayout.LayoutParams.WRAP_CONTENT);
        params.setMargins(dp(left), dp(top), dp(right), dp(bottom));
        view.setLayoutParams(params);
        return view;
    }

    private int dp(int value) {
        return Math.round(value * layoutScale());
    }

    private float uiScale() {
        return layoutScale() * (settings == null ? 120f : settings.uiTextScalePercent()) / 100f;
    }

    private float layoutScale() {
        return Math.max(1f, Math.min(1.22f, getResources().getDisplayMetrics().widthPixels / 720f));
    }

    private int statusBarHeight() {
        return 0;
    }

    private void enableStandaloneChrome() {
        getWindow().setFlags(WindowManager.LayoutParams.FLAG_FULLSCREEN, WindowManager.LayoutParams.FLAG_FULLSCREEN);
        getWindow().getDecorView().setSystemUiVisibility(
                View.SYSTEM_UI_FLAG_FULLSCREEN
                        | View.SYSTEM_UI_FLAG_HIDE_NAVIGATION
                        | View.SYSTEM_UI_FLAG_IMMERSIVE_STICKY
                        | View.SYSTEM_UI_FLAG_LAYOUT_FULLSCREEN
                        | View.SYSTEM_UI_FLAG_LAYOUT_HIDE_NAVIGATION
                        | View.SYSTEM_UI_FLAG_LAYOUT_STABLE);
    }

    private String formatBytes(long bytes) {
        if (bytes >= 1024 * 1024) return String.format(Locale.US, "%.1f MB", bytes / 1024.0 / 1024.0);
        if (bytes >= 1024) return String.format(Locale.US, "%.1f KB", bytes / 1024.0);
        return bytes + " B";
    }

    private void openHotwordDialog() {
        LinearLayout root = vertical();
        root.setPadding(dp(14), dp(12), dp(14), dp(12));
        root.setBackground(rounded(CARD, BORDER, 1, 8));

        activeHotwordItems = parseHotwordItems(settings.hotwordsText());

        TextView panelTitle = text("Quản lý hotword.txt", 22, true, TEXT);
        panelTitle.setPadding(dp(4), 0, dp(4), dp(10));
        root.addView(panelTitle, new LinearLayout.LayoutParams(
                LinearLayout.LayoutParams.MATCH_PARENT,
                dp(46)));

        LinearLayout addRow = new LinearLayout(this);
        addRow.setOrientation(LinearLayout.HORIZONTAL);
        addRow.setGravity(Gravity.CENTER_VERTICAL);
        addRow.setPadding(0, 0, 0, dp(8));
        EditText addInput = darkSingleLineInput("Nhập hotword mới");
        addRow.addView(addInput, new LinearLayout.LayoutParams(0, dp(46), 1f));
        Button addButton = button("Thêm", PRIMARY, v -> {
            addHotwordFromInput(addInput);
        });
        LinearLayout.LayoutParams addButtonParams = new LinearLayout.LayoutParams(dp(92), dp(46));
        addButtonParams.leftMargin = dp(8);
        addRow.addView(addButton, addButtonParams);
        root.addView(addRow);

        LinearLayout toolbar = new LinearLayout(this);
        toolbar.setOrientation(LinearLayout.HORIZONTAL);
        toolbar.setGravity(Gravity.CENTER_VERTICAL);
        toolbar.setPadding(0, 0, 0, dp(8));
        activeHotwordSummary = text("0 hotword", 15, false, MUTED);
        toolbar.addView(activeHotwordSummary, new LinearLayout.LayoutParams(0, dp(42), 1f));
        toolbar.addView(button("Tải .txt", INPUT, v -> exportHotwordTxt()), compactHotwordButtonParams());
        toolbar.addView(button("Nạp .txt", INPUT, v -> importHotwordTxt()), compactHotwordButtonParams());
        toolbar.addView(button("Khôi phục", INPUT, v -> {
            settings.resetHotwords();
            activeHotwordItems = parseHotwordItems(settings.defaultHotwordsText());
            renderHotwordRows();
        }), compactHotwordButtonParams());
        root.addView(toolbar);

        activeHotwordSearch = darkSingleLineInput("Tìm hotword...");
        activeHotwordSearch.addTextChangedListener(new TextWatcher() {
            @Override public void beforeTextChanged(CharSequence s, int start, int count, int after) {}
            @Override public void onTextChanged(CharSequence s, int start, int before, int count) {
                renderHotwordRows();
            }
            @Override public void afterTextChanged(Editable s) {}
        });
        root.addView(activeHotwordSearch, new LinearLayout.LayoutParams(
                LinearLayout.LayoutParams.MATCH_PARENT,
                dp(44)));

        ScrollView scroll = new ScrollView(this);
        scroll.setPadding(0, dp(8), 0, 0);
        activeHotwordList = vertical();
        scroll.addView(activeHotwordList);
        root.addView(scroll, new LinearLayout.LayoutParams(
                LinearLayout.LayoutParams.MATCH_PARENT,
                dp(360)));
        renderHotwordRows();

        final AlertDialog[] dialogRef = new AlertDialog[1];
        LinearLayout footer = new LinearLayout(this);
        footer.setOrientation(LinearLayout.HORIZONTAL);
        footer.setGravity(Gravity.RIGHT | Gravity.CENTER_VERTICAL);
        footer.setPadding(0, dp(10), 0, 0);
        Button cancel = button("Hủy", INPUT, v -> {
            if (dialogRef[0] != null) dialogRef[0].dismiss();
        });
        Button save = button("Lưu", PRIMARY, v -> {
            settings.setHotwordsText(buildHotwordsText(activeHotwordItems));
            if (dialogRef[0] != null) dialogRef[0].dismiss();
        });
        footer.addView(cancel, new LinearLayout.LayoutParams(dp(120), dp(46)));
        LinearLayout.LayoutParams saveParams = new LinearLayout.LayoutParams(dp(120), dp(46));
        saveParams.leftMargin = dp(8);
        footer.addView(save, saveParams);
        root.addView(footer);

        AlertDialog dialog = new AlertDialog.Builder(this)
                .setView(root)
                .create();
        dialogRef[0] = dialog;
        dialog.setOnDismissListener(d -> {
            clearActiveHotwordDialogState();
        });
        dialog.show();
        Window window = dialog.getWindow();
        if (window != null) {
            window.setBackgroundDrawable(new ColorDrawable(Color.TRANSPARENT));
        }
        addInput.requestFocus();
    }

    private EditText darkSingleLineInput(String hint) {
        EditText input = new EditText(this);
        input.setHint(hint);
        input.setSingleLine(true);
        input.setTextColor(TEXT);
        input.setHintTextColor(Color.rgb(145, 145, 145));
        input.setTextSize(TypedValue.COMPLEX_UNIT_PX, 16 * uiScale());
        input.setInputType(InputType.TYPE_CLASS_TEXT);
        input.setPadding(dp(10), 0, dp(10), 0);
        input.setBackground(rounded(INPUT, BORDER, 1, 4));
        return input;
    }

    private void addHotwordFromInput(EditText input) {
        String value = normalizeHotword(input == null ? "" : input.getText().toString());
        if (value.isEmpty()) return;
        if (activeHotwordItems == null) activeHotwordItems = new ArrayList<>();
        for (HotwordItem item : activeHotwordItems) {
            if (value.equals(item.text)) {
                if (input != null) input.selectAll();
                return;
            }
        }
        if (activeHotwordItems.size() >= MAX_HOTWORDS) return;
        activeHotwordItems.add(new HotwordItem(value, DEFAULT_HOTWORD_SCORE));
        if (input != null) input.setText("");
        renderHotwordRows();
    }

    private void renderHotwordRows() {
        if (activeHotwordList == null) return;
        activeHotwordList.removeAllViews();
        List<HotwordItem> items = activeHotwordItems == null ? new ArrayList<>() : activeHotwordItems;
        String query = normalizeHotword(activeHotwordSearch == null ? "" : activeHotwordSearch.getText().toString());
        int visible = 0;
        for (int i = 0; i < items.size(); i++) {
            HotwordItem item = items.get(i);
            if (!query.isEmpty() && !normalizeHotword(item.text).contains(query)) continue;
            activeHotwordList.addView(hotwordRowView(item, i));
            visible++;
        }
        if (activeHotwordSummary != null) {
            String prefix = query.isEmpty() ? String.valueOf(items.size()) : visible + "/" + items.size();
            activeHotwordSummary.setText(prefix + " hotword");
        }
    }

    private View hotwordRowView(HotwordItem item, int index) {
        LinearLayout row = new LinearLayout(this);
        row.setOrientation(LinearLayout.HORIZONTAL);
        row.setGravity(Gravity.CENTER_VERTICAL);
        row.setPadding(dp(8), dp(5), dp(8), dp(5));
        row.setBackground(rounded(CARD, BORDER, 1, 4));

        TextView number = text(String.valueOf(index + 1), 14, false, MUTED);
        number.setGravity(Gravity.RIGHT | Gravity.CENTER_VERTICAL);
        row.addView(number, new LinearLayout.LayoutParams(dp(44), dp(40)));

        EditText textInput = darkSingleLineInput("");
        textInput.setText(item.text);
        textInput.addTextChangedListener(new TextWatcher() {
            @Override public void beforeTextChanged(CharSequence s, int start, int count, int after) {}
            @Override public void onTextChanged(CharSequence s, int start, int before, int count) {
                item.text = normalizeHotword(s == null ? "" : s.toString());
            }
            @Override public void afterTextChanged(Editable s) {}
        });
        row.addView(textInput, new LinearLayout.LayoutParams(0, dp(40), 1f));

        Button remove = button("×", DANGER, v -> {
            if (activeHotwordItems != null) {
                activeHotwordItems.remove(item);
                renderHotwordRows();
            }
        });
        remove.setTextSize(TypedValue.COMPLEX_UNIT_PX, 18 * uiScale());
        LinearLayout.LayoutParams removeParams = new LinearLayout.LayoutParams(dp(42), dp(40));
        removeParams.leftMargin = dp(6);
        row.addView(remove, removeParams);
        return withMargins(row, 0, 0, 0, 5);
    }

    private List<HotwordItem> parseHotwordItems(String text) {
        List<HotwordItem> items = new ArrayList<>();
        if (text == null) return items;
        for (String raw : text.split("\\R")) {
            String line = raw == null ? "" : raw.trim();
            if (line.isEmpty() || line.startsWith("#")) continue;
            double score = DEFAULT_HOTWORD_SCORE;
            int colon = line.lastIndexOf(':');
            if (colon > 0) {
                try {
                    score = Math.max(0.0, Math.min(8.0, Double.parseDouble(line.substring(colon + 1).trim())));
                    line = line.substring(0, colon).trim();
                } catch (NumberFormatException ignored) {
                }
            }
            String hotword = normalizeHotword(line);
            if (hotword.isEmpty()) continue;
            items.add(new HotwordItem(hotword, score));
            if (items.size() >= MAX_HOTWORDS) break;
        }
        return items;
    }

    private String buildHotwordsText(List<HotwordItem> items) {
        StringBuilder out = new StringBuilder();
        if (items == null) return "";
        for (HotwordItem item : items) {
            String hotword = normalizeHotword(item.text);
            if (hotword.isEmpty()) continue;
            if (out.length() > 0) out.append('\n');
            out.append(hotword)
                    .append(" :")
                    .append(formatHotwordScore(item.score));
        }
        return out.toString();
    }

    private String formatHotwordScore(double score) {
        String value = String.format(Locale.US, "%.2f", Math.max(0.0, Math.min(8.0, score)));
        while (value.contains(".") && value.endsWith("0")) value = value.substring(0, value.length() - 1);
        if (value.endsWith(".")) value = value.substring(0, value.length() - 1);
        return value;
    }

    private String normalizeHotword(String value) {
        return TokenParser.normalizeHotwordText(value);
    }

    private void clearActiveHotwordDialogState() {
        activeHotwordList = null;
        activeHotwordSummary = null;
        activeHotwordSearch = null;
        activeHotwordItems = null;
    }

    private LinearLayout.LayoutParams compactHotwordButtonParams() {
        LinearLayout.LayoutParams params = new LinearLayout.LayoutParams(0, dp(42), 1f);
        params.rightMargin = dp(6);
        return params;
    }

    private void importHotwordTxt() {
        Intent intent = new Intent(Intent.ACTION_OPEN_DOCUMENT);
        intent.addCategory(Intent.CATEGORY_OPENABLE);
        intent.setType("text/*");
        intent.putExtra(Intent.EXTRA_MIME_TYPES, new String[]{"text/plain", "text/*"});
        startActivityForResult(intent, REQUEST_IMPORT_HOTWORD_TXT);
    }

    private void exportHotwordTxt() {
        pendingHotwordExportText = activeHotwordItems == null
                ? settings.hotwordsText()
                : buildHotwordsText(activeHotwordItems);
        Intent intent = new Intent(Intent.ACTION_CREATE_DOCUMENT);
        intent.addCategory(Intent.CATEGORY_OPENABLE);
        intent.setType("text/plain");
        intent.putExtra(Intent.EXTRA_TITLE, "hotword.txt");
        startActivityForResult(intent, REQUEST_EXPORT_HOTWORD_TXT);
    }

    private void openMediaPicker() {
        Intent intent = new Intent(Intent.ACTION_OPEN_DOCUMENT);
        intent.addCategory(Intent.CATEGORY_OPENABLE);
        intent.setType("*/*");
        intent.putExtra(Intent.EXTRA_MIME_TYPES, new String[]{
                "audio/*",
                "video/*",
                "audio/mpeg",
                "audio/wav",
                "audio/x-wav",
                "audio/mp4",
                "audio/x-m4a",
                "audio/flac",
                "audio/aac",
                "audio/x-ms-wma",
                "audio/ogg",
                "audio/opus",
                "video/mp4",
                "video/x-matroska",
                "video/x-msvideo",
                "video/quicktime",
                "video/webm",
                "video/x-flv",
                "video/x-ms-wmv"
        });
        startActivityForResult(intent, REQUEST_OPEN_MEDIA);
    }

    @Override
    protected void onActivityResult(int requestCode, int resultCode, Intent data) {
        super.onActivityResult(requestCode, resultCode, data);
        if (requestCode == REQUEST_IMPORT_HOTWORD_TXT) {
            if (resultCode == RESULT_OK && data != null && data.getData() != null) {
                try {
                    String text = readTextUri(data.getData());
                    if (activeHotwordItems != null) {
                        activeHotwordItems = parseHotwordItems(text);
                        renderHotwordRows();
                    } else {
                        settings.setHotwordsText(text);
                    }
                    appendLog("Imported hotword.txt");
                } catch (Exception error) {
                    appendLog("Import hotword.txt failed: " + error.getMessage());
                }
            }
            return;
        }
        if (requestCode == REQUEST_EXPORT_HOTWORD_TXT) {
            if (resultCode == RESULT_OK && data != null && data.getData() != null) {
                try {
                    writeTextUri(data.getData(), pendingHotwordExportText == null ? settings.hotwordsText() : pendingHotwordExportText);
                    appendLog("Exported hotword.txt");
                } catch (Exception error) {
                    appendLog("Export hotword.txt failed: " + error.getMessage());
                } finally {
                    pendingHotwordExportText = null;
                }
            }
            return;
        }
        if (requestCode != REQUEST_OPEN_MEDIA || resultCode != RESULT_OK || data == null) return;
        selectedUri = data.getData();
        if (selectedUri == null) return;
        int flags = data.getFlags() & Intent.FLAG_GRANT_READ_URI_PERMISSION;
        if (flags != 0) getContentResolver().takePersistableUriPermission(selectedUri, flags);
        selectedLibraryItemId = null;
        updateSelectedFileUi(displayName(selectedUri), -1);
        prepareMediaPlayer(selectedUri);
        clearTranscript();
        appendLog("Selected: " + selectedUri);
    }

    private String readTextUri(Uri uri) throws Exception {
        try (InputStream input = getContentResolver().openInputStream(uri)) {
            if (input == null) return "";
            byte[] buffer = new byte[8192];
            ByteArrayOutputStream bytes = new ByteArrayOutputStream();
            int read;
            while ((read = input.read(buffer)) >= 0) {
                bytes.write(buffer, 0, read);
            }
            return new String(bytes.toByteArray(), StandardCharsets.UTF_8);
        }
    }

    private void writeTextUri(Uri uri, String text) throws Exception {
        try (OutputStream output = getContentResolver().openOutputStream(uri, "wt")) {
            if (output != null) output.write((text == null ? "" : text).getBytes(StandardCharsets.UTF_8));
        }
    }

    private String displayName(Uri uri) {
        try (Cursor cursor = getContentResolver().query(uri, null, null, null, null)) {
            if (cursor != null && cursor.moveToFirst()) {
                int index = cursor.getColumnIndex(OpenableColumns.DISPLAY_NAME);
                if (index >= 0) return cursor.getString(index);
            }
        } catch (Exception ignored) {
        }
        return uri.getLastPathSegment() == null ? "selected file" : new File(uri.getLastPathSegment()).getName();
    }

    private void updateSelectedFileUi(String name, long size) {
        selectedDisplayName = name;
        selectedFileSizeBytes = size;
        if (dropPromptView != null) dropPromptView.setVisibility(name == null ? View.VISIBLE : View.GONE);
        if (selectedFileView != null) selectedFileView.setVisibility(name == null ? View.GONE : View.VISIBLE);
        if (fileNameView != null) fileNameView.setText(name == null ? "" : name);
        if (fileSizeView != null) fileSizeView.setText(size > 0 ? formatBytes(size) : "");
        if (dropZoneView != null) {
            android.view.ViewGroup.LayoutParams params = dropZoneView.getLayoutParams();
            if (params != null) {
                params.height = dp(name == null ? 360 : 86);
                dropZoneView.setLayoutParams(params);
            }
        }
    }

    private void clearSelectedFile() {
        selectedUri = null;
        selectedLibraryItemId = null;
        updateSelectedFileUi(null, 0);
        releaseMediaPlayer();
        clearTranscript();
        if (resultPanelView != null) resultPanelView.setVisibility(View.GONE);
        if (progressContainer != null) progressContainer.setVisibility(View.GONE);
    }

    private void openLibraryItem(NativeFileLibrary.LibraryItem item) {
        closeDrawer();
        selectedLibraryItemId = item.id;
        selectedUri = Uri.fromFile(item.sourceFile);
        updateSelectedFileUi(item.displayName, item.sourceFile.length());
        prepareMediaPlayer(selectedUri);
        if (progressContainer != null) progressContainer.setVisibility(View.GONE);
        try {
            String json = library.readResult(item.id);
            if (json != null && !json.trim().isEmpty()) {
                renderTranscriptFromJson(json);
            } else {
                clearTranscript();
                if (resultPanelView != null) resultPanelView.setVisibility(View.GONE);
            }
            appendLog("Opened library item: " + item.displayName);
        } catch (Exception error) {
            appendLog("Open library item failed: " + error.getMessage());
        }
    }

    private void handleDebugIntent(Intent intent) {
        if (intent == null || !intent.hasExtra("debug_input_path")) return;
        String debugInputPath = intent.getStringExtra("debug_input_path");
        String speaker = intent.getStringExtra("debug_speaker_model");
        if (NativeSettings.SPEAKER_PYANNOTE.equals(speaker) || NativeSettings.SPEAKER_CAMPP.equals(speaker)) {
            pendingDebugSpeakerModel = speaker;
            settings.setSpeakerModel(speaker);
            if (speakerModelSpinner != null) {
                speakerModelSpinner.setSelection(NativeSettings.SPEAKER_PYANNOTE.equals(speaker) ? 1 : 0);
            }
        }
        if (intent.hasExtra("debug_accelerator")) {
            Object value = intent.getExtras() == null ? null : intent.getExtras().get("debug_accelerator");
            boolean enabled = value instanceof Boolean ? (Boolean) value : Boolean.parseBoolean(String.valueOf(value));
            settings.set("accelerator", enabled);
        }
        settings.set("diarization", true);
        File file = new File(debugInputPath);
        selectedUri = Uri.fromFile(file);
        selectedLibraryItemId = null;
        updateSelectedFileUi(file.getName(), file.length());
        prepareMediaPlayer(selectedUri);
        clearTranscript();
        appendLog("Debug input: " + file.getAbsolutePath());
        if (intent.getBooleanExtra("debug_diar_only", false)) {
            rootFrame.postDelayed(this::debugDiarizationSelectedFile, 500);
        } else if (intent.getBooleanExtra("debug_dump_only", false)) {
            rootFrame.postDelayed(this::debugDumpSelectedFile, 500);
        } else {
            rootFrame.postDelayed(this::processSelectedFile, 500);
        }
        clearDebugIntentExtras(intent);
    }

    private void clearDebugIntentExtras(Intent intent) {
        if (intent == null) return;
        intent.removeExtra("debug_input_path");
        intent.removeExtra("debug_speaker_model");
        intent.removeExtra("debug_accelerator");
        intent.removeExtra("debug_diar_only");
        intent.removeExtra("debug_dump_only");
        setIntent(intent);
    }

    private void debugDiarizationSelectedFile() {
        if (selectedUri == null) {
            appendLog("No file selected.");
            return;
        }
        setProgress("Diarization dump", 2);
        appendLog("Debug diarization dump started.");
        pipeline.debugDumpDiarization(selectedUri, settings.acceleratorEnabled(), new NativeOfflinePipeline.ProgressListener() {
            @Override
            public void onProgress(String phase, int percent, String message) {
                Log.d(TAG, "Pipeline " + percent + "% " + phase + ": " + message);
                runOnUiThread(() -> {
                    setProgress(phase, percent);
                    appendLog(phase + ": " + message);
                });
            }

            @Override
            public void onError(Throwable error) {
                runOnUiThread(() -> appendLog("Debug diarization failed: " + error.getMessage()));
            }
        });
    }

    private void debugDumpSelectedFile() {
        if (selectedUri == null) {
            appendLog("No file selected.");
            return;
        }
        setProgress("Encoder dump", 2);
        appendLog("Debug encoder dump started.");
        pipeline.debugDumpEncoders(selectedUri, settings.acceleratorEnabled(), new NativeOfflinePipeline.ProgressListener() {
            @Override
            public void onProgress(String phase, int percent, String message) {
                runOnUiThread(() -> {
                    setProgress(phase, percent);
                    appendLog(phase + ": " + message);
                });
            }

            @Override
            public void onError(Throwable error) {
                runOnUiThread(() -> appendLog("Debug dump failed: " + error.getMessage()));
            }
        });
    }

    private void processSelectedFile() {
        if (selectedUri == null) {
            appendLog("No file selected.");
            return;
        }
        if (!allBundledModelsReady()) {
            appendLog("Một số model offline bị thiếu trong APK. Hãy cài lại bản full-model.");
        }
        if (pendingDebugSpeakerModel != null) {
            settings.setSpeakerModel(pendingDebugSpeakerModel);
        }
        clearTranscript();
        if (resultPanelView != null) resultPanelView.setVisibility(View.GONE);
        setProgress("Reading input", 2);
        appendLog("Processing started.");
        pipeline.importFile(selectedUri, new NativeOfflinePipeline.ProgressListener() {
            @Override
            public void onProgress(String phase, int percent, String message) {
                runOnUiThread(() -> {
                    setProgress(phase, percent);
                    appendLog(phase + ": " + message);
                    if (percent >= 100) refreshLibraryList();
                });
            }

            @Override
            public void onComplete(
                    NativeFileLibrary.LibraryItem item,
                    String resultJson,
                    PureOrtRecognizer.DecodeResult asr,
                    DiarizationResult diarization
            ) {
                Log.d(TAG, "Pipeline complete: " + item.id + ", resultJson=" + resultJson.length());
                runOnUiThread(() -> {
                    selectedLibraryItemId = item.id;
                    try {
                        renderTranscriptFromJson(resultJson);
                    } catch (Exception parseError) {
                        Log.w(TAG, "Result JSON render failed, falling back to live ASR render", parseError);
                        renderTranscript(asr, diarization);
                    }
                    if (resultPanelView != null) resultPanelView.setVisibility(View.VISIBLE);
                    refreshLibraryList();
                });
            }

            @Override
            public void onError(Throwable error) {
                Log.e(TAG, "Pipeline failed", error);
                runOnUiThread(() -> {
                    setProgress("Pipeline failed", 100);
                    appendLog("Error: " + error.getMessage());
                    refreshLibraryList();
                });
            }
        });
    }

    private void showInitialModelDownloadPromptIfNeeded() {
        setProgress("Chuẩn bị model offline", 1);
        new Thread(() -> {
            int bundled = models.installBundledModelsIfPresent();
            boolean ready = allBundledModelsReady();
            runOnUiThread(() -> {
                if (bundled > 0) appendLog("Installed bundled model file(s): " + bundled);
                if (ready) {
                    setProgress("Models ready", 100);
                    rootFrame.postDelayed(() -> {
                        if (progressContainer != null) progressContainer.setVisibility(View.GONE);
                    }, 700);
                    return;
                }
                setProgress("Thiếu model offline", 100);
                new AlertDialog.Builder(this)
                        .setTitle("Thiếu model offline")
                        .setMessage("APK này thiếu một số model đóng gói sẵn. Hãy cài lại bản full-model để chạy offline, không cần Hugging Face.")
                        .setPositiveButton("OK", null)
                        .show();
            });
        }, "bundled-model-installer").start();
    }

    private boolean allBundledModelsReady() {
        return models.hasRequiredCoreModels()
                && models.hasCamppModel()
                && models.hasPyannoteModels()
                && models.hasDnsmosModel()
                && models.hasPunctuationModel();
    }

    private void prepareMediaPlayer(Uri uri) {
        releaseMediaPlayer();
        if (uri == null) return;
        try {
            mediaPlayer = new MediaPlayer();
            mediaPlayer.setDataSource(this, uri);
            mediaPlayer.setOnPreparedListener(mp -> {
                if (fixedPlayerView != null) fixedPlayerView.setVisibility(View.VISIBLE);
                updatePlayerTime();
                schedulePlayerTick();
            });
            mediaPlayer.setOnCompletionListener(mp -> updatePlayerTime());
            mediaPlayer.prepare();
        } catch (Exception error) {
            releaseMediaPlayer();
            appendLog("Audio preview unavailable: " + error.getMessage());
        }
    }

    private void releaseMediaPlayer() {
        if (mediaPlayer != null) {
            try {
                mediaPlayer.release();
            } catch (Exception ignored) {
            }
            mediaPlayer = null;
        }
        if (fixedPlayerView != null) fixedPlayerView.setVisibility(View.GONE);
    }

    private void schedulePlayerTick() {
        uiHandler.postDelayed(() -> {
            updatePlayerTime();
            if (mediaPlayer != null) schedulePlayerTick();
        }, 250);
    }

    private void updatePlayerTime() {
        if (playerTimeView == null || playerSeekBar == null) return;
        if (mediaPlayer == null) {
            playerTimeView.setText("00:00 / 00:00");
            playerSeekBar.setProgress(0);
            updatePlayerButton();
            return;
        }
        int duration = Math.max(0, mediaPlayer.getDuration());
        int position = Math.max(0, mediaPlayer.getCurrentPosition());
        playerTimeView.setText(formatClock(position) + " / " + formatClock(duration));
        if (!userSeeking && duration > 0) {
            playerSeekBar.setProgress((int) Math.round((position / (double) duration) * 1000.0));
        }
        updateActiveSegment(position / 1000.0, false);
        updatePlayerButton();
    }

    private void updatePlayerButton() {
        if (playerPlayPauseButton == null) return;
        boolean playing = mediaPlayer != null && mediaPlayer.isPlaying();
        playerPlayPauseButton.setText(playing ? "⏸" : "▶");
        playerPlayPauseButton.setEnabled(selectedUri != null || mediaPlayer != null);
    }

    private void togglePlayback() {
        if (mediaPlayer == null && selectedUri != null) {
            prepareMediaPlayer(selectedUri);
        }
        if (mediaPlayer == null) return;
        if (mediaPlayer.isPlaying()) {
            mediaPlayer.pause();
        } else {
            mediaPlayer.start();
        }
        updatePlayerTime();
    }

    private String formatClock(int millis) {
        int total = Math.max(0, millis / 1000);
        int minutes = total / 60;
        int seconds = total % 60;
        return String.format(Locale.US, "%02d:%02d", minutes, seconds);
    }

    private void seekTo(double seconds) {
        if (selectedUri != null && mediaPlayer == null) {
            prepareMediaPlayer(selectedUri);
        }
        if (mediaPlayer == null) return;
        int targetMs = Math.max(0, (int) Math.round(seconds * 1000.0));
        if (Build.VERSION.SDK_INT >= 26) {
            mediaPlayer.seekTo(targetMs, MediaPlayer.SEEK_CLOSEST);
        } else {
            mediaPlayer.seekTo(targetMs);
        }
        if (!mediaPlayer.isPlaying()) mediaPlayer.start();
        updateActiveSegment(targetMs / 1000.0, true);
        updatePlayerTime();
    }

    private int activeSegmentAtTime(double seconds) {
        if (editorSegments.isEmpty()) return -1;
        double epsilon = 0.035;
        if (activeSegmentIndex >= 0 && activeSegmentIndex < editorSegments.size()) {
            EditorSegment current = editorSegments.get(activeSegmentIndex);
            if (seconds + epsilon >= current.start && seconds < current.end + epsilon) {
                return activeSegmentIndex;
            }
        }
        int index = -1;
        for (int i = 0; i < editorSegments.size(); i++) {
            EditorSegment segment = editorSegments.get(i);
            if (Double.isFinite(segment.start) && segment.start <= seconds + epsilon) {
                index = i;
            }
        }
        return index >= 0 ? index : 0;
    }

    private void updateActiveSegment(double seconds, boolean allowScroll) {
        int index = activeSegmentAtTime(seconds);
        if (index < 0 || index == activeSegmentIndex) {
            if (allowScroll) scrollActiveSegmentIntoView();
            return;
        }
        activeSegmentIndex = index;
        renderEditorTranscript();
        if (allowScroll) scrollActiveSegmentIntoView();
    }

    private void scrollActiveSegmentIntoView() {
        if (mainScrollView == null || transcriptRoot == null || activeSegmentIndex < 0 || activeSegmentIndex >= editorSegments.size()) {
            return;
        }
        long now = System.currentTimeMillis();
        if (now - lastActiveAutoScrollAtMs < 350) return;
        lastActiveAutoScrollAtMs = now;
        int blockStart = findBlockStart(activeSegmentIndex);
        View target = transcriptRoot.findViewWithTag("segment-block-" + blockStart);
        if (target == null) return;
        mainScrollView.post(() -> {
            int y = resultPanelView == null ? target.getTop() : resultPanelView.getTop() + transcriptRoot.getTop() + target.getTop();
            mainScrollView.smoothScrollTo(0, Math.max(0, y - dp(64)));
        });
    }

    private void clearTranscript() {
        currentTranscriptText = "";
        editorSegments.clear();
        editorSpeakers.clear();
        activeSegmentIndex = -1;
        if (transcriptRoot == null) return;
        transcriptRoot.removeAllViews();
        transcriptEmptyView = text("Chưa có nội dung. Chọn file và bấm Xử lý để hiển thị transcript tại đây.", 16, false, MUTED);
        transcriptEmptyView.setPadding(dp(10), dp(8), dp(10), dp(8));
        transcriptRoot.addView(transcriptEmptyView);
    }

    private void renderTranscript(PureOrtRecognizer.DecodeResult asr, DiarizationResult diarization) {
        if (asr == null || asr.words.isEmpty()) {
            clearTranscript();
            return;
        }
        currentTranscriptText = asr.text == null ? "" : asr.text;
        currentDurationSeconds = mediaPlayer == null ? 0 : mediaPlayer.getDuration() / 1000.0;
        List<TimedWord> words = new ArrayList<>();
        for (PureOrtRecognizer.Word word : asr.words) {
            words.add(new TimedWord(word.text, word.start, word.end));
        }
        buildEditorSegments(words, diarization == null ? null : diarization.turns);
        renderEditorTranscript();
    }

    private void renderTranscriptFromJson(String jsonText) throws Exception {
        JSONObject json = new JSONObject(jsonText);
        currentDurationSeconds = json.optDouble("duration_sec", mediaPlayer == null ? 0 : mediaPlayer.getDuration() / 1000.0);
        currentTranscriptText = json.optString("text", "");
        editorSegments.clear();
        editorSpeakers.clear();

        JSONObject names = json.optJSONObject("speaker_names");
        JSONObject colors = json.optJSONObject("speaker_colors");
        if (buildEditorSegmentsFromJsonWords(json, names, colors)) {
            renderEditorTranscript();
            return;
        }

        JSONArray segments = json.optJSONArray("segments");
        if (segments != null && segments.length() > 0) {
            int currentSpeaker = 0;
            for (int i = 0; i < segments.length(); i++) {
                JSONObject item = segments.optJSONObject(i);
                if (item == null) continue;
                String type = item.optString("type", "text");
                if ("speaker".equals(type)) {
                    currentSpeaker = item.optInt("speaker_id", currentSpeaker);
                    speakerMeta(currentSpeaker).name = names == null
                            ? item.optString("speaker", defaultSpeakerName(currentSpeaker))
                            : names.optString(String.valueOf(currentSpeaker), item.optString("speaker", defaultSpeakerName(currentSpeaker)));
                    applySavedSpeakerColor(currentSpeaker, colors);
                    continue;
                }
                if (!"text".equals(type)) continue;
                int speaker = item.has("speaker_id") ? item.optInt("speaker_id", currentSpeaker) : currentSpeaker;
                addTextSegmentMaybeSplit(
                        item.optDouble("start_time", item.optDouble("start", 0)),
                        item.optDouble("end_time", item.optDouble("end", 0)),
                        item.optString("text", ""),
                        speaker);
                speakerMeta(speaker);
                applySavedSpeakerColor(speaker, colors);
            }
        } else {
            JSONArray wordsJson = json.optJSONArray("words");
            List<TimedWord> words = new ArrayList<>();
            if (wordsJson != null) {
                for (int i = 0; i < wordsJson.length(); i++) {
                    JSONObject word = wordsJson.optJSONObject(i);
                    if (word == null) continue;
                    words.add(new TimedWord(word.optString("text", ""), word.optDouble("start", 0), word.optDouble("end", 0)));
                }
            }
            buildEditorSegments(words, null);
            if (editorSegments.isEmpty() && !currentTranscriptText.isEmpty()) {
                buildTextOnlySegments(currentTranscriptText, Math.max(0.01, currentDurationSeconds));
            }
        }
        renderEditorTranscript();
    }

    private boolean buildEditorSegmentsFromJsonWords(JSONObject json, JSONObject names, JSONObject colors) {
        JSONArray wordsJson = json.optJSONArray("words");
        JSONObject diarization = json.optJSONObject("diarization");
        JSONArray turnsJson = diarization == null ? null : diarization.optJSONArray("turns");
        if (wordsJson == null || wordsJson.length() == 0 || turnsJson == null || turnsJson.length() == 0) return false;

        List<TimedWord> words = new ArrayList<>();
        for (int i = 0; i < wordsJson.length(); i++) {
            JSONObject word = wordsJson.optJSONObject(i);
            if (word == null) continue;
            words.add(new TimedWord(
                    word.optString("text", ""),
                    word.optDouble("start", 0),
                    word.optDouble("end", 0)));
        }
        if (words.isEmpty()) return false;

        List<DiarizationResult.Turn> turns = new ArrayList<>();
        for (int i = 0; i < turnsJson.length(); i++) {
            JSONObject turn = turnsJson.optJSONObject(i);
            if (turn == null) continue;
            turns.add(new DiarizationResult.Turn(
                    turn.optDouble("start", 0),
                    turn.optDouble("end", 0),
                    turn.optInt("speaker", 0)));
        }
        if (turns.isEmpty()) return false;

        buildEditorSegments(words, turns);
        for (EditorSegment segment : editorSegments) {
            if (names != null) {
                speakerMeta(segment.speaker).name = names.optString(
                        String.valueOf(segment.speaker),
                        speakerMeta(segment.speaker).name);
            } else {
                speakerMeta(segment.speaker);
            }
            applySavedSpeakerColor(segment.speaker, colors);
        }
        return !editorSegments.isEmpty();
    }

    private void addTextSegmentMaybeSplit(double start, double end, String text, int speaker) {
        String[] words = text == null ? new String[0] : text.trim().split("\\s+");
        if (words.length <= 40) {
            editorSegments.add(new EditorSegment(start, Math.max(start + 0.01, end), text, speaker));
            return;
        }
        int cursor = 0;
        while (cursor < words.length) {
            int next = Math.min(words.length, cursor + 32);
            StringBuilder line = new StringBuilder();
            for (int i = cursor; i < next; i++) {
                if (line.length() > 0) line.append(' ');
                line.append(words[i]);
            }
            double partStart = start + (end - start) * cursor / Math.max(1, words.length);
            double partEnd = start + (end - start) * next / Math.max(1, words.length);
            editorSegments.add(new EditorSegment(partStart, Math.max(partStart + 0.01, partEnd), line.toString(), speaker));
            cursor = next;
        }
    }

    private void buildTextOnlySegments(String text, double duration) {
        String[] parts = text.trim().split("\\s+");
        int cursor = 0;
        int segmentIndex = 0;
        while (cursor < parts.length) {
            int end = Math.min(parts.length, cursor + 32);
            StringBuilder line = new StringBuilder();
            for (int i = cursor; i < end; i++) {
                if (line.length() > 0) line.append(' ');
                line.append(parts[i]);
            }
            double startTime = duration * cursor / Math.max(1, parts.length);
            double endTime = duration * end / Math.max(1, parts.length);
            editorSegments.add(new EditorSegment(startTime, Math.max(startTime + 0.01, endTime), line.toString(), 0));
            cursor = end;
            segmentIndex++;
        }
        speakerMeta(0);
    }

    private void buildEditorSegments(List<TimedWord> words, List<DiarizationResult.Turn> turns) {
        editorSegments.clear();
        editorSpeakers.clear();
        if (words == null || words.isEmpty()) return;
        List<TimedWord> group = new ArrayList<>();
        int currentSpeaker = speakerForWord(words.get(0), turns);
        for (TimedWord word : words) {
            int speaker = speakerForWord(word, turns);
            if (!group.isEmpty() && (speaker != currentSpeaker || shouldFlushSentence(group))) {
                addWordGroup(group, currentSpeaker);
                group.clear();
            }
            group.add(word);
            currentSpeaker = speaker;
        }
        addWordGroup(group, currentSpeaker);
    }

    private boolean shouldFlushSentence(List<TimedWord> group) {
        if (group.size() >= 32) return true;
        if (group.isEmpty()) return false;
        String text = group.get(group.size() - 1).text;
        return text.matches(".*[.!?:;]+[\"')\\]]*$");
    }

    private void addWordGroup(List<TimedWord> group, int speaker) {
        if (group == null || group.isEmpty()) return;
        StringBuilder text = new StringBuilder();
        double start = finiteOr(group.get(0).start, 0.0);
        double end = start;
        for (TimedWord word : group) {
            if (text.length() > 0) text.append(' ');
            text.append(word.text);
            end = Math.max(end, finiteOr(word.end, finiteOr(word.start, end)));
        }
        editorSegments.add(new EditorSegment(
                start,
                Math.max(end, start + 0.01),
                text.toString(),
                speaker,
                new ArrayList<>(group)));
        speakerMeta(speaker);
    }

    private int speakerForWord(TimedWord word, List<DiarizationResult.Turn> turns) {
        if (turns == null || turns.isEmpty()) return 0;
        double start = wordIntervalStart(word);
        double end = wordIntervalEnd(word);
        double center = (start + end) / 2.0;
        DiarizationResult.Turn best = null;
        double bestOverlap = 0.0;
        double bestCenterDistance = Double.POSITIVE_INFINITY;
        for (DiarizationResult.Turn turn : turns) {
            double overlap = intervalOverlap(start, end, turn.start, turn.end);
            if (overlap <= 0.0) continue;
            double centerDistance = Math.abs(((turn.start + turn.end) / 2.0) - center);
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

    private double wordIntervalStart(TimedWord word) {
        double start = finiteOr(word.start, 0.0);
        double end = wordEndOrStart(word.end, start);
        return end < start ? end : start;
    }

    private double wordIntervalEnd(TimedWord word) {
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

    private SpeakerMeta speakerMeta(int speaker) {
        SpeakerMeta meta = editorSpeakers.get(speaker);
        if (meta == null) {
            meta = new SpeakerMeta(defaultSpeakerName(speaker), SPEAKER_COLORS[Math.abs(speaker) % SPEAKER_COLORS.length]);
            editorSpeakers.put(speaker, meta);
        }
        return meta;
    }

    private int colorIndex(int color) {
        for (int i = 0; i < SPEAKER_COLORS.length; i++) {
            if (SPEAKER_COLORS[i] == color) return i;
        }
        return 0;
    }

    private void applySavedSpeakerColor(int speaker, JSONObject colors) {
        if (colors == null) return;
        String value = colors.optString(String.valueOf(speaker), "");
        if (value == null || value.isEmpty()) return;
        try {
            speakerMeta(speaker).color = Color.parseColor(value);
        } catch (Exception ignored) {
        }
    }

    private String colorHex(int color) {
        return String.format(Locale.US, "#%06X", 0xFFFFFF & color);
    }

    private String defaultSpeakerName(int speaker) {
        return "Người nói " + (speaker + 1);
    }

    private void renderEditorTranscript() {
        if (transcriptRoot == null) return;
        final int restoreScrollY = mainScrollView == null ? 0 : mainScrollView.getScrollY();
        transcriptRoot.removeAllViews();
        if (resultPanelView != null) resultPanelView.setVisibility(View.VISIBLE);
        if (fixedPlayerView != null && mediaPlayer != null) fixedPlayerView.setVisibility(View.VISIBLE);
        if (editorSegments.isEmpty()) {
            clearTranscript();
            return;
        }
        currentTranscriptText = transcriptText();
        int start = 0;
        while (start < editorSegments.size()) {
            int speaker = editorSegments.get(start).speaker;
            int end = start;
            while (end + 1 < editorSegments.size() && editorSegments.get(end + 1).speaker == speaker) end++;
            transcriptRoot.addView(speakerBlock(speaker, start, end));
            start = end + 1;
        }
        if (mainScrollView != null) {
            mainScrollView.post(() -> mainScrollView.scrollTo(0, restoreScrollY));
        }
    }

    private View speakerBlock(int speakerId, int startIndex, int endIndex) {
        SpeakerMeta meta = speakerMeta(speakerId);
        boolean activeBlock = activeSegmentIndex >= startIndex && activeSegmentIndex <= endIndex;
        LinearLayout block = vertical();
        block.setTag("segment-block-" + startIndex);
        block.setFocusable(false);
        block.setPadding(dp(10), dp(6), dp(10), dp(8));
        GradientDrawable bg = rounded(activeBlock ? Color.rgb(78, 78, 78) : ELEVATED, BORDER, 1, 4);
        bg.setStroke(dp(activeBlock ? 4 : 3), meta.color);
        block.setBackground(bg);
        block.setOnLongClickListener(v -> {
            showSegmentActions(activeBlock ? activeSegmentIndex : startIndex);
            return true;
        });

        TextView speaker = text(meta.name + ":", 17, true, meta.color);
        speaker.setPadding(0, 0, 0, dp(2));
        speaker.setOnClickListener(v -> showRenameDialog(speakerId, false, startIndex));
        block.addView(speaker, new LinearLayout.LayoutParams(
                LinearLayout.LayoutParams.MATCH_PARENT, dp(28)));

        TextView content = text("", 17, false, TEXT);
        content.setLineSpacing(0, 0.96f);
        content.setText(segmentSpans(startIndex, endIndex));
        content.setMovementMethod(LinkMovementMethod.getInstance());
        content.setHighlightColor(Color.argb(45, 0, 123, 255));
        content.setTextIsSelectable(false);
        content.setFocusable(false);
        content.setFocusableInTouchMode(false);
        content.setOnLongClickListener(v -> {
            showSegmentActions(activeBlock ? activeSegmentIndex : startIndex);
            return true;
        });
        block.addView(content);
        return withMargins(block, 0, 0, 0, 8);
    }

    private SpannableStringBuilder segmentSpans(int startIndex, int endIndex) {
        SpannableStringBuilder builder = new SpannableStringBuilder();
        for (int i = startIndex; i <= endIndex; i++) {
            EditorSegment segment = editorSegments.get(i);
            appendSeekSpan(builder, segment.text, segment.start, i, i == activeSegmentIndex);
            builder.append(' ');
        }
        return builder;
    }

    private void appendSeekSpan(SpannableStringBuilder builder, String value, double seconds, int segmentIndex, boolean active) {
        int start = builder.length();
        builder.append(value == null ? "" : value);
        int end = builder.length();
        builder.setSpan(new ClickableSpan() {
            @Override
            public void onClick(View widget) {
                activeSegmentIndex = segmentIndex;
                renderEditorTranscript();
                scrollActiveSegmentIntoView();
                seekTo(seconds);
            }

            @Override
            public void updateDrawState(TextPaint ds) {
                super.updateDrawState(ds);
                ds.setColor(TEXT);
                ds.setUnderlineText(false);
            }
        }, start, end, Spanned.SPAN_EXCLUSIVE_EXCLUSIVE);
        int bg = active ? Color.argb(120, 0, 123, 255) : Color.TRANSPARENT;
        builder.setSpan(new android.text.style.BackgroundColorSpan(bg), start, end, Spanned.SPAN_EXCLUSIVE_EXCLUSIVE);
    }

    private void showSegmentActions(int segmentIndex) {
        if (segmentIndex < 0 || segmentIndex >= editorSegments.size()) return;
        String[] actions = new String[]{
                "Tách/gán người nói từ chữ này",
                "Gộp từ chữ này lên trên",
                "Gộp từ chữ này xuống dưới",
                "Đổi tên/màu người nói",
                "Sao chép đoạn này"
        };
        LinearLayout layout = vertical();
        layout.setPadding(dp(8), dp(8), dp(8), dp(8));
        layout.setBackground(rounded(CARD, BORDER, 1, 8));
        AlertDialog actionDialog = new AlertDialog.Builder(this).create();
        for (int i = 0; i < actions.length; i++) {
            final int which = i;
            TextView row = text(actions[i], 18, false, TEXT);
            row.setPadding(dp(14), 0, dp(14), 0);
            row.setOnClickListener(v -> {
                actionDialog.dismiss();
                if (which == 0) showSplitDialog(segmentIndex);
                else if (which == 1) mergeSpeakerBlock(segmentIndex, true);
                else if (which == 2) mergeSpeakerBlock(segmentIndex, false);
                else if (which == 3) showRenameDialog(editorSegments.get(segmentIndex).speaker, true, segmentIndex);
                else copySegment(segmentIndex);
            });
            layout.addView(row, new LinearLayout.LayoutParams(
                    LinearLayout.LayoutParams.MATCH_PARENT,
                    dp(52)));
        }
        actionDialog.setView(layout);
        actionDialog.show();
        Window window = actionDialog.getWindow();
        if (window != null) {
            window.setBackgroundDrawable(new ColorDrawable(Color.TRANSPARENT));
        }
    }

    private void showSplitDialog(int segmentIndex) {
        LinearLayout layout = vertical();
        layout.setPadding(dp(18), dp(8), dp(18), 0);
        TextView speakerLabel = text("Chọn người nói:", 15, true, Color.BLACK);
        layout.addView(speakerLabel);
        Spinner speakerSelect = new Spinner(this);
        List<Integer> ids = new ArrayList<>(editorSpeakers.keySet());
        List<String> labels = new ArrayList<>();
        labels.add("-- Người nói mới --");
        for (Integer id : ids) labels.add(speakerMeta(id).name);
        speakerSelect.setAdapter(new ArrayAdapter<>(this, android.R.layout.simple_spinner_dropdown_item, labels));
        layout.addView(speakerSelect);

        TextView nameLabel = text("Tên mới:", 15, true, Color.BLACK);
        nameLabel.setPadding(0, dp(10), 0, 0);
        layout.addView(nameLabel);
        EditText name = new EditText(this);
        name.setHint("Nhập tên người nói mới...");
        name.setSingleLine(true);
        name.setTextColor(Color.BLACK);
        name.setTextSize(TypedValue.COMPLEX_UNIT_PX, 18 * uiScale());
        layout.addView(name);

        TextView colorLabel = text("Màu:", 15, true, Color.BLACK);
        colorLabel.setPadding(0, dp(10), 0, 0);
        layout.addView(colorLabel);
        final int[] selectedColor = new int[]{SPEAKER_COLORS[Math.abs(nextSpeakerId()) % SPEAKER_COLORS.length]};
        LinearLayout colorDots = new LinearLayout(this);
        layout.addView(colorDots);
        renderColorDots(colorDots, selectedColor);

        speakerSelect.setOnItemSelectedListener(new android.widget.AdapterView.OnItemSelectedListener() {
            @Override
            public void onItemSelected(android.widget.AdapterView<?> parent, View view, int position, long id) {
                if (position > 0) {
                    selectedColor[0] = speakerMeta(ids.get(position - 1)).color;
                } else {
                    selectedColor[0] = SPEAKER_COLORS[Math.abs(nextSpeakerId()) % SPEAKER_COLORS.length];
                }
                renderColorDots(colorDots, selectedColor);
            }

            @Override
            public void onNothingSelected(android.widget.AdapterView<?> parent) {
            }
        });

        RadioGroup scope = new RadioGroup(this);
        scope.setOrientation(RadioGroup.VERTICAL);
        RadioButton toEnd = new RadioButton(this);
        toEnd.setId(View.generateViewId());
        toEnd.setText("Từ đoạn này đến hết block hiện tại");
        toEnd.setTextColor(Color.BLACK);
        toEnd.setTextSize(TypedValue.COMPLEX_UNIT_PX, 17 * uiScale());
        RadioButton single = new RadioButton(this);
        single.setId(View.generateViewId());
        single.setText("Chỉ đoạn này");
        single.setTextColor(Color.BLACK);
        single.setTextSize(TypedValue.COMPLEX_UNIT_PX, 17 * uiScale());
        scope.addView(toEnd);
        scope.addView(single);
        scope.check(toEnd.getId());
        layout.addView(scope);

        TextView title = text("Tách/gán người nói", 22, true, Color.BLACK);
        title.setPadding(dp(22), dp(18), dp(22), 0);
        AlertDialog splitDialog = new AlertDialog.Builder(this)
                .setCustomTitle(title)
                .setView(layout)
                .setPositiveButton("Áp dụng", (dlg, which) -> {
                    int selected = speakerSelect.getSelectedItemPosition();
                    int targetSpeaker;
                    if (selected > 0) {
                        targetSpeaker = ids.get(selected - 1);
                    } else {
                        targetSpeaker = nextSpeakerId();
                        String value = name.getText().toString().trim();
                        editorSpeakers.put(targetSpeaker, new SpeakerMeta(
                                value.isEmpty() ? defaultSpeakerName(targetSpeaker) : value,
                                selectedColor[0]));
                    }
                    speakerMeta(targetSpeaker).color = selectedColor[0];
                    int end = scope.getCheckedRadioButtonId() == toEnd.getId() ? findBlockEnd(segmentIndex) : segmentIndex;
                    for (int i = segmentIndex; i <= end; i++) editorSegments.get(i).speaker = targetSpeaker;
                    renderEditorTranscript();
                    saveEditedResult();
                })
                .setNegativeButton("Hủy", null)
                .create();
        splitDialog.setOnShowListener(d -> {
            name.clearFocus();
            Window window = splitDialog.getWindow();
            if (window != null) window.setSoftInputMode(WindowManager.LayoutParams.SOFT_INPUT_STATE_ALWAYS_HIDDEN);
        });
        splitDialog.show();
    }

    private void showRenameDialog(int speakerId, boolean allowSingleBlock, int segmentIndex) {
        LinearLayout layout = vertical();
        layout.setPadding(dp(18), dp(8), dp(18), 0);

        TextView selectLabel = text("Chọn người nói có sẵn:", 15, true, Color.BLACK);
        layout.addView(selectLabel);
        Spinner existingSelect = new Spinner(this);
        List<Integer> ids = new ArrayList<>(editorSpeakers.keySet());
        List<String> labels = new ArrayList<>();
        labels.add("-- Nhập tên / giữ hiện tại --");
        int currentPosition = 0;
        for (int i = 0; i < ids.size(); i++) {
            Integer id = ids.get(i);
            labels.add(speakerMeta(id).name);
            if (id == speakerId) currentPosition = i + 1;
        }
        existingSelect.setAdapter(new ArrayAdapter<>(this, android.R.layout.simple_spinner_dropdown_item, labels));
        existingSelect.setSelection(currentPosition);
        layout.addView(existingSelect);

        TextView nameLabel = text("Tên:", 15, true, Color.BLACK);
        nameLabel.setPadding(0, dp(10), 0, 0);
        layout.addView(nameLabel);
        EditText input = new EditText(this);
        input.setSingleLine(true);
        input.setText(speakerMeta(speakerId).name);
        input.setSelectAllOnFocus(true);
        input.setTextColor(Color.BLACK);
        input.setTextSize(TypedValue.COMPLEX_UNIT_PX, 18 * uiScale());
        layout.addView(input);

        TextView colorLabel = text("Màu:", 15, true, Color.BLACK);
        colorLabel.setPadding(0, dp(10), 0, 0);
        layout.addView(colorLabel);
        final int[] selectedColor = new int[]{speakerMeta(speakerId).color};
        LinearLayout colorDots = new LinearLayout(this);
        layout.addView(colorDots);
        renderColorDots(colorDots, selectedColor);

        existingSelect.setOnItemSelectedListener(new android.widget.AdapterView.OnItemSelectedListener() {
            @Override
            public void onItemSelected(android.widget.AdapterView<?> parent, View view, int position, long id) {
                if (position <= 0) return;
                SpeakerMeta meta = speakerMeta(ids.get(position - 1));
                input.setText(meta.name);
                input.setSelection(input.getText().length());
                selectedColor[0] = meta.color;
                renderColorDots(colorDots, selectedColor);
            }

            @Override
            public void onNothingSelected(android.widget.AdapterView<?> parent) {
            }
        });

        TextView title = text("Đổi tên người nói", 22, true, Color.BLACK);
        title.setPadding(dp(22), dp(18), dp(22), 0);
        AlertDialog.Builder builder = new AlertDialog.Builder(this)
                .setCustomTitle(title)
                .setView(layout)
                .setPositiveButton("Sửa tất cả", (dlg, which) -> {
                    String value = input.getText().toString().trim();
                    if (!value.isEmpty()) speakerMeta(speakerId).name = value;
                    speakerMeta(speakerId).color = selectedColor[0];
                    renderEditorTranscript();
                    saveEditedResult();
                })
                .setNegativeButton("Hủy", null);
        if (allowSingleBlock) {
            builder.setNeutralButton("Áp dụng block này", (dlg, which) -> {
                int targetId;
                int selected = existingSelect.getSelectedItemPosition();
                if (selected > 0) {
                    targetId = ids.get(selected - 1);
                    String value = input.getText().toString().trim();
                    if (!value.isEmpty()) speakerMeta(targetId).name = value;
                    speakerMeta(targetId).color = selectedColor[0];
                } else {
                    targetId = nextSpeakerId();
                    String value = input.getText().toString().trim();
                    editorSpeakers.put(targetId, new SpeakerMeta(
                            value.isEmpty() ? defaultSpeakerName(targetId) : value,
                            selectedColor[0]));
                }
                int start = findBlockStart(segmentIndex);
                int end = findBlockEnd(segmentIndex);
                for (int i = start; i <= end; i++) editorSegments.get(i).speaker = targetId;
                renderEditorTranscript();
                saveEditedResult();
            });
        }
        AlertDialog renameDialog = builder.create();
        renameDialog.setOnShowListener(d -> {
            input.clearFocus();
            Window window = renameDialog.getWindow();
            if (window != null) window.setSoftInputMode(WindowManager.LayoutParams.SOFT_INPUT_STATE_ALWAYS_HIDDEN);
        });
        renameDialog.show();
    }

    private int nextSpeakerId() {
        int max = -1;
        for (Integer id : editorSpeakers.keySet()) max = Math.max(max, id);
        return max + 1;
    }

    private int findBlockStart(int index) {
        int speaker = editorSegments.get(index).speaker;
        while (index > 0 && editorSegments.get(index - 1).speaker == speaker) index--;
        return index;
    }

    private int findBlockEnd(int index) {
        int speaker = editorSegments.get(index).speaker;
        while (index + 1 < editorSegments.size() && editorSegments.get(index + 1).speaker == speaker) index++;
        return index;
    }

    private void mergeSpeakerBlock(int index, boolean previous) {
        if (index < 0 || index >= editorSegments.size()) return;
        int start = findBlockStart(index);
        int end = findBlockEnd(index);
        int neighbor = previous ? start - 1 : end + 1;
        if (neighbor < 0 || neighbor >= editorSegments.size()) return;
        int targetSpeaker = editorSegments.get(neighbor).speaker;
        int from = previous ? start : index;
        int to = previous ? index : end;
        for (int i = from; i <= to; i++) editorSegments.get(i).speaker = targetSpeaker;
        renderEditorTranscript();
        saveEditedResult();
    }

    private void copySegment(int index) {
        if (index < 0 || index >= editorSegments.size()) return;
        ClipboardManager clipboard = (ClipboardManager) getSystemService(CLIPBOARD_SERVICE);
        if (clipboard != null) {
            clipboard.setPrimaryClip(ClipData.newPlainText("ASR segment", editorSegments.get(index).text));
        }
    }

    private String transcriptText() {
        StringBuilder text = new StringBuilder();
        for (EditorSegment segment : editorSegments) {
            if (text.length() > 0) text.append(' ');
            text.append(segment.text);
        }
        return text.toString().trim();
    }

    private void saveEditedResult() {
        if (selectedLibraryItemId == null || editorSegments.isEmpty()) return;
        try {
            library.saveResult(selectedLibraryItemId, editedResultJson());
            refreshLibraryList();
        } catch (Exception error) {
            appendLog("Save edited result failed: " + error.getMessage());
        }
    }

    private String editedResultJson() throws Exception {
        JSONObject root = new JSONObject();
        root.put("schema", "asr-vn-native-0.2");
        root.put("duration_sec", currentDurationSeconds);
        root.put("text", transcriptText());
        JSONObject names = new JSONObject();
        JSONObject colors = new JSONObject();
        JSONArray segments = new JSONArray();
        int previousSpeaker = Integer.MIN_VALUE;
        for (EditorSegment segment : editorSegments) {
            if (segment.speaker != previousSpeaker) {
                JSONObject speaker = new JSONObject();
                speaker.put("type", "speaker");
                speaker.put("speaker_id", segment.speaker);
                speaker.put("speaker", speakerMeta(segment.speaker).name);
                segments.put(speaker);
                previousSpeaker = segment.speaker;
            }
            JSONObject item = new JSONObject();
            item.put("type", "text");
            item.put("speaker_id", segment.speaker);
            item.put("start_time", segment.start);
            item.put("end_time", segment.end);
            item.put("text", segment.text);
            if (segment.words != null && !segment.words.isEmpty()) {
                JSONArray rawWords = new JSONArray();
                for (TimedWord word : segment.words) {
                    JSONObject rawWord = new JSONObject();
                    rawWord.put("text", word.text);
                    rawWord.put("start", word.start);
                    rawWord.put("end", word.end);
                    rawWords.put(rawWord);
                }
                item.put("raw_words", rawWords);
            }
            segments.put(item);
            names.put(String.valueOf(segment.speaker), speakerMeta(segment.speaker).name);
            colors.put(String.valueOf(segment.speaker), colorHex(speakerMeta(segment.speaker).color));
        }
        root.put("speaker_names", names);
        root.put("speaker_colors", colors);
        root.put("segments", segments);
        return root.toString();
    }

    private void copyTranscript() {
        if (currentTranscriptText == null || currentTranscriptText.trim().isEmpty()) {
            appendLog("Chưa có văn bản để sao chép.");
            return;
        }
        ClipboardManager clipboard = (ClipboardManager) getSystemService(CLIPBOARD_SERVICE);
        if (clipboard != null) {
            clipboard.setPrimaryClip(ClipData.newPlainText("ASR transcript", currentTranscriptText));
            appendLog("Đã sao chép văn bản.");
        }
    }

    private void setProgress(String phase, int percent) {
        if (progressContainer != null) {
            progressContainer.setVisibility(percent >= 100 ? View.GONE : View.VISIBLE);
        }
        progressBar.setProgress(Math.max(0, Math.min(100, percent)));
        stageView.setText(phase);
        progressText.setText(Math.max(0, Math.min(100, percent)) + "%");
    }

    private void appendLog(String line) {
        String value = new SimpleDateFormat("HH:mm:ss", Locale.US).format(new Date()) + "  " + line;
        Log.i(TAG, value);
        if (logView != null) logView.append(value + "\n");
    }
}

