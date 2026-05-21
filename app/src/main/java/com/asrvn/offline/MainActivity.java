package com.asrvn.offline;

import android.app.Activity;
import android.app.AlertDialog;
import android.Manifest;
import android.content.BroadcastReceiver;
import android.content.ClipData;
import android.content.ClipboardManager;
import android.content.Context;
import android.content.Intent;
import android.content.IntentFilter;
import android.content.pm.PackageManager;
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
import android.text.Layout;
import android.text.SpannableStringBuilder;
import android.text.Spanned;
import android.text.Editable;
import android.text.InputType;
import android.text.TextPaint;
import android.text.TextWatcher;
import android.text.method.LinkMovementMethod;
import android.text.style.ClickableSpan;
import android.text.style.ForegroundColorSpan;
import android.util.TypedValue;
import android.util.Log;
import android.view.Gravity;
import android.view.MotionEvent;
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
import java.io.FileOutputStream;
import java.io.InputStream;
import java.io.OutputStream;
import java.text.SimpleDateFormat;
import java.nio.charset.StandardCharsets;
import java.util.Date;
import java.util.ArrayList;
import java.util.HashSet;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Set;

public final class MainActivity extends Activity {
    private static final String TAG = "ASRVN";
    private static final int REQUEST_OPEN_MEDIA = 1001;
    private static final int REQUEST_IMPORT_HOTWORD_TXT = 1002;
    private static final int REQUEST_EXPORT_HOTWORD_TXT = 1003;
    private static final int REQUEST_POST_NOTIFICATIONS = 1004;

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
    private static final int HIGHLIGHT_BG = Color.argb(115, 255, 193, 7);
    private static final int HIGHLIGHT_TEXT = Color.rgb(31, 41, 55);
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

    private static final class SegmentSplit {
        final int beforeIndex;
        final int afterIndex;

        SegmentSplit(int beforeIndex, int afterIndex) {
            this.beforeIndex = beforeIndex;
            this.afterIndex = afterIndex;
        }
    }

    private final class WordSeekSpan extends ClickableSpan {
        final double seconds;
        final int segmentIndex;
        final int wordIndex;
        final boolean active;

        WordSeekSpan(double seconds, int segmentIndex, int wordIndex, boolean active) {
            this.seconds = seconds;
            this.segmentIndex = segmentIndex;
            this.wordIndex = wordIndex;
            this.active = active;
        }

        @Override
        public void onClick(View widget) {
            selectTranscriptTarget(segmentIndex, wordIndex);
            renderEditorTranscript();
            seekTo(segmentStartSeconds(segmentIndex));
        }

        @Override
        public void updateDrawState(TextPaint ds) {
            super.updateDrawState(ds);
            ds.setColor(active ? HIGHLIGHT_TEXT : TEXT);
            ds.setUnderlineText(false);
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
    private View hotwordPanel;
    private View hotwordScrim;
    private FrameLayout contentScrollbar;
    private View contentScrollThumb;
    private LinearLayout libraryList;
    private CheckBox librarySelectAll;
    private EditText librarySearch;
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
    private LinearLayout qualityStripView;
    private LinearLayout audioSummaryView;
    private LinearLayout resultTimingView;
    private TextView playerTimeView;
    private Button playerPlayPauseButton;
    private SeekBar playerSeekBar;
    private TextView stageView;
    private TextView progressText;
    private ProgressBar progressBar;
    private View progressContainer;
    private Button cancelProcessingButton;
    private View resultPanelView;
    private View fixedPlayerView;
    private Spinner speakerModelSpinner;
    private MediaPlayer mediaPlayer;
    private boolean userSeeking;
    private boolean configExpanded = true;
    private int activeSegmentIndex = -1;
    private int activeWordIndex = -1;
    private long lastActiveAutoScrollAtMs;
    private Uri selectedUri;
    private File selectedStagedInputFile;
    private String selectedLibraryItemId;
    private String selectedDisplayName;
    private long selectedFileSizeBytes;
    private String pendingDebugSpeakerModel;
    private String activeProcessingItemId;
    private String librarySearchQuery = "";
    private String currentTranscriptText = "";
    private String currentQualityInfoJson;
    private String currentTimingJson;
    private String currentAudioSummaryJson;
    private double currentDurationSeconds;
    private final List<EditorSegment> editorSegments = new ArrayList<>();
    private final Map<Integer, SpeakerMeta> editorSpeakers = new LinkedHashMap<>();
    private final Set<String> selectedLibraryItemIds = new HashSet<>();
    private final Handler uiHandler = new Handler(Looper.getMainLooper());

    private NativeOfflinePipeline pipeline;
    private NativeSettings settings;
    private ModelFileRegistry models;
    private NativeFileLibrary library;
    private BroadcastReceiver processingReceiver;

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
        registerProcessingReceiver();
        requestNotificationPermissionForProgress();
        rootFrame.postDelayed(this::showInitialModelDownloadPromptIfNeeded, 700);
        handleDebugIntent(getIntent());
        handleIncomingFileIntent(getIntent());
    }

    @Override
    protected void onNewIntent(Intent intent) {
        super.onNewIntent(intent);
        setIntent(intent);
        handleDebugIntent(intent);
        handleIncomingFileIntent(intent);
    }

    @Override
    protected void onDestroy() {
        uiHandler.removeCallbacksAndMessages(null);
        if (processingReceiver != null) {
            try {
                unregisterReceiver(processingReceiver);
            } catch (Exception ignored) {
            }
            processingReceiver = null;
        }
        releaseMediaPlayer();
        super.onDestroy();
    }

    private void registerProcessingReceiver() {
        processingReceiver = new BroadcastReceiver() {
            @Override
            public void onReceive(Context context, Intent intent) {
                handleProcessingBroadcast(intent);
            }
        };
        IntentFilter filter = new IntentFilter();
        filter.addAction(ProcessingForegroundService.ACTION_PROGRESS);
        filter.addAction(ProcessingForegroundService.ACTION_COMPLETE);
        filter.addAction(ProcessingForegroundService.ACTION_ERROR);
        filter.addAction(ProcessingForegroundService.ACTION_CANCELLED);
        if (Build.VERSION.SDK_INT >= 33) {
            registerReceiver(processingReceiver, filter, Context.RECEIVER_NOT_EXPORTED);
        } else {
            registerReceiver(processingReceiver, filter);
        }
    }

    private void requestNotificationPermissionForProgress() {
        if (Build.VERSION.SDK_INT < 33) return;
        if (checkSelfPermission(Manifest.permission.POST_NOTIFICATIONS) == PackageManager.PERMISSION_GRANTED) return;
        requestPermissions(new String[]{Manifest.permission.POST_NOTIFICATIONS}, REQUEST_POST_NOTIFICATIONS);
    }

    private void handleProcessingBroadcast(Intent intent) {
        if (intent == null) return;
        String action = intent.getAction();
        if (ProcessingForegroundService.ACTION_PROGRESS.equals(action)) {
            String phase = intent.getStringExtra(ProcessingForegroundService.EXTRA_PHASE);
            String message = intent.getStringExtra(ProcessingForegroundService.EXTRA_MESSAGE);
            String itemId = intent.getStringExtra(ProcessingForegroundService.EXTRA_ITEM_ID);
            if (itemId != null && !itemId.isEmpty()) activeProcessingItemId = itemId;
            int percent = intent.getIntExtra(ProcessingForegroundService.EXTRA_PERCENT, 0);
            setProgress(phase == null ? "Đang xử lý" : phase, percent);
            appendLog((phase == null ? "Đang xử lý" : phase) + ": " + (message == null ? "" : message));
            if (percent >= 100) refreshLibraryList();
            return;
        }
        if (ProcessingForegroundService.ACTION_COMPLETE.equals(action)) {
            String itemId = intent.getStringExtra(ProcessingForegroundService.EXTRA_ITEM_ID);
            String resultJson = intent.getStringExtra(ProcessingForegroundService.EXTRA_RESULT_JSON);
            try {
                NativeFileLibrary.LibraryItem item = itemId == null ? null : library.getItem(itemId);
                discardStagedInputFile();
                if (item != null) {
                    selectedLibraryItemId = item.id;
                    selectedUri = Uri.fromFile(item.sourceFile);
                    updateSelectedFileUi(item.originalName, item.sourceFile.length());
                    prepareMediaPlayer(selectedUri);
                }
                if (resultJson != null) {
                    renderTranscriptFromJson(resultJson);
                    if (resultPanelView != null) resultPanelView.setVisibility(View.VISIBLE);
                }
                setProgress("Done", 100);
                activeProcessingItemId = null;
                refreshLibraryList();
            } catch (Exception error) {
                appendLog("Render background result failed: " + error.getMessage());
            }
            return;
        }
        if (ProcessingForegroundService.ACTION_CANCELLED.equals(action)) {
            activeProcessingItemId = null;
            setProgress("Đã hủy", 100);
            appendLog("Đã hủy xử lý.");
            refreshLibraryList();
            return;
        }
        if (ProcessingForegroundService.ACTION_ERROR.equals(action)) {
            String message = intent.getStringExtra(ProcessingForegroundService.EXTRA_MESSAGE);
            activeProcessingItemId = null;
            setProgress("Pipeline failed", 100);
            appendLog("Error: " + (message == null ? "Unknown error" : message));
            refreshLibraryList();
        }
    }

    private View buildContentView() {
        rootFrame = new FrameLayout(this);
        rootFrame.setBackgroundColor(BG);

        LinearLayout app = vertical();
        app.setBackgroundColor(BG);
        app.setPadding(0, statusBarHeight(), 0, 0);
        app.addView(topBar(), new LinearLayout.LayoutParams(
                LinearLayout.LayoutParams.MATCH_PARENT,
                topBarHeight()));
        app.addView(mainScroll(), new LinearLayout.LayoutParams(
                LinearLayout.LayoutParams.MATCH_PARENT, 0, 1f));
        rootFrame.addView(app);

        fixedPlayerView = fixedPlayerBar();
        fixedPlayerView.setVisibility(View.GONE);
        rootFrame.addView(fixedPlayerView, new FrameLayout.LayoutParams(
                FrameLayout.LayoutParams.MATCH_PARENT,
                dp(62),
                Gravity.BOTTOM));

        contentScrollbar = contentFastScrollbar();
        FrameLayout.LayoutParams scrollbarParams = new FrameLayout.LayoutParams(
                dp(30),
                FrameLayout.LayoutParams.MATCH_PARENT,
                Gravity.RIGHT);
        scrollbarParams.topMargin = statusBarHeight() + topBarHeight() + dp(8);
        scrollbarParams.bottomMargin = dp(74);
        rootFrame.addView(contentScrollbar, scrollbarParams);

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

        hotwordScrim = new View(this);
        hotwordScrim.setBackgroundColor(Color.argb(80, 0, 0, 0));
        hotwordScrim.setVisibility(View.GONE);
        hotwordScrim.setOnClickListener(v -> closeHotwordPanel(false));
        rootFrame.addView(hotwordScrim, new FrameLayout.LayoutParams(
                FrameLayout.LayoutParams.MATCH_PARENT,
                FrameLayout.LayoutParams.MATCH_PARENT));
        refreshLibraryList();
        return rootFrame;
    }

    private View topBar() {
        int height = topBarHeight();
        LinearLayout bar = new LinearLayout(this);
        bar.setOrientation(LinearLayout.HORIZONTAL);
        bar.setGravity(Gravity.CENTER_VERTICAL);
        bar.setPadding(dp(16), 0, dp(16), 0);
        bar.setBackgroundColor(CARD);

        TextView title = text("Sherpa Vietnamese ASR", 19, true, ACCENT);
        title.setLineSpacing(0, 1.0f);
        title.setSingleLine(false);
        title.setMaxLines(2);
        title.setGravity(Gravity.LEFT | Gravity.CENTER_VERTICAL);
        bar.addView(title, new LinearLayout.LayoutParams(0, LinearLayout.LayoutParams.MATCH_PARENT, 1f));

        TextView info = text("i", 18, true, Color.rgb(205, 225, 255));
        info.setGravity(Gravity.CENTER);
        GradientDrawable infoBg = new GradientDrawable();
        infoBg.setShape(GradientDrawable.OVAL);
        infoBg.setColor(Color.TRANSPARENT);
        infoBg.setStroke(dp(2), Color.rgb(170, 205, 255));
        info.setBackground(infoBg);
        info.setOnClickListener(v -> showAboutDialog());
        int iconSize = Math.max(dp(38), Math.min(dp(52), scaledTextHeight(18, 1, 38)));
        LinearLayout.LayoutParams infoParams = new LinearLayout.LayoutParams(iconSize, iconSize);
        infoParams.leftMargin = dp(4);
        infoParams.rightMargin = dp(6);
        bar.addView(info, infoParams);

        Button files = button("Quản lý tập tin", ELEVATED, v -> openDrawer());
        files.setTextSize(TypedValue.COMPLEX_UNIT_PX, 17 * uiScale());
        files.setGravity(Gravity.CENTER);
        files.setSingleLine(false);
        files.setMaxLines(2);
        files.setLineSpacing(0, 1.0f);
        LinearLayout.LayoutParams params = new LinearLayout.LayoutParams(dp(180), Math.max(dp(52), height - dp(10)));
        params.leftMargin = dp(8);
        bar.addView(files, params);
        return bar;
    }

    private void showAboutDialog() {
        LinearLayout body = vertical();
        body.setPadding(dp(18), dp(16), dp(18), dp(14));
        body.setBackground(rounded(CARD, BORDER, 1, 8));

        body.addView(text("Thông tin", 24, true, TEXT));
        ScrollView scroll = new ScrollView(this);
        LinearLayout content = vertical();
        content.setPadding(0, dp(10), 0, dp(4));

        TextView appName = text("sherpa-vietnamese-asr", 20, true, ACCENT);
        content.addView(appName);
        TextView version = text("Phiên bản " + appVersionName(), 16, false, MUTED);
        version.setPadding(0, dp(3), 0, dp(12));
        content.addView(version);

        addAboutInfoBlock(content, "Thiết kế", new String[]{
                "Nguyễn Hồng Quân",
                "nhquan.thanhuy@tphcm.gov.vn — 098.558.3555",
                "Phòng Chuyển đổi số - Cơ yếu, VP Thành ủy TP.HCM"
        });
        addAboutInfoBlock(content, "Lập trình", new String[]{
                "Claude và những người bạn"
        });

        TextView license = text(
                "Phần mềm sử dụng trong môi trường giáo dục, hành chính công, tổ chức Đảng, đoàn thể. Không sử dụng cho mục đích thương mại.",
                16,
                false,
                Color.rgb(255, 215, 0));
        license.setGravity(Gravity.CENTER);
        license.setLineSpacing(dp(2), 1.0f);
        license.setPadding(dp(14), dp(12), dp(14), dp(12));
        license.setBackground(rounded(Color.argb(20, 255, 215, 0), Color.argb(52, 255, 215, 0), 1, 6));
        content.addView(withMargins(license, 0, 2, 0, 12));

        addAboutDetailsBlock(content, "Chức năng", new String[]{
                "• Chuyển ghi âm thành văn bản tiếng Việt (offline)",
                "• 3 model ASR: Zipformer 30M, 68M, ROVER",
                "• Phân tách người nói: Pyannote Community-1, Senko CAM++",
                "• NaturalTurn: nhận diện lượt nói tự nhiên",
                "• Tự động thêm dấu câu, viết hoa",
                "• Tóm tắt cuộc họp (Gemma 4 E2B)",
                "• Hỗ trợ hotwords (từ khóa tùy chỉnh)",
                "• Đánh giá chất lượng âm thanh (DNSMOS)",
                "• PWA — cài trên mobile/desktop như app native"
        });
        addAboutDetailsBlock(content, "Công nghệ", new String[]{
                "• ASR: Sherpa-ONNX, Zipformer RNN-T (30M + 68M)",
                "• Diarization: Pyannote Community-1 + Senko CAM++ (Pure ONNX Runtime)",
                "• Dấu câu: ViBERT-capu (ONNX)",
                "• VAD: Pyannote Segmentation (ONNX)",
                "• Summarizer: Gemma 4 E2B (GGUF, llama-cpp-python)",
                "• Resampling: SoXR VHQ",
                "• Web: FastAPI, WebSocket, SQLite"
        });
        scroll.addView(content);
        body.addView(scroll, new LinearLayout.LayoutParams(
                LinearLayout.LayoutParams.MATCH_PARENT,
                dp(520)));

        final AlertDialog[] holder = new AlertDialog[1];
        Button closeButton = button("Đóng", INPUT, v -> {
            if (holder[0] != null) holder[0].dismiss();
        });
        LinearLayout.LayoutParams closeParams = new LinearLayout.LayoutParams(
                LinearLayout.LayoutParams.MATCH_PARENT,
                dp(46));
        closeParams.topMargin = dp(16);
        body.addView(closeButton, closeParams);

        AlertDialog dialog = new AlertDialog.Builder(this)
                .setView(body)
                .create();
        holder[0] = dialog;
        dialog.show();
        Window window = dialog.getWindow();
        if (window != null) {
            window.setBackgroundDrawable(new ColorDrawable(Color.TRANSPARENT));
        }
    }

    private void addAboutInfoBlock(LinearLayout parent, String label, String[] lines) {
        LinearLayout block = vertical();
        block.setPadding(0, dp(3), 0, dp(8));
        TextView title = text(label.toUpperCase(Locale.ROOT), 12, true, MUTED);
        title.setPadding(0, 0, 0, dp(3));
        block.addView(title);
        for (int i = 0; i < lines.length; i++) {
            TextView item = text(lines[i], i == 0 ? 15 : 13, false, i == 0 ? TEXT : MUTED);
            item.setLineSpacing(dp(1), 1.0f);
            item.setPadding(0, dp(1), 0, dp(1));
            block.addView(item);
        }
        parent.addView(block);
    }

    private void addAboutDetailsBlock(LinearLayout parent, String label, String[] lines) {
        LinearLayout block = vertical();
        block.setBackground(rounded(CARD, BORDER, 1, 6));

        TextView summary = text("▶  " + label, 15, true, SUCCESS);
        summary.setPadding(dp(12), 0, dp(12), 0);
        summary.setBackground(rounded(Color.argb(8, 255, 255, 255), BORDER, 0, 6));
        block.addView(summary, new LinearLayout.LayoutParams(
                LinearLayout.LayoutParams.MATCH_PARENT,
                dp(42)));

        LinearLayout content = vertical();
        content.setPadding(dp(18), dp(4), dp(12), dp(10));
        content.setVisibility(View.GONE);
        for (String line : lines) {
            TextView item = text(line, 14, false, MUTED);
            item.setSingleLine(false);
            item.setLineSpacing(dp(2), 1.0f);
            item.setPadding(0, dp(2), 0, dp(2));
            content.addView(item);
        }
        block.addView(content);

        summary.setOnClickListener(v -> {
            boolean opening = content.getVisibility() != View.VISIBLE;
            content.setVisibility(opening ? View.VISIBLE : View.GONE);
            summary.setText((opening ? "▼  " : "▶  ") + label);
        });
        parent.addView(withMargins(block, 0, 0, 0, 6));
    }

    private String appVersionName() {
        try {
            if (Build.VERSION.SDK_INT >= 33) {
                return getPackageManager()
                        .getPackageInfo(getPackageName(), android.content.pm.PackageManager.PackageInfoFlags.of(0))
                        .versionName;
            }
            return getPackageManager().getPackageInfo(getPackageName(), 0).versionName;
        } catch (Exception ignored) {
            return "0.1.0";
        }
    }

    private View mainScroll() {
        ScrollView scroll = new ScrollView(this);
        mainScrollView = scroll;
        scroll.setFillViewport(false);
        scroll.setClipToPadding(false);
        scroll.setPadding(0, 0, 0, dp(72));
        scroll.setVerticalScrollBarEnabled(true);
        scroll.setScrollbarFadingEnabled(false);
        scroll.setScrollBarStyle(View.SCROLLBARS_INSIDE_INSET);
        if (Build.VERSION.SDK_INT >= 23) {
            scroll.setOnScrollChangeListener((v, scrollX, scrollY, oldScrollX, oldScrollY) -> updateContentScrollbar());
        }
        scroll.setDescendantFocusability(ViewGroup.FOCUS_BEFORE_DESCENDANTS);
        LinearLayout main = vertical();
        main.setPadding(dp(14), dp(12), dp(14), dp(14));
        main.addView(configPanel());
        main.addView(filePanel());
        main.addView(resultPanel());
        scroll.addView(main);
        return scroll;
    }

    private FrameLayout contentFastScrollbar() {
        FrameLayout track = new FrameLayout(this);
        track.setVisibility(View.GONE);
        track.setPadding(dp(10), 0, dp(8), 0);
        track.setOnTouchListener((v, event) -> {
            int action = event.getActionMasked();
            if (action == MotionEvent.ACTION_DOWN || action == MotionEvent.ACTION_MOVE) {
                scrollContentFromScrollbar(event.getY());
                return true;
            }
            return action == MotionEvent.ACTION_UP || action == MotionEvent.ACTION_CANCEL;
        });

        contentScrollThumb = new View(this);
        contentScrollThumb.setBackground(rounded(Color.rgb(135, 135, 135), Color.rgb(160, 160, 160), 1, 4));
        FrameLayout.LayoutParams thumbParams = new FrameLayout.LayoutParams(
                dp(8),
                dp(80),
                Gravity.TOP | Gravity.CENTER_HORIZONTAL);
        track.addView(contentScrollThumb, thumbParams);
        return track;
    }

    private void updateContentScrollbar() {
        if (contentScrollbar == null || contentScrollThumb == null || mainScrollView == null) return;
        if (drawer != null && drawer.getVisibility() == View.VISIBLE) {
            contentScrollbar.setVisibility(View.GONE);
            return;
        }
        View child = mainScrollView.getChildAt(0);
        if (child == null) {
            contentScrollbar.setVisibility(View.GONE);
            return;
        }
        int maxScroll = maxMainScrollY(child);
        int trackHeight = contentScrollbar.getHeight();
        if (maxScroll <= dp(24) || trackHeight <= 0) {
            contentScrollbar.setVisibility(View.GONE);
            return;
        }
        int contentHeight = Math.max(mainScrollView.getHeight(), child.getHeight() + mainScrollView.getPaddingBottom());
        int thumbHeight = Math.max(dp(76), Math.round(trackHeight * (mainScrollView.getHeight() / (float) contentHeight)));
        thumbHeight = Math.min(trackHeight, thumbHeight);
        int top = Math.round((trackHeight - thumbHeight) * (mainScrollView.getScrollY() / (float) maxScroll));
        FrameLayout.LayoutParams params = (FrameLayout.LayoutParams) contentScrollThumb.getLayoutParams();
        params.height = thumbHeight;
        params.topMargin = Math.max(0, Math.min(trackHeight - thumbHeight, top));
        contentScrollThumb.setLayoutParams(params);
        contentScrollbar.setVisibility(View.VISIBLE);
    }

    private void scrollContentFromScrollbar(float y) {
        if (contentScrollbar == null || contentScrollThumb == null || mainScrollView == null) return;
        View child = mainScrollView.getChildAt(0);
        if (child == null) return;
        int maxScroll = maxMainScrollY(child);
        int trackHeight = contentScrollbar.getHeight();
        ViewGroup.LayoutParams rawParams = contentScrollThumb.getLayoutParams();
        int thumbHeight = rawParams == null ? dp(76) : Math.max(dp(40), rawParams.height);
        int range = Math.max(1, trackHeight - thumbHeight);
        float ratio = Math.max(0f, Math.min(1f, (y - thumbHeight / 2f) / range));
        mainScrollView.scrollTo(0, Math.round(maxScroll * ratio));
        updateContentScrollbar();
    }

    private int maxMainScrollY(View child) {
        if (mainScrollView == null || child == null) return 0;
        return Math.max(0, child.getHeight() + mainScrollView.getPaddingBottom() - mainScrollView.getHeight());
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
        int selectedFileRowHeight = scaledTextHeight(24, 1, 42);
        TextView fileIcon = text("📄", 24, false, MUTED);
        selectedFileView.addView(fileIcon, new LinearLayout.LayoutParams(dp(34), selectedFileRowHeight));
        fileNameView = text("", 17, true, TEXT);
        selectedFileView.addView(fileNameView, new LinearLayout.LayoutParams(0, selectedFileRowHeight, 1f));
        fileSizeView = text("", 13, false, MUTED);
        fileSizeView.setGravity(Gravity.CENTER_VERTICAL | Gravity.RIGHT);
        selectedFileView.addView(fileSizeView, new LinearLayout.LayoutParams(dp(90), selectedFileRowHeight));
        TextView clear = text("×", 26, true, DANGER);
        clear.setGravity(Gravity.CENTER);
        clear.setOnClickListener(v -> {
            clearSelectedFile();
            v.getParent().requestDisallowInterceptTouchEvent(true);
        });
        selectedFileView.addView(clear, new LinearLayout.LayoutParams(dp(38), selectedFileRowHeight));
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
        stageView.setSingleLine(false);
        stageView.setGravity(Gravity.LEFT | Gravity.CENTER_VERTICAL);
        progressText = text("0%", 13, true, MUTED);
        progressText.setSingleLine(true);
        progressText.setGravity(Gravity.RIGHT | Gravity.CENTER_VERTICAL);
        int headHeight = scaledTextHeight(13, 2, 34);
        head.addView(stageView, new LinearLayout.LayoutParams(0, headHeight, 1f));
        head.addView(progressText, new LinearLayout.LayoutParams(dp(72), headHeight));
        cancelProcessingButton = button("Hủy", DANGER, v -> cancelProcessing());
        cancelProcessingButton.setTextSize(TypedValue.COMPLEX_UNIT_PX, 12 * uiScale());
        LinearLayout.LayoutParams cancelParams = new LinearLayout.LayoutParams(dp(78), Math.max(dp(32), scaledTextHeight(12, 1, 28)));
        cancelParams.leftMargin = dp(8);
        head.addView(cancelProcessingButton, cancelParams);
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

        qualityStripView = new LinearLayout(this);
        qualityStripView.setOrientation(LinearLayout.HORIZONTAL);
        qualityStripView.setGravity(Gravity.CENTER_VERTICAL);
        qualityStripView.setPadding(dp(18), dp(6), dp(18), dp(2));
        qualityStripView.setVisibility(View.GONE);
        panel.addView(qualityStripView, new LinearLayout.LayoutParams(
                LinearLayout.LayoutParams.MATCH_PARENT,
                LinearLayout.LayoutParams.WRAP_CONTENT));

        audioSummaryView = vertical();
        audioSummaryView.setPadding(dp(18), dp(4), dp(18), dp(6));
        audioSummaryView.setVisibility(View.GONE);
        panel.addView(audioSummaryView, new LinearLayout.LayoutParams(
                LinearLayout.LayoutParams.MATCH_PARENT,
                LinearLayout.LayoutParams.WRAP_CONTENT));

        resultTimingView = new LinearLayout(this);
        resultTimingView.setOrientation(LinearLayout.HORIZONTAL);
        resultTimingView.setGravity(Gravity.CENTER_VERTICAL);
        resultTimingView.setPadding(dp(18), dp(4), dp(18), dp(6));
        resultTimingView.setVisibility(View.GONE);
        panel.addView(resultTimingView, new LinearLayout.LayoutParams(
                LinearLayout.LayoutParams.MATCH_PARENT,
                LinearLayout.LayoutParams.WRAP_CONTENT));

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
        playerPlayPauseButton.setPadding(0, 0, 0, 0);
        playerPlayPauseButton.setMinWidth(0);
        playerPlayPauseButton.setMinimumWidth(0);
        playerPlayPauseButton.setIncludeFontPadding(false);
        playerPlayPauseButton.setTextAlignment(View.TEXT_ALIGNMENT_CENTER);
        LinearLayout.LayoutParams playParams = new LinearLayout.LayoutParams(dp(48), dp(48));
        playParams.rightMargin = dp(8);
        player.addView(playerPlayPauseButton, playParams);

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

        librarySearch = new EditText(this);
        librarySearch.setHint("Tìm kiếm...");
        librarySearch.setSingleLine(true);
        librarySearch.setTextSize(TypedValue.COMPLEX_UNIT_PX, 16 * uiScale());
        librarySearch.setTextColor(TEXT);
        librarySearch.setHintTextColor(Color.rgb(135, 135, 135));
        librarySearch.setInputType(InputType.TYPE_CLASS_TEXT);
        librarySearch.setPadding(dp(14), 0, dp(14), 0);
        librarySearch.setBackground(rounded(INPUT, BORDER, 1, 4));
        librarySearch.addTextChangedListener(new TextWatcher() {
            @Override public void beforeTextChanged(CharSequence s, int start, int count, int after) {}
            @Override public void onTextChanged(CharSequence s, int start, int before, int count) {
                librarySearchQuery = s == null ? "" : s.toString();
                refreshLibraryList();
            }
            @Override public void afterTextChanged(Editable s) {}
        });
        header.addView(librarySearch, new LinearLayout.LayoutParams(dp(260), dp(48)));

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
        librarySelectAll = checkPlain("Chọn tất cả", false);
        librarySelectAll.setTextSize(TypedValue.COMPLEX_UNIT_PX, 18 * uiScale());
        librarySelectAll.setOnClickListener(v -> toggleSelectAllLibraryItems(librarySelectAll.isChecked()));
        toolbar.addView(librarySelectAll, new LinearLayout.LayoutParams(0, dp(48), 1f));
        toolbar.addView(button("Xóa đã chọn", DANGER, v -> deleteSelectedLibraryItems()),
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
        if (contentScrollbar != null) contentScrollbar.setVisibility(View.GONE);
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
            updateContentScrollbar();
        }).start();
    }

    private void refreshLibraryList() {
        if (libraryList == null) return;
        libraryList.removeAllViews();
        List<NativeFileLibrary.LibraryItem> allItems = library.listItems();
        List<NativeFileLibrary.LibraryItem> items = new ArrayList<>();
        for (NativeFileLibrary.LibraryItem item : allItems) {
            if (libraryItemMatchesSearch(item)) items.add(item);
        }
        selectedLibraryItemIds.retainAll(itemIds(allItems));
        updateLibrarySelectAllState(items);
        if (items.isEmpty()) {
            TextView empty = text("Chưa có tập tin.", 16, false, MUTED);
            empty.setPadding(dp(12), dp(16), dp(12), dp(16));
            libraryList.addView(empty);
            return;
        }
        SimpleDateFormat fmt = new SimpleDateFormat("d/M/yyyy HH:mm", Locale.getDefault());
        for (NativeFileLibrary.LibraryItem item : items) {
            LinearLayout row = new LinearLayout(this);
            row.setOrientation(LinearLayout.HORIZONTAL);
            row.setGravity(Gravity.CENTER_VERTICAL);
            row.setMinimumHeight(dp(104));
            row.setPadding(0, dp(10), dp(14), dp(10));
            row.setBackground(rounded(CARD, BORDER, 1, 8));
            row.setOnClickListener(v -> openLibraryItem(item));

            View checkArea = libraryCheckArea(item);
            row.addView(checkArea, new LinearLayout.LayoutParams(dp(72), LinearLayout.LayoutParams.MATCH_PARENT));

            LinearLayout body = vertical();
            LinearLayout top = new LinearLayout(this);
            top.setOrientation(LinearLayout.HORIZONTAL);
            top.setGravity(Gravity.CENTER_VERTICAL);

            TextView name = text(item.displayName, 20, true, TEXT);
            name.setSingleLine(false);
            name.setMaxLines(2);
            top.addView(name, new LinearLayout.LayoutParams(0, LinearLayout.LayoutParams.WRAP_CONTENT, 1f));
            top.addView(fileStatusBadge(item));
            body.addView(top);

            TextView original = text(item.originalName, 16, false, MUTED);
            original.setPadding(0, 0, 0, 0);
            original.setSingleLine(false);
            original.setMaxLines(2);
            body.addView(original);

            String meta = formatBytes(item.sourceBytes) + "    " + fmt.format(new Date(item.updatedAtMillis));
            TextView details = text(meta, 15, false, MUTED);
            details.setPadding(0, dp(2), 0, 0);
            body.addView(details);
            row.addView(body, new LinearLayout.LayoutParams(0, LinearLayout.LayoutParams.WRAP_CONTENT, 1f));
            libraryList.addView(withMargins(row, 0, 0, 0, 10));
        }
    }

    private View libraryCheckArea(NativeFileLibrary.LibraryItem item) {
        FrameLayout area = new FrameLayout(this);
        area.setPadding(dp(8), 0, dp(8), 0);
        CheckBox box = checkPlain("", selectedLibraryItemIds.contains(item.id));
        box.setClickable(false);
        box.setFocusable(false);
        FrameLayout.LayoutParams params = new FrameLayout.LayoutParams(dp(54), dp(54), Gravity.CENTER);
        area.addView(box, params);
        area.setOnClickListener(v -> toggleLibraryItemSelection(item.id));
        area.setOnTouchListener((v, event) -> {
            if (event.getActionMasked() == MotionEvent.ACTION_UP) {
                v.performClick();
            }
            return true;
        });
        return area;
    }

    private boolean libraryItemMatchesSearch(NativeFileLibrary.LibraryItem item) {
        String query = normalizeSearch(librarySearchQuery);
        if (query.isEmpty()) return true;
        String haystack = normalizeSearch(item.displayName + " " + item.originalName + " " + item.status);
        return haystack.contains(query);
    }

    private String normalizeSearch(String value) {
        return value == null ? "" : value.trim().toLowerCase(Locale.ROOT);
    }

    private Set<String> itemIds(List<NativeFileLibrary.LibraryItem> items) {
        Set<String> ids = new HashSet<>();
        for (NativeFileLibrary.LibraryItem item : items) ids.add(item.id);
        return ids;
    }

    private void toggleLibraryItemSelection(String id) {
        if (id == null) return;
        if (selectedLibraryItemIds.contains(id)) selectedLibraryItemIds.remove(id);
        else selectedLibraryItemIds.add(id);
        refreshLibraryList();
    }

    private void toggleSelectAllLibraryItems(boolean selected) {
        List<NativeFileLibrary.LibraryItem> items = library.listItems();
        for (NativeFileLibrary.LibraryItem item : items) {
            if (!libraryItemMatchesSearch(item)) continue;
            if (selected) selectedLibraryItemIds.add(item.id);
            else selectedLibraryItemIds.remove(item.id);
        }
        refreshLibraryList();
    }

    private void updateLibrarySelectAllState(List<NativeFileLibrary.LibraryItem> visibleItems) {
        if (librarySelectAll == null) return;
        librarySelectAll.setOnClickListener(null);
        boolean allVisibleSelected = !visibleItems.isEmpty();
        for (NativeFileLibrary.LibraryItem item : visibleItems) {
            if (!selectedLibraryItemIds.contains(item.id)) {
                allVisibleSelected = false;
                break;
            }
        }
        librarySelectAll.setChecked(allVisibleSelected);
        librarySelectAll.setOnClickListener(v -> toggleSelectAllLibraryItems(librarySelectAll.isChecked()));
    }

    private void deleteSelectedLibraryItems() {
        if (selectedLibraryItemIds.isEmpty()) {
            appendLog("Chưa chọn tập tin để xóa.");
            return;
        }
        List<String> ids = new ArrayList<>(selectedLibraryItemIds);
        int deleted = 0;
        for (String id : ids) {
            try {
                if (id.equals(activeProcessingItemId)) cancelProcessing();
                library.deleteItem(id);
                deleted++;
            } catch (Exception error) {
                appendLog("Xóa tập tin thất bại: " + error.getMessage());
            }
        }
        selectedLibraryItemIds.clear();
        refreshLibraryList();
        appendLog("Đã xóa " + deleted + " tập tin.");
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
        labelView.setSingleLine(false);
        labelView.setMaxLines(2);
        row.addView(labelView, new LinearLayout.LayoutParams(controlLabelWidth(), LinearLayout.LayoutParams.WRAP_CONTENT));
        row.addView(control, new LinearLayout.LayoutParams(0, settings.uiTextScalePercent() >= 170 ? dp(56) : dp(44), 1f));
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
        int rowHeight = scaledTextHeight(17, 1, 38);
        row.addView(labelView, new LinearLayout.LayoutParams(controlLabelWidth(), rowHeight));
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
        row.addView(seek, new LinearLayout.LayoutParams(0, rowHeight, 1f));
        TextView val = text(formatConfidenceLabel(initial), 17, false, MUTED);
        val.setSingleLine(true);
        val.setGravity(Gravity.RIGHT | Gravity.CENTER_VERTICAL);
        row.addView(val, new LinearLayout.LayoutParams(sliderValueWidth(), rowHeight));
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
        int rowHeight = scaledTextHeight(17, 1, 38);
        row.addView(labelView, new LinearLayout.LayoutParams(controlLabelWidth(), rowHeight));

        SeekBar seek = new SeekBar(this);
        seek.setPadding(0, 0, 0, 0);
        seek.setThumbOffset(0);
        seek.setMax(10);
        int current = settings.uiTextScalePercent();
        seek.setProgress(Math.max(0, Math.min(10, (current - 100) / 10)));
        if (Build.VERSION.SDK_INT >= 21) {
            seek.setProgressTintList(ColorStateList.valueOf(ACCENT));
            seek.setThumbTintList(ColorStateList.valueOf(ACCENT));
            seek.setProgressBackgroundTintList(ColorStateList.valueOf(Color.rgb(235, 235, 235)));
        }
        row.addView(seek, new LinearLayout.LayoutParams(0, rowHeight, 1f));

        TextView val = text(formatTextScale(current), 17, false, MUTED);
        val.setSingleLine(true);
        val.setGravity(Gravity.RIGHT | Gravity.CENTER_VERTICAL);
        row.addView(val, new LinearLayout.LayoutParams(sliderValueWidth(), rowHeight));
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
        if (isProcessingActive()) {
            appendLog("Cỡ chữ sẽ áp dụng đầy đủ sau khi xử lý xong.");
            return;
        }
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
            mainScrollView.post(() -> {
                mainScrollView.scrollTo(0, restoreScrollY);
                updateContentScrollbar();
            });
        }
    }

    private int controlLabelWidth() {
        int width = getResources().getDisplayMetrics().widthPixels;
        return Math.max(dp(140), Math.min(dp(220), Math.round(width * 0.34f)));
    }

    private int sliderValueWidth() {
        int width = getResources().getDisplayMetrics().widthPixels;
        return Math.max(dp(74), Math.min(dp(120), Math.round(width * 0.18f)));
    }

    private View fullCheck(String label, boolean checked, String key) {
        CheckBox box = checkPlain(label, checked);
        box.setPadding(0, dp(2), 0, dp(2));
        box.setTextSize(TypedValue.COMPLEX_UNIT_PX, 17 * uiScale());
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
                view.setTextSize(TypedValue.COMPLEX_UNIT_PX, 17 * uiScale());
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
            boolean selected = selectedColor[0] == color;
            TextView dot = new TextView(this);
            GradientDrawable bg = new GradientDrawable();
            bg.setShape(GradientDrawable.OVAL);
            bg.setColor(color);
            bg.setStroke(dp(selected ? 3 : 1), selected ? TEXT : BORDER);
            dot.setBackground(bg);
            dot.setOnClickListener(v -> {
                selectedColor[0] = color;
                renderColorDots(row, selectedColor);
            });
            int size = selected ? dp(44) : dp(32);
            LinearLayout.LayoutParams params = new LinearLayout.LayoutParams(size, size);
            params.setMargins(0, 0, dp(12), 0);
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
        LinearLayout.LayoutParams params = new LinearLayout.LayoutParams(0, scaledTextHeight(17, 2, 46), 1f);
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
        box.setTextSize(TypedValue.COMPLEX_UNIT_PX, 17 * uiScale());
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
        button.setSingleLine(false);
        button.setMaxLines(2);
        button.setIncludeFontPadding(true);
        button.setLineSpacing(0, 1.0f);
        button.setMinHeight(0);
        button.setMinimumHeight(0);
        button.setMinWidth(0);
        button.setMinimumWidth(0);
        button.setBackground(rounded(color, BORDER, 1, 5));
        button.setOnClickListener(listener);
        return button;
    }

    private TextView badge(String label) {
        return badge(label, SUCCESS);
    }

    private TextView badge(String label, int color) {
        TextView badge = text(label, 16, true, TEXT);
        badge.setGravity(Gravity.CENTER);
        badge.setPadding(dp(12), 0, dp(12), 0);
        badge.setBackground(rounded(color, color, 1, 4));
        return badge;
    }

    private TextView fileStatusBadge(NativeFileLibrary.LibraryItem item) {
        if (library.hasResult(item.id) || NativeFileLibrary.STATUS_COMPLETE.equals(item.status)) {
            return badge("Hoàn thành", SUCCESS);
        }
        if (NativeFileLibrary.STATUS_ERROR.equals(item.status)) {
            return badge("Lỗi", DANGER);
        }
        if (NativeFileLibrary.STATUS_CANCELLED.equals(item.status)) {
            return badge("Đã hủy", ELEVATED);
        }
        if (NativeFileLibrary.STATUS_PROCESSING.equals(item.status)) {
            return badge("Đang xử lý", PRIMARY);
        }
        return badge("Sẵn sàng", ELEVATED);
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

    private int topBarHeight() {
        return Math.max(dp(78), scaledTextHeight(17, 2, 68) + dp(10));
    }

    private int scaledTextHeight(int sp, int lines, int minDp) {
        int safeLines = Math.max(1, lines);
        return Math.max(dp(minDp), Math.round(sp * uiScale() * 1.35f * safeLines + dp(8)));
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
        if (hotwordPanel != null && hotwordPanel.getParent() != null && hotwordPanel.getVisibility() == View.VISIBLE) {
            hotwordPanel.bringToFront();
            return;
        }
        if (hotwordPanel != null && hotwordPanel.getParent() != null) {
            rootFrame.removeView(hotwordPanel);
            hotwordPanel = null;
        }

        LinearLayout root = vertical();
        root.setPadding(dp(14), dp(12) + statusBarHeight(), dp(14), dp(12));
        root.setBackground(rounded(CARD, BORDER, 1, 0));

        activeHotwordItems = parseHotwordItems(settings.hotwordsText());

        LinearLayout header = new LinearLayout(this);
        header.setOrientation(LinearLayout.HORIZONTAL);
        header.setGravity(Gravity.CENTER_VERTICAL);
        TextView panelTitle = text("Quản lý hotword.txt", 22, true, TEXT);
        header.addView(panelTitle, new LinearLayout.LayoutParams(0, dp(54), 1f));
        TextView close = text("×", 34, true, MUTED);
        close.setGravity(Gravity.CENTER);
        close.setOnClickListener(v -> closeHotwordPanel(false));
        header.addView(close, new LinearLayout.LayoutParams(dp(52), dp(54)));
        root.addView(header);

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
                0,
                1f));
        renderHotwordRows();

        LinearLayout footer = new LinearLayout(this);
        footer.setOrientation(LinearLayout.HORIZONTAL);
        footer.setGravity(Gravity.RIGHT | Gravity.CENTER_VERTICAL);
        footer.setPadding(0, dp(10), 0, 0);
        Button cancel = button("Hủy", INPUT, v -> closeHotwordPanel(false));
        Button save = button("Lưu", PRIMARY, v -> {
            closeHotwordPanel(true);
        });
        footer.addView(cancel, new LinearLayout.LayoutParams(dp(120), dp(46)));
        LinearLayout.LayoutParams saveParams = new LinearLayout.LayoutParams(dp(120), dp(46));
        saveParams.leftMargin = dp(8);
        footer.addView(save, saveParams);
        root.addView(footer);

        hotwordPanel = root;
        int screenWidth = getResources().getDisplayMetrics().widthPixels;
        int panelWidth = screenWidth < dp(900)
                ? (int) (screenWidth * 0.95f)
                : Math.min(dp(640), (int) (screenWidth * 0.95f));
        FrameLayout.LayoutParams params = new FrameLayout.LayoutParams(
                panelWidth,
                FrameLayout.LayoutParams.MATCH_PARENT,
                Gravity.RIGHT);
        if (hotwordScrim != null) {
            hotwordScrim.setVisibility(View.VISIBLE);
            hotwordScrim.bringToFront();
        }
        if (contentScrollbar != null) contentScrollbar.setVisibility(View.GONE);
        rootFrame.addView(hotwordPanel, params);
        hotwordPanel.bringToFront();
        hotwordPanel.setTranslationX(panelWidth);
        hotwordPanel.animate().translationX(0).setDuration(190).setInterpolator(new DecelerateInterpolator()).start();
        if (fixedPlayerView != null) fixedPlayerView.setVisibility(View.GONE);
    }

    private void closeHotwordPanel(boolean save) {
        if (save && activeHotwordItems != null) {
            settings.setHotwordsText(buildHotwordsText(activeHotwordItems));
        }
        if (hotwordPanel == null || hotwordPanel.getParent() == null) {
            if (hotwordScrim != null) hotwordScrim.setVisibility(View.GONE);
            clearActiveHotwordDialogState();
            return;
        }
        View closing = hotwordPanel;
        closing.animate().translationX(closing.getWidth()).setDuration(160).withEndAction(() -> {
            if (closing.getParent() != null) rootFrame.removeView(closing);
            if (hotwordPanel == closing) hotwordPanel = null;
            if (hotwordScrim != null) hotwordScrim.setVisibility(View.GONE);
            clearActiveHotwordDialogState();
            if (fixedPlayerView != null && mediaPlayer != null) fixedPlayerView.setVisibility(View.VISIBLE);
            updateContentScrollbar();
        }).start();
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
        discardStagedInputFile();
        selectedLibraryItemId = null;
        updateSelectedFileUi(displayName(selectedUri), -1);
        prepareMediaPlayer(selectedUri);
        clearTranscript();
        appendLog("Selected: " + selectedUri);
    }

    private void handleIncomingFileIntent(Intent intent) {
        if (intent == null) return;
        Uri uri = incomingFileUri(intent);
        if (uri == null) return;
        try {
            Log.d(TAG, "Incoming shared file: " + uri);
            String name = displayName(uri);
            File staged = stageIncomingFile(uri, name);
            discardStagedInputFile();
            selectedStagedInputFile = staged;
            selectedUri = Uri.fromFile(staged);
            selectedLibraryItemId = null;
            updateSelectedFileUi(name, staged.length());
            prepareMediaPlayer(selectedUri);
            clearTranscript();
            appendLog("Received shared file: " + name);
        } catch (Exception error) {
            Log.e(TAG, "Receive shared file failed", error);
            appendLog("Receive shared file failed: " + error.getMessage());
        }
    }

    private boolean isProcessingActive() {
        return activeProcessingItemId != null
                || (progressContainer != null && progressContainer.getVisibility() == View.VISIBLE);
    }

    private Uri incomingFileUri(Intent intent) {
        String action = intent.getAction();
        if (Intent.ACTION_SEND.equals(action)) {
            Object stream = intent.getExtras() == null ? null : intent.getExtras().get(Intent.EXTRA_STREAM);
            if (stream instanceof Uri) return (Uri) stream;
            if (stream instanceof String) return Uri.parse((String) stream);
        }
        if (Intent.ACTION_SEND_MULTIPLE.equals(action)) {
            Object streams = intent.getExtras() == null ? null : intent.getExtras().get(Intent.EXTRA_STREAM);
            if (streams instanceof ArrayList && !((ArrayList<?>) streams).isEmpty()) {
                Object first = ((ArrayList<?>) streams).get(0);
                if (first instanceof Uri) return (Uri) first;
                if (first instanceof String) return Uri.parse((String) first);
            }
        }
        if (Intent.ACTION_VIEW.equals(action)) {
            Uri data = intent.getData();
            if (data != null) return data;
        }
        ClipData clip = intent.getClipData();
        if (clip != null && clip.getItemCount() > 0) return clip.getItemAt(0).getUri();
        return null;
    }

    private File stageIncomingFile(Uri uri, String displayName) throws Exception {
        File inbox = new File(getFilesDir(), "inbox");
        if (!inbox.exists() && !inbox.mkdirs()) throw new IllegalStateException("Cannot create inbox directory.");
        File target = new File(inbox, "incoming_" + System.currentTimeMillis() + extensionOf(displayName));
        try (InputStream input = getContentResolver().openInputStream(uri);
             FileOutputStream output = new FileOutputStream(target)) {
            if (input == null) throw new IllegalStateException("Cannot open shared file.");
            byte[] buffer = new byte[1024 * 1024];
            int read;
            while ((read = input.read(buffer)) >= 0) {
                output.write(buffer, 0, read);
            }
        }
        return target;
    }

    private void discardStagedInputFile() {
        if (selectedStagedInputFile != null && selectedStagedInputFile.isFile()) {
            selectedStagedInputFile.delete();
        }
        selectedStagedInputFile = null;
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

    private String extensionOf(String name) {
        if (name == null) return ".bin";
        int slash = Math.max(name.lastIndexOf('/'), name.lastIndexOf('\\'));
        int dot = name.lastIndexOf('.');
        if (dot <= slash || dot < 0 || dot == name.length() - 1) return ".bin";
        String ext = name.substring(dot).replaceAll("[^A-Za-z0-9.]", "");
        return ext.isEmpty() ? ".bin" : ext.toLowerCase(Locale.US);
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
        if (ProcessingForegroundService.isProcessing()) {
            cancelProcessing();
        }
        selectedUri = null;
        selectedLibraryItemId = null;
        discardStagedInputFile();
        updateSelectedFileUi(null, 0);
        releaseMediaPlayer();
        clearTranscript();
        if (resultPanelView != null) resultPanelView.setVisibility(View.GONE);
        if (progressContainer != null) progressContainer.setVisibility(View.GONE);
    }

    private void openLibraryItem(NativeFileLibrary.LibraryItem item) {
        closeDrawer();
        discardStagedInputFile();
        selectedLibraryItemId = item.id;
        selectedUri = Uri.fromFile(item.sourceFile);
        updateSelectedFileUi(item.originalName, item.sourceFile.length());
        prepareMediaPlayer(selectedUri);
        if (progressContainer != null) progressContainer.setVisibility(View.GONE);
        try {
            String json = library.readResult(item.id);
            if (json != null && !json.trim().isEmpty()) {
                renderTranscriptFromJson(json);
            } else if (NativeFileLibrary.STATUS_PROCESSING.equals(item.status)) {
                clearTranscript();
                if (resultPanelView != null) resultPanelView.setVisibility(View.GONE);
                resumeLibraryItemInForeground(item);
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
        discardStagedInputFile();
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
            rootFrame.postDelayed(() -> startProcessingSelectedFile(null), 500);
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

        String fallbackName = baseNameForDisplay(selectedDisplayName);
        EditText input = darkSingleLineInput(fallbackName);
        input.setText(fallbackName);
        input.selectAll();

        LinearLayout body = vertical();
        body.setPadding(dp(22), dp(18), dp(22), dp(18));
        body.setBackground(rounded(CARD, BORDER, 1, 8));
        TextView title = text("Đặt tên kết quả", 22, true, TEXT);
        title.setPadding(0, 0, 0, dp(12));
        body.addView(title);
        TextView label = text("Tên cuộc họp / tập tin:", 17, false, MUTED);
        label.setPadding(0, 0, 0, dp(8));
        body.addView(label);
        body.addView(input, new LinearLayout.LayoutParams(
                LinearLayout.LayoutParams.MATCH_PARENT,
                dp(48)));

        final AlertDialog[] holder = new AlertDialog[1];
        LinearLayout actions = new LinearLayout(this);
        actions.setOrientation(LinearLayout.HORIZONTAL);
        actions.setGravity(Gravity.RIGHT);
        actions.setPadding(0, dp(16), 0, 0);
        Button cancel = button("Hủy", INPUT, v -> {
            if (holder[0] != null) holder[0].dismiss();
        });
        Button run = button("Xử lý", PRIMARY, v -> {
            String requestedName = input.getText() == null ? "" : input.getText().toString();
            if (holder[0] != null) holder[0].dismiss();
            startProcessingSelectedFile(requestedName);
        });
        LinearLayout.LayoutParams actionParams = new LinearLayout.LayoutParams(0, dp(46), 1f);
        actionParams.rightMargin = dp(8);
        actions.addView(cancel, actionParams);
        LinearLayout.LayoutParams runParams = new LinearLayout.LayoutParams(0, dp(46), 1f);
        actions.addView(run, runParams);
        body.addView(actions);

        AlertDialog dialog = new AlertDialog.Builder(this)
                .setView(body)
                .create();
        holder[0] = dialog;
        dialog.show();
        Window window = dialog.getWindow();
        if (window != null) {
            window.setBackgroundDrawable(new ColorDrawable(Color.TRANSPARENT));
        }
        input.requestFocus();
        window = dialog.getWindow();
        if (window != null) {
            window.setSoftInputMode(WindowManager.LayoutParams.SOFT_INPUT_STATE_ALWAYS_VISIBLE);
        }
    }

    private void startProcessingSelectedFile(String requestedDisplayName) {
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
        appendLog("Processing started in foreground service.");
        activeProcessingItemId = selectedLibraryItemId;
        Intent service = new Intent(this, ProcessingForegroundService.class)
                .setAction(ProcessingForegroundService.ACTION_START)
                .setData(selectedUri)
                .addFlags(Intent.FLAG_GRANT_READ_URI_PERMISSION)
                .putExtra(ProcessingForegroundService.EXTRA_DISPLAY_NAME, normalizeRequestedDisplayName(requestedDisplayName));
        if (Build.VERSION.SDK_INT >= 26) {
            startForegroundService(service);
        } else {
            startService(service);
        }
    }

    private void resumeLibraryItemInForeground(NativeFileLibrary.LibraryItem item) {
        if (item == null) return;
        activeProcessingItemId = item.id;
        setProgress("Resume", 3);
        appendLog("Resuming from library: " + item.displayName);
        Intent service = new Intent(this, ProcessingForegroundService.class)
                .setAction(ProcessingForegroundService.ACTION_RESUME)
                .putExtra(ProcessingForegroundService.EXTRA_ITEM_ID, item.id);
        if (Build.VERSION.SDK_INT >= 26) {
            startForegroundService(service);
        } else {
            startService(service);
        }
    }

    private void cancelProcessing() {
        activeProcessingItemId = null;
        Intent service = new Intent(this, ProcessingForegroundService.class)
                .setAction(ProcessingForegroundService.ACTION_CANCEL);
        startService(service);
        setProgress("Đang hủy", 99);
    }

    private void resumeInterruptedJobIfNeeded() {
        if (ProcessingForegroundService.isProcessing()) return;
        if (selectedUri != null || !settings.resumeAfterKill()) return;
        NativeFileLibrary.LibraryItem item = library.latestProcessingItem();
        if (item == null) return;
        selectedLibraryItemId = item.id;
        selectedUri = Uri.fromFile(item.sourceFile);
        updateSelectedFileUi(item.originalName, item.sourceFile.length());
        prepareMediaPlayer(selectedUri);
        clearTranscript();
        if (resultPanelView != null) resultPanelView.setVisibility(View.GONE);
        setProgress("Resume", 3);
        appendLog("Resuming interrupted job: " + item.displayName);
        pipeline.resumeLibraryItem(item.id, processingListener());
    }

    private NativeOfflinePipeline.ProgressListener processingListener() {
        return new NativeOfflinePipeline.ProgressListener() {
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
                    discardStagedInputFile();
                    selectedLibraryItemId = item.id;
                    selectedUri = Uri.fromFile(item.sourceFile);
                    updateSelectedFileUi(item.originalName, item.sourceFile.length());
                    prepareMediaPlayer(selectedUri);
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
        };
    }

    private String normalizeRequestedDisplayName(String value) {
        String trimmed = value == null ? "" : value.replace('\r', ' ').replace('\n', ' ').trim();
        return trimmed.isEmpty() ? baseNameForDisplay(selectedDisplayName) : trimmed;
    }

    private String baseNameForDisplay(String name) {
        if (name == null || name.trim().isEmpty()) return "Tập tin";
        String value = name.trim();
        int slash = Math.max(value.lastIndexOf('/'), value.lastIndexOf('\\'));
        if (slash >= 0 && slash < value.length() - 1) value = value.substring(slash + 1);
        int dot = value.lastIndexOf('.');
        if (dot > 0) value = value.substring(0, dot);
        return value.isEmpty() ? "Tập tin" : value;
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
        updateActiveSegment(targetMs / 1000.0, false);
        updatePlayerTime();
    }

    private double segmentStartSeconds(int segmentIndex) {
        if (segmentIndex >= 0 && segmentIndex < editorSegments.size()) {
            return Math.max(0.0, editorSegments.get(segmentIndex).start);
        }
        return 0.0;
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
        activeWordIndex = -1;
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
        currentQualityInfoJson = null;
        currentTimingJson = null;
        currentAudioSummaryJson = null;
        editorSegments.clear();
        editorSpeakers.clear();
        activeSegmentIndex = -1;
        activeWordIndex = -1;
        renderResultMetadata(null);
        if (transcriptRoot == null) return;
        transcriptRoot.removeAllViews();
        transcriptEmptyView = text("Chưa có nội dung. Chọn file và bấm Xử lý để hiển thị transcript tại đây.", 16, false, MUTED);
        transcriptEmptyView.setPadding(dp(10), dp(8), dp(10), dp(8));
        transcriptRoot.addView(transcriptEmptyView);
        updateContentScrollbar();
    }

    private void renderResultMetadata(JSONObject json) {
        if (json == null) {
            if (qualityStripView != null) qualityStripView.setVisibility(View.GONE);
            if (audioSummaryView != null) audioSummaryView.setVisibility(View.GONE);
            if (resultTimingView != null) resultTimingView.setVisibility(View.GONE);
            return;
        }
        renderQualityStrip(json.optJSONObject("quality_info"));
        renderAudioSummary(json.optJSONArray("audio_summary"));
        renderResultTiming(json.optJSONObject("timing"));
    }

    private void renderQualityStrip(JSONObject quality) {
        if (qualityStripView == null) return;
        qualityStripView.removeAllViews();
        if (quality == null) {
            qualityStripView.setVisibility(View.GONE);
            return;
        }
        SpannableStringBuilder line = new SpannableStringBuilder("Chất lượng:");
        int itemCount = 0;
        itemCount += appendQualityItem(line, itemCount, quality, "dnsmos_sig", "Giọng nói", true);
        itemCount += appendQualityItem(line, itemCount, quality, "dnsmos_bak", "Nhiễu nền", true);
        itemCount += appendQualityItem(line, itemCount, quality, "dnsmos_ovrl", "Tổng thể", true);
        itemCount += appendQualityItem(line, itemCount, quality, "asr_confidence", "Mức độ tự tin dịch chính xác", false);
        if (itemCount == 0) {
            qualityStripView.setVisibility(View.GONE);
            return;
        }
        TextView text = text("", 14, true, MUTED);
        text.setText(line);
        text.setSingleLine(false);
        text.setLineSpacing(dp(1), 1.0f);
        qualityStripView.addView(text, new LinearLayout.LayoutParams(
                LinearLayout.LayoutParams.MATCH_PARENT,
                LinearLayout.LayoutParams.WRAP_CONTENT));
        qualityStripView.setVisibility(View.VISIBLE);
    }

    private int appendQualityItem(
            SpannableStringBuilder line,
            int index,
            JSONObject quality,
            String key,
            String label,
            boolean dnsScore
    ) {
        if (!quality.has(key)) return 0;
        double score = quality.optDouble(key, Double.NaN);
        if (!Double.isFinite(score)) return 0;
        line.append(index == 0 ? " " : " · ");
        line.append(label).append(' ');
        String value = dnsScore
                ? String.format(Locale.US, "%.1f/5", score)
                : String.format(Locale.US, "%.1f%%", score * 100.0);
        int start = line.length();
        line.append(value);
        int color = dnsScore ? dnsmosColor(score) : confidenceColor(score);
        line.setSpan(new ForegroundColorSpan(color), start, line.length(), Spanned.SPAN_EXCLUSIVE_EXCLUSIVE);
        return 1;
    }

    private int dnsmosColor(double score) {
        if (score >= 4.0) return Color.rgb(40, 167, 69);
        if (score >= 3.0) return Color.rgb(92, 184, 92);
        if (score >= 2.0) return Color.rgb(255, 193, 7);
        return DANGER;
    }

    private int confidenceColor(double score) {
        if (score >= 0.80) return Color.rgb(40, 167, 69);
        if (score >= 0.60) return Color.rgb(255, 193, 7);
        return DANGER;
    }

    private void renderAudioSummary(JSONArray items) {
        if (audioSummaryView == null) return;
        audioSummaryView.removeAllViews();
        audioSummaryView.setVisibility(View.GONE);
    }

    private LinearLayout.LayoutParams summaryTileParams(boolean left) {
        LinearLayout.LayoutParams params = new LinearLayout.LayoutParams(0, LinearLayout.LayoutParams.WRAP_CONTENT, 1f);
        params.setMargins(left ? 0 : dp(4), dp(3), left ? dp(4) : 0, dp(3));
        return params;
    }

    private View summaryTile(JSONObject item) {
        LinearLayout tile = vertical();
        tile.setPadding(dp(8), dp(5), dp(8), dp(5));
        tile.setBackground(rounded(INPUT, BORDER, 1, 4));
        String label = item == null ? "" : item.optString("label", "");
        String value = item == null ? "" : item.optString("value", "");
        TextView labelView = text(label, 12, false, MUTED);
        labelView.setSingleLine(false);
        TextView valueView = text(value, 14, true, TEXT);
        valueView.setSingleLine(false);
        valueView.setLineSpacing(0, 0.95f);
        tile.addView(labelView);
        tile.addView(valueView);
        return tile;
    }

    private void renderResultTiming(JSONObject timing) {
        if (resultTimingView == null) return;
        resultTimingView.removeAllViews();
        if (timing == null) {
            resultTimingView.setVisibility(View.GONE);
            return;
        }
        String[][] labels = new String[][]{
                {"preprocessing", "PreProcessing"},
                {"transcription_detail", "ASR"},
                {"diarization", "Phân tách người nói"},
                {"punctuation", "Dấu câu"},
                {"overlap_separation", "Tách overlap"},
                {"total", "Tổng thời gian"}
        };
        StringBuilder out = new StringBuilder();
        for (String[] item : labels) {
            double value = timing.optDouble(item[0], 0.0);
            if (!Double.isFinite(value) || value <= 0.0) continue;
            if (out.length() > 0) out.append("  ·  ");
            out.append(item[1]).append(": ").append(String.format(Locale.US, "%.1fs", value));
        }
        if (out.length() == 0) {
            resultTimingView.setVisibility(View.GONE);
            return;
        }
        TextView text = text(out.toString(), 13, true, MUTED);
        text.setSingleLine(false);
        text.setLineSpacing(dp(1), 1.0f);
        resultTimingView.addView(text, new LinearLayout.LayoutParams(
                LinearLayout.LayoutParams.MATCH_PARENT,
                LinearLayout.LayoutParams.WRAP_CONTENT));
        resultTimingView.setVisibility(View.VISIBLE);
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
        JSONObject qualityInfo = json.optJSONObject("quality_info");
        JSONObject timingInfo = json.optJSONObject("timing");
        JSONArray audioSummary = json.optJSONArray("audio_summary");
        currentQualityInfoJson = qualityInfo == null ? null : qualityInfo.toString();
        currentTimingJson = timingInfo == null ? null : timingInfo.toString();
        currentAudioSummaryJson = audioSummary == null ? null : audioSummary.toString();
        renderResultMetadata(json);
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
                        speaker,
                        item.optJSONArray("raw_words"));
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
        addTextSegmentMaybeSplit(start, end, text, speaker, null);
    }

    private void addTextSegmentMaybeSplit(double start, double end, String text, int speaker, JSONArray rawWords) {
        if (rawWords != null && rawWords.length() > 0) {
            List<TimedWord> timedWords = new ArrayList<>();
            for (int i = 0; i < rawWords.length(); i++) {
                JSONObject word = rawWords.optJSONObject(i);
                if (word == null) continue;
                timedWords.add(new TimedWord(
                        word.optString("text", ""),
                        word.optDouble("start", start),
                        word.optDouble("end", start)));
            }
            if (!timedWords.isEmpty()) {
                int cursor = 0;
                while (cursor < timedWords.size()) {
                    int next = Math.min(timedWords.size(), cursor + 32);
                    editorSegments.add(segmentFromWords(timedWords.subList(cursor, next), speaker));
                    cursor = next;
                }
                return;
            }
        }
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
        block.setBackground(rounded(activeBlock ? Color.rgb(78, 78, 78) : ELEVATED, BORDER, 1, 4));
        block.setOnLongClickListener(v -> {
            showSegmentActions(activeBlock ? activeSegmentIndex : startIndex);
            return true;
        });

        TextView speaker = text(meta.name + ":", 17, true, meta.color);
        speaker.setPadding(0, 0, 0, dp(2));
        speaker.setGravity(Gravity.LEFT | Gravity.CENTER_VERTICAL);
        speaker.setTextAlignment(View.TEXT_ALIGNMENT_TEXT_START);
        speaker.setOnClickListener(v -> showRenameDialog(speakerId, false, startIndex));
        block.addView(speaker, new LinearLayout.LayoutParams(
                LinearLayout.LayoutParams.MATCH_PARENT,
                LinearLayout.LayoutParams.WRAP_CONTENT));

        TextView content = text("", 17, false, TEXT);
        content.setGravity(Gravity.FILL_HORIZONTAL | Gravity.TOP);
        content.setTextAlignment(View.TEXT_ALIGNMENT_TEXT_START);
        content.setLineSpacing(dp(3), 1.04f);
        if (Build.VERSION.SDK_INT >= 26) {
            content.setJustificationMode(Layout.JUSTIFICATION_MODE_INTER_WORD);
            content.setBreakStrategy(Layout.BREAK_STRATEGY_HIGH_QUALITY);
            content.setHyphenationFrequency(Layout.HYPHENATION_FREQUENCY_NONE);
        }
        content.setText(segmentSpans(startIndex, endIndex));
        content.setMovementMethod(LinkMovementMethod.getInstance());
        content.setHighlightColor(HIGHLIGHT_BG);
        content.setTextIsSelectable(false);
        content.setFocusable(false);
        content.setFocusableInTouchMode(false);
        content.setOnTouchListener((v, event) -> {
            if (event.getActionMasked() == MotionEvent.ACTION_DOWN) {
                selectSpanUnderTouch((TextView) v, event);
            }
            return false;
        });
        content.setOnLongClickListener(v -> {
            int target = activeSegmentIndex >= startIndex && activeSegmentIndex <= endIndex
                    ? activeSegmentIndex
                    : startIndex;
            showSegmentActions(target);
            return true;
        });
        block.addView(content, new LinearLayout.LayoutParams(
                LinearLayout.LayoutParams.MATCH_PARENT,
                LinearLayout.LayoutParams.WRAP_CONTENT));
        return withMargins(block, 0, 0, 0, 8);
    }

    private SpannableStringBuilder segmentSpans(int startIndex, int endIndex) {
        SpannableStringBuilder builder = new SpannableStringBuilder();
        for (int i = startIndex; i <= endIndex; i++) {
            EditorSegment segment = editorSegments.get(i);
            boolean activeSegment = i == activeSegmentIndex;
            int segmentStart = builder.length();
            if (segment.words != null && !segment.words.isEmpty()) {
                for (int wordIndex = 0; wordIndex < segment.words.size(); wordIndex++) {
                    TimedWord word = segment.words.get(wordIndex);
                    appendSeekSpan(
                            builder,
                            word.text,
                            finiteOr(word.start, segment.start),
                            i,
                            wordIndex,
                            activeSegment);
                    builder.append(' ');
                }
            } else {
                appendSeekSpan(builder, segment.text, segment.start, i, -1, activeSegment);
                builder.append(' ');
            }
            int segmentEnd = builder.length();
            if (activeSegment && segmentEnd > segmentStart) {
                builder.setSpan(new android.text.style.BackgroundColorSpan(HIGHLIGHT_BG),
                        segmentStart,
                        segmentEnd,
                        Spanned.SPAN_EXCLUSIVE_EXCLUSIVE);
            }
        }
        return builder;
    }

    private void appendSeekSpan(SpannableStringBuilder builder, String value, double seconds, int segmentIndex, int wordIndex, boolean active) {
        int start = builder.length();
        builder.append(value == null ? "" : value);
        int end = builder.length();
        if (end <= start) return;
        builder.setSpan(new WordSeekSpan(seconds, segmentIndex, wordIndex, active),
                start,
                end,
                Spanned.SPAN_EXCLUSIVE_EXCLUSIVE);
    }

    private void selectTranscriptTarget(int segmentIndex, int wordIndex) {
        if (segmentIndex < 0 || segmentIndex >= editorSegments.size()) return;
        activeSegmentIndex = segmentIndex;
        activeWordIndex = wordIndex >= 0 ? wordIndex : -1;
    }

    private boolean selectSpanUnderTouch(TextView view, MotionEvent event) {
        CharSequence text = view.getText();
        if (!(text instanceof Spanned) || view.getLayout() == null) return false;
        int x = (int) event.getX() - view.getTotalPaddingLeft() + view.getScrollX();
        int y = (int) event.getY() - view.getTotalPaddingTop() + view.getScrollY();
        Layout layout = view.getLayout();
        int line = layout.getLineForVertical(Math.max(0, y));
        int offset = layout.getOffsetForHorizontal(line, Math.max(0, x));
        Spanned spanned = (Spanned) text;
        WordSeekSpan[] spans = spanned.getSpans(offset, offset, WordSeekSpan.class);
        if (spans.length == 0 && offset > 0) {
            spans = spanned.getSpans(offset - 1, offset - 1, WordSeekSpan.class);
        }
        if (spans.length == 0) return false;
        selectTranscriptTarget(spans[0].segmentIndex, spans[0].wordIndex);
        return true;
    }

    private SegmentSplit splitSegmentAtWordBoundary(int segmentIndex, int wordBoundary) {
        if (segmentIndex < 0 || segmentIndex >= editorSegments.size()) {
            return new SegmentSplit(segmentIndex, segmentIndex);
        }
        EditorSegment segment = editorSegments.get(segmentIndex);
        int count = wordCountForSegment(segment);
        int boundary = Math.max(0, Math.min(count, wordBoundary));
        if (boundary <= 0) return new SegmentSplit(segmentIndex - 1, segmentIndex);
        if (boundary >= count) return new SegmentSplit(segmentIndex, segmentIndex + 1);

        EditorSegment prefix;
        EditorSegment suffix;
        if (segment.words != null && !segment.words.isEmpty()) {
            double splitTime = splitTimeForWordBoundary(segment, boundary, -1);
            prefix = segmentFromWords(segment.words.subList(0, boundary), segment.speaker, segment.start, splitTime);
            suffix = segmentFromWords(segment.words.subList(boundary, segment.words.size()), segment.speaker, splitTime, segment.end);
        } else {
            int charBoundary = charBoundaryForWordBoundary(segment.text, boundary);
            int clippedBoundary = Math.max(0, Math.min(segment.text.length(), charBoundary));
            String prefixText = segment.text.substring(0, clippedBoundary).trim();
            String suffixText = segment.text.substring(clippedBoundary).trim();
            if (prefixText.isEmpty() || suffixText.isEmpty()) {
                return new SegmentSplit(segmentIndex, segmentIndex);
            }
            double splitTime = splitTimeForWordBoundary(segment, boundary, clippedBoundary);
            prefix = new EditorSegment(segment.start, splitTime, prefixText, segment.speaker);
            suffix = new EditorSegment(splitTime, segment.end, suffixText, segment.speaker);
        }
        editorSegments.remove(segmentIndex);
        editorSegments.add(segmentIndex, suffix);
        editorSegments.add(segmentIndex, prefix);
        return new SegmentSplit(segmentIndex, segmentIndex + 1);
    }

    private int selectedWordBoundaryBefore(int wordIndex) {
        return Math.max(0, wordIndex);
    }

    private int selectedWordBoundaryAfter(int wordIndex) {
        return Math.max(1, wordIndex + 1);
    }

    private int wordCountForSegment(EditorSegment segment) {
        if (segment.words != null && !segment.words.isEmpty()) return segment.words.size();
        String text = segment.text == null ? "" : segment.text;
        int count = 0;
        boolean inWord = false;
        for (int i = 0; i < text.length(); i++) {
            if (Character.isWhitespace(text.charAt(i))) {
                inWord = false;
            } else if (!inWord) {
                count++;
                inWord = true;
            }
        }
        return count;
    }

    private int charBoundaryForWordBoundary(String value, int boundary) {
        String text = value == null ? "" : value;
        if (boundary <= 0) return 0;
        int count = 0;
        boolean inWord = false;
        for (int i = 0; i < text.length(); i++) {
            if (Character.isWhitespace(text.charAt(i))) {
                inWord = false;
            } else if (!inWord) {
                if (count == boundary) return i;
                count++;
                inWord = true;
            }
        }
        return text.length();
    }

    private double splitTimeForWordBoundary(EditorSegment segment, int boundary, int charBoundary) {
        if (segment.words != null && boundary >= 0 && boundary < segment.words.size()) {
            double candidate = finiteOr(segment.words.get(boundary).start, Double.NaN);
            if (Double.isFinite(candidate) && candidate > segment.start && candidate < segment.end) {
                return candidate;
            }
        }
        double ratio;
        if (charBoundary >= 0) {
            ratio = Math.max(0.0, Math.min(1.0, charBoundary / (double) Math.max(1, segment.text.length())));
        } else {
            ratio = Math.max(0.0, Math.min(1.0, boundary / (double) Math.max(1, wordCountForSegment(segment))));
        }
        double start = finiteOr(segment.start, 0.0);
        double end = Math.max(start + 0.01, finiteOr(segment.end, start + 0.01));
        return Math.max(start + 0.001, Math.min(end - 0.001, start + (end - start) * ratio));
    }

    private EditorSegment segmentFromWords(List<TimedWord> words, int speaker) {
        List<TimedWord> copy = new ArrayList<>(words);
        StringBuilder text = new StringBuilder();
        double start = copy.isEmpty() ? 0.0 : finiteOr(copy.get(0).start, 0.0);
        double end = start;
        for (TimedWord word : copy) {
            if (text.length() > 0) text.append(' ');
            text.append(word.text);
            end = Math.max(end, finiteOr(word.end, finiteOr(word.start, end)));
        }
        return new EditorSegment(start, end, text.toString(), speaker, copy);
    }

    private EditorSegment segmentFromWords(List<TimedWord> words, int speaker, double start, double end) {
        List<TimedWord> copy = new ArrayList<>(words);
        StringBuilder text = new StringBuilder();
        for (TimedWord word : copy) {
            if (text.length() > 0) text.append(' ');
            text.append(word.text);
        }
        return new EditorSegment(start, Math.max(start + 0.01, end), text.toString(), speaker, copy);
    }

    private void showSegmentActions(int segmentIndex) {
        if (segmentIndex < 0 || segmentIndex >= editorSegments.size()) return;
        if (activeSegmentIndex != segmentIndex) {
            activeSegmentIndex = segmentIndex;
            activeWordIndex = -1;
        }
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
        single.setText("Chỉ từ/đoạn đang chọn");
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
                    int startIndex = segmentIndex;
                    int selectedWord = activeSegmentIndex == segmentIndex ? activeWordIndex : -1;
                    if (selectedWord >= 0) {
                        SegmentSplit split = splitSegmentAtWordBoundary(segmentIndex, selectedWordBoundaryBefore(selectedWord));
                        startIndex = split.afterIndex;
                    }
                    if (startIndex < 0 || startIndex >= editorSegments.size()) return;
                    int end = scope.getCheckedRadioButtonId() == toEnd.getId() ? findBlockEnd(startIndex) : startIndex;
                    for (int i = startIndex; i <= end; i++) editorSegments.get(i).speaker = targetSpeaker;
                    activeSegmentIndex = startIndex;
                    activeWordIndex = -1;
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
        int selectedWord = activeSegmentIndex == index ? activeWordIndex : -1;
        int from;
        int to;
        if (previous) {
            from = start;
            if (selectedWord >= 0) {
                SegmentSplit split = splitSegmentAtWordBoundary(index, selectedWordBoundaryAfter(selectedWord));
                to = split.beforeIndex;
            } else {
                to = index;
            }
        } else {
            if (selectedWord >= 0) {
                SegmentSplit split = splitSegmentAtWordBoundary(index, selectedWordBoundaryBefore(selectedWord));
                from = split.afterIndex;
                int blockEnd = from >= 0 && from < editorSegments.size() ? findBlockEnd(from) : from;
                int currentNeighbor = blockEnd + 1;
                if (currentNeighbor >= 0 && currentNeighbor < editorSegments.size()) {
                    targetSpeaker = editorSegments.get(currentNeighbor).speaker;
                }
                to = blockEnd;
            } else {
                from = index;
                to = end;
            }
        }
        if (from < 0 || to < from || from >= editorSegments.size()) return;
        to = Math.min(to, editorSegments.size() - 1);
        for (int i = from; i <= to; i++) editorSegments.get(i).speaker = targetSpeaker;
        activeSegmentIndex = from;
        activeWordIndex = -1;
        renderEditorTranscript();
        saveEditedResult();
    }

    private void copySegment(int index) {
        if (index < 0 || index >= editorSegments.size()) return;
        ClipboardManager clipboard = (ClipboardManager) getSystemService(CLIPBOARD_SERVICE);
        if (clipboard != null) {
            clipboard.setPrimaryClip(ClipData.newPlainText("ASR segment", formattedSegmentText(index)));
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

    private String formattedSegmentText(int index) {
        if (index < 0 || index >= editorSegments.size()) return "";
        EditorSegment segment = editorSegments.get(index);
        return speakerMeta(segment.speaker).name + ":\n"
                + cleanSegmentText(segment.text);
    }

    private String formattedTranscriptText() {
        if (editorSegments.isEmpty()) {
            return currentTranscriptText == null ? "" : currentTranscriptText.trim();
        }
        StringBuilder text = new StringBuilder();
        int start = 0;
        while (start < editorSegments.size()) {
            int speaker = editorSegments.get(start).speaker;
            StringBuilder blockText = new StringBuilder();
            int end = start;
            while (end < editorSegments.size() && editorSegments.get(end).speaker == speaker) {
                String segmentText = cleanSegmentText(editorSegments.get(end).text);
                if (!segmentText.isEmpty()) {
                    if (blockText.length() > 0) blockText.append(' ');
                    blockText.append(segmentText);
                }
                end++;
            }
            if (blockText.length() > 0) {
                if (text.length() > 0) text.append("\n\n");
                text.append(speakerMeta(speaker).name).append(':').append('\n');
                text.append(blockText);
            }
            start = end;
        }
        return text.toString().trim();
    }

    private String cleanSegmentText(String value) {
        return value == null ? "" : value.trim();
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
        if (currentQualityInfoJson != null) root.put("quality_info", new JSONObject(currentQualityInfoJson));
        if (currentTimingJson != null) root.put("timing", new JSONObject(currentTimingJson));
        if (currentAudioSummaryJson != null) root.put("audio_summary", new JSONArray(currentAudioSummaryJson));
        return root.toString();
    }

    private void copyTranscript() {
        String text = formattedTranscriptText();
        if (text == null || text.trim().isEmpty()) {
            appendLog("Chưa có văn bản để sao chép.");
            return;
        }
        ClipboardManager clipboard = (ClipboardManager) getSystemService(CLIPBOARD_SERVICE);
        if (clipboard != null) {
            clipboard.setPrimaryClip(ClipData.newPlainText("ASR transcript", text));
            appendLog("Đã sao chép văn bản.");
        }
    }

    private void setProgress(String phase, int percent) {
        int clipped = Math.max(0, Math.min(100, percent));
        if (progressContainer != null) {
            progressContainer.setVisibility(clipped >= 100 ? View.GONE : View.VISIBLE);
        }
        if (cancelProcessingButton != null) cancelProcessingButton.setVisibility(clipped >= 100 ? View.GONE : View.VISIBLE);
        progressBar.setProgress(clipped);
        stageView.setText(phase);
        progressText.setText(clipped + "%");
    }

    private void appendLog(String line) {
        String value = new SimpleDateFormat("HH:mm:ss", Locale.US).format(new Date()) + "  " + line;
        Log.i(TAG, value);
        if (logView != null) logView.append(value + "\n");
    }
}

