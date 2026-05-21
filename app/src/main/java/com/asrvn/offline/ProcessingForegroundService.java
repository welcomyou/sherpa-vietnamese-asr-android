package com.asrvn.offline;

import android.app.Notification;
import android.app.NotificationChannel;
import android.app.NotificationManager;
import android.app.PendingIntent;
import android.app.Service;
import android.content.Intent;
import android.content.pm.ServiceInfo;
import android.net.Uri;
import android.os.Build;
import android.os.IBinder;
import android.os.SystemClock;
import android.util.Log;

import com.asrvn.offline.asr.PureOrtRecognizer;
import com.asrvn.offline.diarization.DiarizationResult;
import com.asrvn.offline.pipeline.NativeOfflinePipeline;
import com.asrvn.offline.storage.NativeFileLibrary;

public final class ProcessingForegroundService extends Service {
    public static final String ACTION_START = "com.asrvn.offline.action.PROCESS_START";
    public static final String ACTION_RESUME = "com.asrvn.offline.action.PROCESS_RESUME";
    public static final String ACTION_CANCEL = "com.asrvn.offline.action.PROCESS_CANCEL";
    public static final String ACTION_PROGRESS = "com.asrvn.offline.action.PROCESS_PROGRESS";
    public static final String ACTION_COMPLETE = "com.asrvn.offline.action.PROCESS_COMPLETE";
    public static final String ACTION_ERROR = "com.asrvn.offline.action.PROCESS_ERROR";
    public static final String ACTION_CANCELLED = "com.asrvn.offline.action.PROCESS_CANCELLED";

    public static final String EXTRA_DISPLAY_NAME = "display_name";
    public static final String EXTRA_PHASE = "phase";
    public static final String EXTRA_PERCENT = "percent";
    public static final String EXTRA_MESSAGE = "message";
    public static final String EXTRA_ITEM_ID = "item_id";
    public static final String EXTRA_RESULT_JSON = "result_json";

    private static final String TAG = "ASRVN";
    private static final String CHANNEL_ID = "asrvn_processing";
    private static final int NOTIFICATION_ID = 5107;

    private static volatile boolean processing;

    private NativeOfflinePipeline pipeline;
    private NativeFileLibrary library;
    private NotificationManager notificationManager;
    private long lastNotificationAtMs;
    private int lastNotificationPercent = -1;
    private volatile String currentItemId;
    private volatile boolean cancellationHandled;

    public static boolean isProcessing() {
        return processing;
    }

    @Override
    public void onCreate() {
        super.onCreate();
        pipeline = new NativeOfflinePipeline(this);
        library = new NativeFileLibrary(this);
        notificationManager = (NotificationManager) getSystemService(NOTIFICATION_SERVICE);
        ensureNotificationChannel();
    }

    @Override
    public int onStartCommand(Intent intent, int flags, int startId) {
        if (intent == null) {
            stopSelf(startId);
            return START_NOT_STICKY;
        }
        String action = intent.getAction();
        if (ACTION_CANCEL.equals(action)) {
            cancelProcessing("Người dùng hủy xử lý.");
            return START_NOT_STICKY;
        }
        if (processing) {
            broadcastProgress("Đang xử lý", 1, "Pipeline đang chạy.");
            return START_NOT_STICKY;
        }

        if (ACTION_RESUME.equals(action)) {
            String itemId = intent.getStringExtra(EXTRA_ITEM_ID);
            if (itemId == null || itemId.trim().isEmpty()) {
                finishWithError(new IllegalArgumentException("Missing library item id."));
                return START_NOT_STICKY;
            }
            processing = true;
            cancellationHandled = false;
            currentItemId = itemId;
            startAsForeground("Resume", 3, "Tiếp tục xử lý");
            pipeline.resumeLibraryItem(itemId, serviceListener());
            return START_NOT_STICKY;
        }

        if (!ACTION_START.equals(action)) {
            stopSelf(startId);
            return START_NOT_STICKY;
        }

        Uri uri = intent.getData();
        String displayName = intent.getStringExtra(EXTRA_DISPLAY_NAME);
        if (uri == null) {
            finishWithError(new IllegalArgumentException("Missing input file URI."));
            return START_NOT_STICKY;
        }

        processing = true;
        cancellationHandled = false;
        currentItemId = null;
        startAsForeground("Reading input", 1, "Chuẩn bị xử lý nền");
        pipeline.importFile(uri, displayName, serviceListener());
        return START_NOT_STICKY;
    }

    private NativeOfflinePipeline.ProgressListener serviceListener() {
        return new NativeOfflinePipeline.ProgressListener() {
            @Override
            public void onItemCreated(NativeFileLibrary.LibraryItem item) {
                currentItemId = item == null ? null : item.id;
                if (item != null) broadcastProgress("Library", 5, "Stored " + item.displayName);
            }

            @Override
            public void onProgress(String phase, int percent, String message) {
                updateProgress(phase, percent, message);
            }

            @Override
            public void onComplete(
                    NativeFileLibrary.LibraryItem item,
                    String resultJson,
                    PureOrtRecognizer.DecodeResult asr,
                    DiarizationResult diarization
            ) {
                updateNotification("Hoàn thành", 100, item.displayName, true, false);
                Intent out = baseBroadcast(ACTION_COMPLETE)
                        .putExtra(EXTRA_ITEM_ID, item.id)
                        .putExtra(EXTRA_RESULT_JSON, resultJson)
                        .putExtra(EXTRA_DISPLAY_NAME, item.displayName);
                sendBroadcast(out);
                processing = false;
                currentItemId = null;
                if (Build.VERSION.SDK_INT >= 24) stopForeground(STOP_FOREGROUND_DETACH);
                else stopForeground(false);
                stopSelf();
            }

            @Override
            public void onCancelled(NativeFileLibrary.LibraryItem item) {
                if (item != null) currentItemId = item.id;
                finishCancelled("Đã hủy xử lý.");
            }

            @Override
            public void onError(Throwable error) {
                finishWithError(error);
            }
        };
    }

    @Override
    public IBinder onBind(Intent intent) {
        return null;
    }

    @Override
    public void onDestroy() {
        if (processing) {
            pipeline.cancel();
        }
        processing = false;
        super.onDestroy();
    }

    @Override
    public void onTaskRemoved(Intent rootIntent) {
        cancelProcessing("Ứng dụng bị đóng.");
        super.onTaskRemoved(rootIntent);
    }

    @Override
    public void onTimeout(int startId, int fgsType) {
        finishWithError(new IllegalStateException("Android foreground-service timeout reached."));
        stopSelf(startId);
    }

    private void startAsForeground(String phase, int percent, String message) {
        Notification notification = buildNotification(phase, percent, message, false, false);
        if (Build.VERSION.SDK_INT >= 35) {
            startForeground(NOTIFICATION_ID, notification, ServiceInfo.FOREGROUND_SERVICE_TYPE_MEDIA_PROCESSING);
        } else {
            startForeground(NOTIFICATION_ID, notification);
        }
    }

    private void updateProgress(String phase, int percent, String message) {
        broadcastProgress(phase, percent, message);
        long now = SystemClock.elapsedRealtime();
        int clipped = Math.max(0, Math.min(100, percent));
        if (clipped == lastNotificationPercent && now - lastNotificationAtMs < 1200) return;
        updateNotification(phase, clipped, message, clipped >= 100, false);
    }

    private void updateNotification(String phase, int percent, String message, boolean done, boolean error) {
        lastNotificationAtMs = SystemClock.elapsedRealtime();
        lastNotificationPercent = Math.max(0, Math.min(100, percent));
        notificationManager.notify(NOTIFICATION_ID, buildNotification(phase, lastNotificationPercent, message, done, error));
    }

    private Notification buildNotification(String phase, int percent, String message, boolean done, boolean error) {
        Intent open = new Intent(this, MainActivity.class)
                .addFlags(Intent.FLAG_ACTIVITY_CLEAR_TOP | Intent.FLAG_ACTIVITY_SINGLE_TOP);
        PendingIntent contentIntent = PendingIntent.getActivity(
                this,
                0,
                open,
                PendingIntent.FLAG_UPDATE_CURRENT | PendingIntent.FLAG_IMMUTABLE);

        String title = done ? "Sherpa Vietnamese ASR hoàn thành" : "Sherpa Vietnamese ASR đang xử lý";
        if ("Đã hủy".equals(phase)) title = "Sherpa Vietnamese ASR đã hủy";
        if (error) title = "Sherpa Vietnamese ASR lỗi xử lý";
        String text = (phase == null ? "" : phase) + (message == null || message.isEmpty() ? "" : ": " + message);

        Notification.Builder builder = Build.VERSION.SDK_INT >= 26
                ? new Notification.Builder(this, CHANNEL_ID)
                : new Notification.Builder(this);
        builder.setSmallIcon(R.drawable.ic_stat_asr)
                .setContentTitle(title)
                .setContentText(text)
                .setStyle(new Notification.BigTextStyle().bigText(text))
                .setContentIntent(contentIntent)
                .setOngoing(!done && !error)
                .setOnlyAlertOnce(true)
                .setShowWhen(false)
                .setCategory(Notification.CATEGORY_PROGRESS);
        if (!done && !error) {
            builder.setProgress(100, Math.max(0, Math.min(100, percent)), false);
        } else {
            builder.setProgress(0, 0, false).setAutoCancel(true);
        }
        return builder.build();
    }

    private void finishWithError(Throwable error) {
        Log.e(TAG, "Foreground pipeline failed", error);
        String message = error == null || error.getMessage() == null ? "Unknown error" : error.getMessage();
        updateNotification("Pipeline failed", 100, message, false, true);
        sendBroadcast(baseBroadcast(ACTION_ERROR).putExtra(EXTRA_MESSAGE, message));
        processing = false;
        currentItemId = null;
        if (Build.VERSION.SDK_INT >= 24) stopForeground(STOP_FOREGROUND_DETACH);
        else stopForeground(false);
        stopSelf();
    }

    private void cancelProcessing(String message) {
        pipeline.cancel();
        if (currentItemId != null) {
            try {
                library.markCancelled(currentItemId);
            } catch (Exception ignored) {
            }
        }
        finishCancelled(message);
    }

    private void finishCancelled(String message) {
        if (cancellationHandled) return;
        cancellationHandled = true;
        updateNotification("Đã hủy", 100, message, true, false);
        sendBroadcast(baseBroadcast(ACTION_CANCELLED)
                .putExtra(EXTRA_ITEM_ID, currentItemId == null ? "" : currentItemId)
                .putExtra(EXTRA_MESSAGE, message));
        processing = false;
        currentItemId = null;
        if (Build.VERSION.SDK_INT >= 24) stopForeground(STOP_FOREGROUND_REMOVE);
        else stopForeground(true);
        stopSelf();
    }

    private void broadcastProgress(String phase, int percent, String message) {
        sendBroadcast(baseBroadcast(ACTION_PROGRESS)
                .putExtra(EXTRA_PHASE, phase)
                .putExtra(EXTRA_PERCENT, percent)
                .putExtra(EXTRA_ITEM_ID, currentItemId == null ? "" : currentItemId)
                .putExtra(EXTRA_MESSAGE, message));
    }

    private Intent baseBroadcast(String action) {
        return new Intent(action).setPackage(getPackageName());
    }

    private void ensureNotificationChannel() {
        if (Build.VERSION.SDK_INT < 26) return;
        NotificationChannel channel = new NotificationChannel(
                CHANNEL_ID,
                "Xử lý ASR",
                NotificationManager.IMPORTANCE_LOW);
        channel.setDescription("Hiển thị tiến độ xử lý audio/video nền.");
        channel.setShowBadge(false);
        notificationManager.createNotificationChannel(channel);
    }
}
