package com.asrvn.offline.pipeline;

import android.content.Context;
import android.os.PowerManager;

public final class ScreenWakeLock {
    private final PowerManager.WakeLock wakeLock;

    public ScreenWakeLock(Context context) {
        PowerManager manager = (PowerManager) context.getApplicationContext().getSystemService(Context.POWER_SERVICE);
        wakeLock = manager.newWakeLock(PowerManager.SCREEN_BRIGHT_WAKE_LOCK | PowerManager.ON_AFTER_RELEASE, "AsrVnNative:Processing");
        wakeLock.setReferenceCounted(false);
    }

    public void acquire() {
        if (!wakeLock.isHeld()) wakeLock.acquire(6 * 60 * 60 * 1000L);
    }

    public void release() {
        if (wakeLock.isHeld()) wakeLock.release();
    }
}
