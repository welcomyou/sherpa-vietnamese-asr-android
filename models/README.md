# Local Model Cache

This directory is intentionally ignored by Git.

Place or download model files here, then run:

```powershell
.\scripts\sync-models.ps1
```

The script copies files into `app/src/main/assets/models` for a full offline APK build. Model binaries are large and some exceed GitHub's normal file size limit, so they should not be committed to this repository.
