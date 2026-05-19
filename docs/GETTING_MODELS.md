# Getting Models

This repository does not commit model binaries. The source of truth is
[`model-manifest.json`](../model-manifest.json): every required file has an id,
download source, expected size, and SHA-256 where available.

## Fresh Clone

From a clean clone, download the model files into the local `models/` cache and
copy them into Android assets:

```powershell
cd D:\App\asr-vn-android
.\scripts\sync-models.ps1 -DownloadMissing
```

Then build the APK:

```powershell
.\gradlew.bat :app:assembleDebug
```

The generated APK is:

```text
app\build\outputs\apk\debug\app-debug.apk
```

## Copy From An Existing ASR VN Checkout

If you already have the desktop/PWA repository with `models/` populated, reuse
that cache instead of downloading again:

```powershell
cd D:\App\asr-vn-android
.\scripts\sync-models.ps1 -Source D:\App\asr-vn
```

You can also pass a direct model-cache folder:

```powershell
.\scripts\sync-models.ps1 -Source D:\App\asr-vn\models
```

## Hugging Face Access

Most files are public. Pyannote assets can require accepting upstream Hugging
Face terms before they can be downloaded.

Create a read token at:

```text
https://huggingface.co/settings/tokens
```

Accept the required model terms in the browser:

```text
https://huggingface.co/pyannote/speaker-diarization-community-1
https://huggingface.co/pyannote/segmentation-3.0
https://huggingface.co/pyannote/wespeaker-voxceleb-resnet34-LM
```

Then rerun the sync script with `HF_TOKEN`:

```powershell
$env:HF_TOKEN = "<your read token>"
.\scripts\sync-models.ps1 -DownloadMissing
```

The token is needed only by the person building from source. Users who install
a model-bundled APK do not need a Hugging Face account.

## Generated Pyannote Asset

`pyannote/speaker-diarization-community-1` provides `plda.npz` and
`xvec_transform.npz`. The Android runtime also needs `plda_prepared.npz`, which
is generated locally from those two upstream files:

```powershell
python -m pip install numpy scipy
.\scripts\sync-models.ps1 -DownloadMissing
```

If `plda_prepared.npz` already exists in your local model cache, the sync script
uses it directly. Otherwise it runs `scripts/prepare-pyannote-plda.py`.

## Integrity Checks

`sync-models.ps1` verifies file size and SHA-256 before copying a model into
`app/src/main/assets/models`. The only exception is the locally generated
`plda_prepared.npz`, whose zip container bytes can differ between Python/NumPy
versions while the arrays remain equivalent. If a download changes upstream or
is corrupted, the script fails instead of building a broken APK.

## GitHub Upload Notes

Keep these paths out of Git commits:

```text
models/
app/src/main/assets/models/
```

They are ignored by `.gitignore` because some files are larger than GitHub's
normal file size limit. Use normal Git commands (`git add .`) when publishing
the repository. Do not upload the whole local folder through the GitHub web UI,
because your local ignored model cache still exists on disk.
