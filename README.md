# ASR VN Android

Native Android offline Vietnamese ASR app, ported from the ASR VN desktop/PWA pipeline.

The app is designed to run locally on-device: audio decode, VAD, ASR, punctuation, speaker diarization, and quality scoring are all executed in the Android app with ONNX Runtime.

## Features

- File-based Vietnamese ASR for common audio/video inputs.
- Zipformer Vietnamese ASR 68M, using separated ONNX encoder/decoder/joiner with a Java Pure ORT decoder.
- Silero VAD ONNX.
- Speaker diarization:
  - Senko CAM++ optimized path with CAM++ speaker embeddings and spectral clustering.
  - Pyannote Community-1 ONNX split path with segmentation, speaker embedding, PLDA/VBx-style postprocessing.
- ViBERT-CAPU FP32 ONNX punctuation and capitalization.
- DNSMOS audio quality score.
- Hotword boosting.
- Native Android transcript UI with speaker colors, speaker renaming, click-to-seek playback, and fixed player bar.
- Resume-after-kill checkpointing enabled by default.
- NNAPI acceleration is requested for the stages where it is useful: CAM++ embedding, Pyannote embedding, punctuation, and DNSMOS. CPU/XNNPACK fallback is used when NNAPI is unavailable.

## Technology

| Area | Implementation |
|---|---|
| Android | Native Android app, Java, Android SDK 35 |
| Build | Gradle Android plugin, NDK/CMake |
| Inference | ONNX Runtime Android |
| Audio decode | FFmpeg Kit first, Android MediaCodec/WAV fallback |
| ASR features | kaldi-native-fbank through a small JNI bridge |
| ASR | Zipformer RNN-T ONNX encoder/decoder/joiner |
| VAD | Silero VAD ONNX |
| Speaker diarization | CAM++ / Pyannote Community-1 ONNX, Java clustering/postprocess |
| Punctuation | ViBERT-CAPU ONNX FP32 |
| Audio quality | DNSMOS ONNX |

## Repository Layout

```text
app/                         Android application module
  src/main/java/              App UI and offline pipeline
  src/main/cpp/               JNI fbank bridge and bundled C++ deps
  src/main/res/               Android resources
  src/main/assets/models/     Local build output for bundled models; ignored by Git
gradle/                       Gradle wrapper
scripts/                      Local setup/cleanup scripts
docs/                         Model and third-party license notes
models/                       Local model cache; ignored by Git
model-manifest.json           Model file list, URLs, expected sizes
```

## GitHub Policy For Models

Do not commit model binaries. Several files are larger than GitHub's normal 100 MB file limit, especially:

- `vibert-capu.onnx` (~459 MB)
- `encoder-epoch-12-avg-8.onnx` (~261 MB)

The repository keeps only source code, scripts, and the model manifest. Local model files live in `models/` and are ignored by Git. See [docs/GETTING_MODELS.md](docs/GETTING_MODELS.md) for the complete model download flow.

Before building a full offline APK, copy or download models into the Android asset folder:

```powershell
.\scripts\sync-models.ps1
```

If the local `models/` cache is empty, download public model files:

```powershell
.\scripts\sync-models.ps1 -DownloadMissing
```

Some Pyannote files may require accepting the upstream Hugging Face terms. If needed:

```powershell
$env:HF_TOKEN = "<your token>"
.\scripts\sync-models.ps1 -DownloadMissing
```

If you already have the desktop/PWA checkout with models populated, reuse it:

```powershell
.\scripts\sync-models.ps1 -Source D:\App\asr-vn
```

`sync-models.ps1` validates model size and SHA-256 before copying files into the APK assets. The Pyannote `plda_prepared.npz` asset is generated locally from upstream `plda.npz` and `xvec_transform.npz` when it is not already present. Hugging Face credentials are only needed by builders when downloading gated models; end users who install a bundled APK do not need an account.

## Build

Prerequisites:

- Android Studio or Android SDK installed.
- Java 17. Android Studio's bundled JBR works locally, and GitHub Actions uses Temurin 17.
- SDK path in `local.properties`. You can copy `local.properties.example` and edit it:

```properties
sdk.dir=C\:\\Users\\<your-user>\\AppData\\Local\\Android\\Sdk
```

- NDK `30.0.14904198` installed, or update `app/build.gradle.kts` to the installed NDK version.

Build a full offline debug APK:

```powershell
cd D:\App\asr-vn-android
.\scripts\sync-models.ps1
.\gradlew.bat :app:assembleDebug
```

Output:

```text
app\build\outputs\apk\debug\app-debug.apk
```

## GitHub Actions APK

The repository includes `.github/workflows/android-apk.yml`.

- Every push or pull request to `main` builds a debug APK and uploads it as a workflow artifact.
- The default GitHub build is source-only: it verifies the Android project and creates an APK, but it does not bundle large model binaries because they are intentionally ignored by Git.
- To build a full offline APK on GitHub, run the workflow manually with `bundle_models=true`. Add a repository secret named `HF_TOKEN` first if the selected Hugging Face model files require accepted terms or authentication.

Download the APK from the workflow run's `Artifacts` section:

```text
asr-vn-android-v0.2.0-debug
```

For a release APK intended for distribution, add Android signing secrets and a separate signed release workflow. The current workflow produces an installable debug APK for testing.

Clean generated files:

```powershell
.\scripts\clean.ps1
```

Clean generated files and remove copied bundled model assets:

```powershell
.\scripts\clean.ps1 -RemoveBundledModels
```

## Model Sources

See [docs/MODELS.md](docs/MODELS.md) for model repositories, licenses, and distribution notes. See [docs/GETTING_MODELS.md](docs/GETTING_MODELS.md) for exact commands to download or reuse model files.

The Android project uses these main model families:

- `csukuangfj/sherpa-onnx-zipformer-vi-2025-04-20` for Vietnamese ASR 68M.
- `snakers4/silero-vad` for VAD.
- `welcomyou/campplus-3dspeaker-200k-onnx` for CAM++ speaker embedding.
- `welcomyou/pyannote-community-1-onnx-split` and `pyannote/speaker-diarization-community-1` for Pyannote Community-1.
- `welcomyou/vibert-capu-onnx` for punctuation/capitalization.
- Microsoft DNS Challenge DNSMOS for audio quality.

## License

Project code is released under the MIT License. See [LICENSE](LICENSE).

Third-party libraries and model files have their own licenses. See [docs/THIRD_PARTY.md](docs/THIRD_PARTY.md) and the upstream repositories before redistributing a model-bundled APK.
