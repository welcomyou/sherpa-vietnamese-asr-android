# Third-Party Notices

This file is a practical checklist for distribution review. It is not legal advice.

## Direct Android Dependencies

| Dependency | Use | License / notes |
|---|---|---|
| Android SDK / Gradle Android Plugin | Android build/runtime APIs | Android SDK terms / Apache-2.0 components |
| ONNX Runtime Android | ONNX inference | MIT License |
| Apache Commons Math | Matrix/eigen math for diarization | Apache-2.0 |
| FFmpeg Kit / FFmpeg | Audio/video decode path | FFmpeg licensing depends on exact binary configuration. Verify LGPL/GPL implications before public APK distribution. |
| kaldi-native-fbank | ASR feature extraction C++ implementation | Check upstream license before redistribution |
| KissFFT | FFT dependency used by fbank code | BSD-3-Clause |

## Model Families

See [MODELS.md](MODELS.md) and `model-manifest.json`.

## Project License

The app source code in this repository is MIT licensed. Third-party libraries and model files remain under their own licenses.
