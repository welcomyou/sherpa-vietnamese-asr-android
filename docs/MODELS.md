# Models

Model binaries are not committed to this repository. They are listed in `model-manifest.json` and can be copied or downloaded into `app/src/main/assets/models` with:

```powershell
.\scripts\sync-models.ps1
```

Use `.\scripts\sync-models.ps1 -DownloadMissing` to fetch missing files where public downloads are available.

For a fresh clone, Hugging Face token setup, and reuse of an existing desktop/PWA model cache, see [GETTING_MODELS.md](GETTING_MODELS.md).

## Required Runtime Models

| Component | Files | Source | License / notes |
|---|---|---|---|
| Vietnamese ASR 68M | encoder, decoder, joiner, tokens, BPE files | `csukuangfj/sherpa-onnx-zipformer-vi-2025-04-20` | See upstream model card |
| VAD | `silero_vad_16k_op15.onnx` | `snakers4/silero-vad` | See upstream model card |
| CAM++ speaker embedding | `campplus_cn_en_common_200k.onnx` | `welcomyou/campplus-3dspeaker-200k-onnx` / 3D-Speaker | Apache-2.0 per upstream README |
| Pyannote segmentation/embedding split | `segmentation-community-1.onnx`, `embedding_encoder.onnx`, projection weights | `welcomyou/pyannote-community-1-onnx-split` | CC-BY-4.0 per original project README |
| Pyannote PLDA/VBx assets | `plda.npz`, `xvec_transform.npz`; `plda_prepared.npz` is generated locally | `pyannote/speaker-diarization-community-1` | Check upstream model terms; may require Hugging Face access acceptance |
| Punctuation/capitalization | `vibert-capu.onnx`, vocab and labels | `welcomyou/vibert-capu-onnx` / `dragonSwing/vibert-capu` | CC-BY-SA-4.0 per original project README |
| Audio quality | `sig_bak_ovr.onnx` | Microsoft DNS Challenge DNSMOS | See Microsoft DNS-Challenge repository |

## Distribution Notes

- Keep model binaries out of normal Git commits.
- For private distribution, build the APK after syncing local models into `app/src/main/assets/models`.
- For public distribution, review each model's license and model card before bundling it in an APK.
- Pyannote assets may require accepting upstream terms on Hugging Face. Do not assume they can be mirrored publicly without checking the model card.
- The sync script verifies file size and SHA-256 values from `model-manifest.json` before bundling downloaded models.
- `plda_prepared.npz` is derived locally from upstream Pyannote PLDA files with `scripts/prepare-pyannote-plda.py`.
