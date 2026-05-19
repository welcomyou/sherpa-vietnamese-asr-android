#include <jni.h>

#include <algorithm>
#include <cstdint>
#include <string>
#include <vector>

#include "kaldi-native-fbank/csrc/online-feature.h"

namespace {

constexpr float kSampleRate = 16000.0f;
constexpr int32_t kNumMelBins = 80;

enum class FbankKind {
  kAsr,
  kCampp,
  kWespeaker,
};

knf::FbankOptions MakeOptions(FbankKind kind) {
  knf::FbankOptions opts;
  opts.frame_opts.dither = 0.0f;
  opts.frame_opts.samp_freq = kSampleRate;
  opts.frame_opts.frame_length_ms = 25.0f;
  opts.frame_opts.frame_shift_ms = 10.0f;
  opts.frame_opts.preemph_coeff = 0.97f;
  opts.frame_opts.remove_dc_offset = true;
  opts.frame_opts.round_to_power_of_two = true;
  opts.mel_opts.num_bins = kNumMelBins;
  opts.mel_opts.low_freq = 20.0f;
  opts.use_energy = false;
  opts.use_log_fbank = true;
  opts.use_power = true;

  switch (kind) {
    case FbankKind::kAsr:
      opts.frame_opts.snip_edges = false;
      opts.frame_opts.window_type = "povey";
      opts.mel_opts.high_freq = 7600.0f;
      opts.energy_floor = 1.0f;
      break;
    case FbankKind::kCampp:
      opts.frame_opts.snip_edges = true;
      opts.frame_opts.window_type = "povey";
      opts.mel_opts.high_freq = 0.0f;
      opts.energy_floor = 1.0f;
      break;
    case FbankKind::kWespeaker:
      opts.frame_opts.snip_edges = true;
      opts.frame_opts.window_type = "hamming";
      opts.mel_opts.high_freq = 0.0f;
      opts.energy_floor = 0.0f;
      break;
  }
  return opts;
}

jobject MakeResult(JNIEnv *env, const std::vector<float> &features,
                   int32_t frames) {
  jfloatArray data = env->NewFloatArray(static_cast<jsize>(features.size()));
  if (data == nullptr) return nullptr;
  if (!features.empty()) {
    env->SetFloatArrayRegion(data, 0, static_cast<jsize>(features.size()),
                             features.data());
  }

  jclass result_class = env->FindClass("com/asrvn/offline/fbank/NativeFbank$Result");
  if (result_class == nullptr) return nullptr;
  jmethodID ctor = env->GetMethodID(result_class, "<init>", "([FI)V");
  if (ctor == nullptr) return nullptr;
  return env->NewObject(result_class, ctor, data, static_cast<jint>(frames));
}

jobject Compute(JNIEnv *env, jfloatArray samples_array, FbankKind kind,
                bool scale_audio, bool floor_log_to_zero, bool cmvn) {
  if (samples_array == nullptr) {
    return MakeResult(env, std::vector<float>(), 0);
  }

  jsize sample_count = env->GetArrayLength(samples_array);
  if (sample_count <= 0) {
    return MakeResult(env, std::vector<float>(), 0);
  }

  std::vector<float> samples(static_cast<size_t>(sample_count));
  env->GetFloatArrayRegion(samples_array, 0, sample_count, samples.data());
  if (env->ExceptionCheck()) return nullptr;

  if (scale_audio) {
    for (float &sample : samples) sample *= 32768.0f;
  }

  knf::OnlineFbank fbank(MakeOptions(kind));
  fbank.AcceptWaveform(kSampleRate, samples.data(), sample_count);
  fbank.InputFinished();

  int32_t frames = fbank.NumFramesReady();
  if (frames <= 0) {
    return MakeResult(env, std::vector<float>(), 0);
  }

  std::vector<float> features(static_cast<size_t>(frames) * kNumMelBins);
  for (int32_t frame = 0; frame < frames; ++frame) {
    const float *src = fbank.GetFrame(frame);
    std::copy(src, src + kNumMelBins,
              features.begin() + static_cast<size_t>(frame) * kNumMelBins);
  }

  if (floor_log_to_zero) {
    for (float &feature : features) feature = std::max(feature, 0.0f);
  }

  if (cmvn) {
    float means[kNumMelBins] = {};
    for (int32_t frame = 0; frame < frames; ++frame) {
      const size_t offset = static_cast<size_t>(frame) * kNumMelBins;
      for (int32_t mel = 0; mel < kNumMelBins; ++mel) {
        means[mel] += features[offset + mel];
      }
    }
    for (float &mean : means) mean /= static_cast<float>(frames);
    for (int32_t frame = 0; frame < frames; ++frame) {
      const size_t offset = static_cast<size_t>(frame) * kNumMelBins;
      for (int32_t mel = 0; mel < kNumMelBins; ++mel) {
        features[offset + mel] -= means[mel];
      }
    }
  }

  return MakeResult(env, features, frames);
}

}  // namespace

extern "C" JNIEXPORT jobject JNICALL
Java_com_asrvn_offline_fbank_NativeFbank_computeAsr(JNIEnv *env, jclass,
                                                    jfloatArray samples) {
  return Compute(env, samples, FbankKind::kAsr, false, false, false);
}

extern "C" JNIEXPORT jobject JNICALL
Java_com_asrvn_offline_fbank_NativeFbank_computeCampp(JNIEnv *env, jclass,
                                                      jfloatArray samples) {
  return Compute(env, samples, FbankKind::kCampp, true, true, true);
}

extern "C" JNIEXPORT jobject JNICALL
Java_com_asrvn_offline_fbank_NativeFbank_computeWespeaker(JNIEnv *env, jclass,
                                                          jfloatArray samples) {
  return Compute(env, samples, FbankKind::kWespeaker, true, false, true);
}
