package com.asrvn.offline.diarization;

public final class FbankDebugMain {
    public static void main(String[] args) {
        int samplesCount = 16000 * 3 + 123;
        float[] samples = new float[samplesCount];
        for (int i = 0; i < samples.length; i++) {
            double t = i / 16000.0;
            samples[i] = (float) (0.07 * Math.sin(2.0 * Math.PI * 233.0 * t)
                    + 0.03 * Math.cos(2.0 * Math.PI * 701.0 * t)
                    + 0.01 * Math.sin(2.0 * Math.PI * 1234.5 * t));
        }
        CamppFbankExtractor.Features campp = new CamppFbankExtractor().compute(samples);
        WespeakerFbankExtractor.Features wespeaker = new WespeakerFbankExtractor().compute(samples);
        print("campp", campp.data, campp.frames);
        print("wespeaker", wespeaker.data, wespeaker.frames);
    }

    private static void print(String name, float[] data, int frames) {
        double sum = 0.0;
        double abs = 0.0;
        double max = Double.NEGATIVE_INFINITY;
        double min = Double.POSITIVE_INFINITY;
        for (float value : data) {
            sum += value;
            abs += Math.abs(value);
            max = Math.max(max, value);
            min = Math.min(min, value);
        }
        System.out.println(name + ".frames=" + frames);
        System.out.println(name + ".sum=" + sum);
        System.out.println(name + ".abs=" + abs);
        System.out.println(name + ".min=" + min);
        System.out.println(name + ".max=" + max);
        StringBuilder first = new StringBuilder();
        int n = Math.min(data.length, 80);
        for (int i = 0; i < n; i++) {
            if (i > 0) first.append(',');
            first.append(data[i]);
        }
        System.out.println(name + ".first80=" + first);
    }
}
