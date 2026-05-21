plugins {
    id("com.android.application")
}

android {
    namespace = "com.asrvn.offline"
    compileSdk = 35
    ndkVersion = "30.0.14904198"

    defaultConfig {
        applicationId = "com.asrvn.offline"
        minSdk = 26
        targetSdk = 35
        versionCode = 2
        versionName = "0.2.0"

        ndk {
            abiFilters += listOf("arm64-v8a", "armeabi-v7a", "x86_64")
        }
    }

    externalNativeBuild {
        cmake {
            path = file("src/main/cpp/CMakeLists.txt")
        }
    }

    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }

    packaging {
        jniLibs {
            pickFirsts += listOf("**/libc++_shared.so")
        }
    }
}

tasks.withType<JavaCompile>().configureEach {
    options.encoding = "UTF-8"
}

dependencies {
    implementation("com.microsoft.onnxruntime:onnxruntime-android:1.25.1")
    implementation("org.apache.commons:commons-math3:3.6.1")
    implementation("com.moizhassan.ffmpeg:ffmpeg-kit-16kb:6.1.1")
}
