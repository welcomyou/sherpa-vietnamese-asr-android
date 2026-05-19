cmake_minimum_required(VERSION 3.22.1)

project(asrvn_fbank LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)
set(KALDI_NATIVE_FBANK_ENABLE_CHECK OFF CACHE BOOL "" FORCE)
set(KALDI_NATIVE_FBANK_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(KALDI_NATIVE_FBANK_BUILD_PYTHON OFF CACHE BOOL "" FORCE)
set(KISSFFT_PKGCONFIG OFF CACHE BOOL "" FORCE)
set(KISSFFT_STATIC ON CACHE BOOL "" FORCE)
set(KISSFFT_TEST OFF CACHE BOOL "" FORCE)
set(KISSFFT_TOOLS OFF CACHE BOOL "" FORCE)

add_subdirectory(third_party/kissfft EXCLUDE_FROM_ALL)
add_subdirectory(kaldi-native-fbank/csrc)

add_library(asrvn_fbank SHARED native-fbank-jni.cc)
target_include_directories(asrvn_fbank PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(asrvn_fbank PRIVATE kaldi-native-fbank-core log)
