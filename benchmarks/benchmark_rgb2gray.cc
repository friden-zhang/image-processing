#include "image-processing/color-convert/kernels/rgb2gray.hpp"
#include <array>
#include <benchmark/benchmark.h>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <vector>

constexpr int width = 1920 * 2;
constexpr int height = 1080 * 2;
constexpr int pixel_count = width * height;

std::vector<unsigned char> read_raw_image(const std::string &filename,
                                          int width, int height) {

  size_t size = width * height * 3;

  std::vector<unsigned char> data(size);

  std::ifstream file(filename, std::ios::binary);
  if (!file) {
    throw std::runtime_error("Failed to open file: " + filename);
  }

  file.read(reinterpret_cast<char *>(data.data()), size);
  if (!file) {
    throw std::runtime_error("Failed to read data from file: " + filename);
  }

  return data;
}

static void BenchmarkRGB2GrayNative(benchmark::State &state) {

  std::vector<unsigned char> input_image(pixel_count * 3, 128);
  std::vector<unsigned char> output_image(pixel_count, 0);

  for (auto _ : state) {
    image_processing::color_convert::kernels::rgb_2_gray(
        input_image.data(), output_image.data(), width, height,
        image_processing::color_convert::AlgoType::kNativeCpu);
  }
}

static void BenchmarkRGB2GrayParallel(benchmark::State &state) {

  std::vector<unsigned char> input_image(pixel_count * 3, 128);
  std::vector<unsigned char> output_image(pixel_count, 0);

  for (auto _ : state) {
    image_processing::color_convert::kernels::rgb_2_gray(
        input_image.data(), output_image.data(), width, height,
        image_processing::color_convert::AlgoType::kParallelCpu);
  }
}

static void BenchmarkRGB2GraySimd(benchmark::State &state) {

  unsigned char *input_image_aligned = reinterpret_cast<unsigned char *>(
      std::aligned_alloc(16, pixel_count * 3));

  for (int i = 0; i < pixel_count * 3; i++) {
    input_image_aligned[i] = 128;
  }

  unsigned char *output_image_aligned =
      reinterpret_cast<unsigned char *>(std::aligned_alloc(16, pixel_count));

  for (auto _ : state) {
    image_processing::color_convert::kernels::rgb_2_gray(
        input_image_aligned, output_image_aligned, width, height,
        image_processing::color_convert::AlgoType::kSimdCpu);
  }
}

BENCHMARK(BenchmarkRGB2GrayNative);
BENCHMARK(BenchmarkRGB2GrayParallel);
BENCHMARK(BenchmarkRGB2GraySimd);
