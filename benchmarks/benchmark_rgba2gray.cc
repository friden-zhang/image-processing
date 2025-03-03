#include "image-processing/color-convert/kernels/rgba2gray.hpp"
#include <array>
#include <benchmark/benchmark.h>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <vector>

constexpr int width = 1920 * 2;
constexpr int height = 1080 * 2;
constexpr int pixel_count = width * height;

template <image_processing::color_convert::AlgoType algo_type,
          image_processing::color_convert::MemLayout mem_layout>
static void BenchmarkRGBA2Gray(benchmark::State &state) {

  std::vector<unsigned char> input_image(pixel_count * 4, 128);
  std::vector<unsigned char> output_image(pixel_count, 0);

  for (auto _ : state) {
    image_processing::color_convert::kernels::rgba_2_gray(
        input_image.data(), output_image.data(), width, height, algo_type,
        mem_layout);
  }
}

BENCHMARK(
    BenchmarkRGBA2Gray<image_processing::color_convert::AlgoType::kNativeCpu,
                       image_processing::color_convert::MemLayout::Packed>);
BENCHMARK(
    BenchmarkRGBA2Gray<image_processing::color_convert::AlgoType::kParallelCpu,
                       image_processing::color_convert::MemLayout::Packed>);
BENCHMARK(
    BenchmarkRGBA2Gray<image_processing::color_convert::AlgoType::kSimdCpu,
                       image_processing::color_convert::MemLayout::Packed>);
#if HAS_CUDA

BENCHMARK(
    BenchmarkRGBA2Gray<image_processing::color_convert::AlgoType::kCuda,
                       image_processing::color_convert::MemLayout::Packed>);
#endif // HAS_CUDA

BENCHMARK(
    BenchmarkRGBA2Gray<image_processing::color_convert::AlgoType::kNativeCpu,
                       image_processing::color_convert::MemLayout::Planar>);
BENCHMARK(
    BenchmarkRGBA2Gray<image_processing::color_convert::AlgoType::kParallelCpu,
                       image_processing::color_convert::MemLayout::Planar>);
BENCHMARK(
    BenchmarkRGBA2Gray<image_processing::color_convert::AlgoType::kSimdCpu,
                       image_processing::color_convert::MemLayout::Planar>);
#if HAS_CUDA
BENCHMARK(
    BenchmarkRGBA2Gray<image_processing::color_convert::AlgoType::kCuda,
                       image_processing::color_convert::MemLayout::Planar>);

#endif // HAS_CUDA