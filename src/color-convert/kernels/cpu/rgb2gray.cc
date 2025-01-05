#include "rgb2gray.hpp"
#include "image-processing/color-convert/common/vector-type.hpp"
#include <array>
#include <assert.h>
#include <iostream>
#include <stddef.h>
#include <tbb/blocked_range.h>
#include <tbb/parallel_for.h>
#ifdef __ARM_NEON__
#include <arm_neon.h>
#elif defined(__AVX__)
#include <immintrin.h>
#endif

#include <experimental/simd>

namespace image_processing {

namespace color_convert {

namespace kernels {

bool rgb_packed_2_gray_native(const unsigned char *input, unsigned char *output,
                              int width, int height) {

  int pixel_count = width * height;
  for (int i = 0; i < pixel_count; ++i) {
    // Each pixel in RGB is represented by three consecutive bytes: R, G, B
    unsigned char r = input[i * 3];     // Red
    unsigned char g = input[i * 3 + 1]; // Green
    unsigned char b = input[i * 3 + 2]; // Blue

    // Convert to grayscale using the luminosity method
    unsigned char gray = (unsigned char)(0.299 * r + 0.587 * g + 0.114 * b);

    output[i] = gray; // Set the output pixel to the grayscale value
  }
  return true; // Successful conversion
}

bool rgb_packed_2_gray_parallel(const unsigned char *input,
                                unsigned char *output, int width, int height) {

  const auto *input_vec = reinterpret_cast<const uchar3 *>(input);
  int pixel_count = width * height;

  tbb::parallel_for(tbb::blocked_range<int>(0, pixel_count),
                    [&](const tbb::blocked_range<int> &range) {
                      for (int i = range.begin(); i != range.end(); ++i) {
                        // Each pixel in RGB is represented by three consecutive
                        // bytes: R, G, B
                        unsigned char r = input_vec[i].x; // Red
                        unsigned char g = input_vec[i].y; // Green
                        unsigned char b = input_vec[i].z; // Blue
                        // Convert to grayscale using the luminosity method
                        unsigned char gray =
                            (unsigned char)(0.299 * r + 0.587 * g + 0.114 * b);
                        output[i] = gray;
                      }
                    });
  return true;
}

// use std::experimental::simd to load packed data, the performance is bad than planner version
bool rgb_packed_2_gray_simd(const unsigned char *input, unsigned char *output,
                            int width, int height) {

  int pixel_count = width * height;

  namespace stdx = std::experimental;

  using simd_t = stdx::native_simd<uint16_t>;
  constexpr auto step = simd_t::size();

  using fixed_size_simd_t = stdx::fixed_size_simd<uint8_t, step>;

  size_t tile = pixel_count / step;
  size_t left = pixel_count % step;

  simd_t r_mul = simd_t(77);
  simd_t g_mul = simd_t(150);
  simd_t b_mul = simd_t(29);

#pragma omp parallel for num_threads(4)
  for (size_t i = 0; i < tile; i += 1) {

    simd_t r_vec;
    simd_t g_vec;
    simd_t b_vec;
    simd_t gray_vec;

#pragma GCC unroll step
    for (size_t j = 0; j < step; j += 1) {
      r_vec[j] = input[3 * (i * step + j) + 0];
      g_vec[j] = input[3 * (i * step + j) + 1];
      b_vec[j] = input[3 * (i * step + j) + 2];
    }
    gray_vec = (r_vec * r_mul + g_vec * g_mul + b_vec * b_mul) >> 8;

    stdx::parallelism_v2::static_simd_cast<fixed_size_simd_t>(gray_vec).copy_to(
        output + i * step, stdx::element_aligned);
  }

  for (size_t i = 0; i < left; i += 1) {
    uint16_t r = input[3 * (tile * step + i) + 0];
    uint16_t g = input[3 * (tile * step + i) + 1];
    uint16_t b = input[3 * (tile * step + i) + 2];

    uint16_t gray = (r * 77 + g * 150 + b * 29);

    output[tile * step + i] = gray >> 8;
  }

  return true;
}



bool rgb_planar_2_gray_native(const unsigned char *input, unsigned char *output,
                              int width, int height) {

  int pixel_count = width * height;
  for (int i = 0; i < pixel_count; ++i) {
    unsigned char r = input[i];
    unsigned char g = input[i + pixel_count];
    unsigned char b = input[i + 2 * pixel_count];
    // Convert to grayscale using the luminosity method
    unsigned char gray = (unsigned char)(0.299 * r + 0.587 * g + 0.114 * b);
    output[i] = gray; // Set the output pixel to the grayscale value
  }
  return true;
}

bool rgb_planar_2_gray_parallel(const unsigned char *input,
                                unsigned char *output, int width, int height) {
  int pixel_count = width * height;

  tbb::parallel_for(tbb::blocked_range<int>(0, pixel_count),
                    [&](const tbb::blocked_range<int> &range) {
                      for (int i = range.begin(); i != range.end(); ++i) {
                        // Each pixel in RGB is represented by three consecutive
                        // bytes: R, G, B
                        unsigned char r = input[i];                   // Red
                        unsigned char g = input[i + pixel_count];     // Green
                        unsigned char b = input[i + 2 * pixel_count]; // Blue
                        // Convert to grayscale using the luminosity method
                        unsigned char gray =
                            (unsigned char)(0.299 * r + 0.587 * g + 0.114 * b);
                        output[i] = gray;
                      }
                    });
  return true;
}

bool rgb_planar_2_gray_simd(const unsigned char *input, unsigned char *output,
                            int width, int height) {

  int pixel_count = width * height;

  namespace stdx = std::experimental;

  using simd_t = stdx::native_simd<uint16_t>;
  constexpr auto step = simd_t::size();

  using fixed_size_simd_t = stdx::fixed_size_simd<uint8_t, step>;

  size_t tile = pixel_count / step;
  size_t left = pixel_count % step;

  simd_t r_mul = simd_t(77);
  simd_t g_mul = simd_t(150);
  simd_t b_mul = simd_t(29);

  for (size_t i = 0; i < tile; i += 1) {
    simd_t r_vec;
    simd_t g_vec;
    simd_t b_vec;
    simd_t gray_vec;

    fixed_size_simd_t r_vec_fixed(input + i * step, stdx::element_aligned);
    r_vec = stdx::parallelism_v2::static_simd_cast<simd_t>(r_vec_fixed);

    fixed_size_simd_t g_vec_fixed(input + i * step + pixel_count,
                                  stdx::element_aligned);
    g_vec = stdx::parallelism_v2::static_simd_cast<simd_t>(g_vec_fixed);

    fixed_size_simd_t b_vec_fixed(input + i * step + 2 * pixel_count,
                                  stdx::element_aligned);
    b_vec = stdx::parallelism_v2::static_simd_cast<simd_t>(b_vec_fixed);

    gray_vec = (r_vec * r_mul + g_vec * g_mul + b_vec * b_mul) >> 8;

    stdx::parallelism_v2::static_simd_cast<fixed_size_simd_t>(gray_vec).copy_to(
        output + i * step, stdx::element_aligned);
  }

  for (size_t i = 0; i < left; i += 1) {
    uint16_t r = input[tile * step + i];
    uint16_t g = input[tile * step + i + pixel_count];
    uint16_t b = input[tile * step + i + 2 * pixel_count];
    uint16_t gray = (r * 77 + g * 150 + b * 29);
    output[tile * step + i] = gray >> 8;
  }

  return true;
}

} // namespace kernels

} // namespace color_convert
} // namespace image_processing
