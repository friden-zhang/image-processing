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
#endif

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

bool rgb_packed_2_gray_simd(const unsigned char *input, unsigned char *output,
                            int width, int height) {

  int pixel_count = width * height;
  int index = 0;

#ifdef __ARM_NEON__
  for (; index <= pixel_count - 8; index += 8) {
    alignas(16) std::array<uint8_t, 8> r_values = {
        input[index * 3],       input[(index + 1) * 3], input[(index + 2) * 3],
        input[(index + 3) * 3], input[(index + 4) * 3], input[(index + 5) * 3],
        input[(index + 6) * 3], input[(index + 7) * 3]};

    alignas(16) std::array<uint8_t, 8> g_values = {
        input[index * 3 + 1],       input[(index + 1) * 3 + 1],
        input[(index + 2) * 3 + 1], input[(index + 3) * 3 + 1],
        input[(index + 4) * 3 + 1], input[(index + 5) * 3 + 1],
        input[(index + 6) * 3 + 1], input[(index + 7) * 3 + 1]};

    alignas(16) std::array<uint8_t, 8> b_values = {
        input[index * 3 + 2],       input[(index + 1) * 3 + 2],
        input[(index + 2) * 3 + 2], input[(index + 3) * 3 + 2],
        input[(index + 4) * 3 + 2], input[(index + 5) * 3 + 2],
        input[(index + 6) * 3 + 2], input[(index + 7) * 3 + 2]};

    // Convert to grayscale using the luminosity method
    uint8x8_t r_vec = vld1_u8(r_values.data());
    uint8x8_t g_vec = vld1_u8(g_values.data());
    uint8x8_t b_vec = vld1_u8(b_values.data());

    // use uint16x8_t to avoid overflow
    uint16x8_t r_mul = vmull_u8(r_vec, vdup_n_u8(77));
    uint16x8_t g_mul = vmull_u8(g_vec, vdup_n_u8(150));
    uint16x8_t b_mul = vmull_u8(b_vec, vdup_n_u8(29));

    uint16x8_t gray_sum = vaddq_u16(r_mul, g_mul);
    gray_sum = vaddq_u16(gray_sum, b_mul);

    gray_sum = vshrq_n_u16(gray_sum, 8);

    // jest throw away the high bits, >= 256 will be 0
    uint8x8_t gray_vec = vmovn_u16(gray_sum);
    vst1_u8(output + index, gray_vec);
  }

  for (; index < pixel_count; ++index) {
    // Each pixel in RGB is represented by three consecutive bytes: R, G, B
    unsigned char r = input[index * 3];     // Red
    unsigned char g = input[index * 3 + 1]; // Green
    unsigned char b = input[index * 3 + 2]; // Blue
    // Convert to grayscale using the luminosity method
    unsigned char gray = (unsigned char)(0.299 * r + 0.587 * g + 0.114 * b);

    output[index] = gray; // Set the output pixel to the grayscale value
  }

#else
  assert(false && "SIMD not supported on this platform");
#endif
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
  int index = 0;

#ifdef __ARM_NEON__
  for (; index <= pixel_count - 8; index += 8) {
    // Convert to grayscale using the luminosity method
    uint8x8_t r_vec = vld1_u8(input + index);
    uint8x8_t g_vec = vld1_u8(input + index + pixel_count);
    uint8x8_t b_vec = vld1_u8(input + index + 2 * pixel_count);

    // use uint16x8_t to avoid overflow
    uint16x8_t r_mul = vmull_u8(r_vec, vdup_n_u8(77));
    uint16x8_t g_mul = vmull_u8(g_vec, vdup_n_u8(150));
    uint16x8_t b_mul = vmull_u8(b_vec, vdup_n_u8(29));

    uint16x8_t gray_sum = vaddq_u16(r_mul, g_mul);
    gray_sum = vaddq_u16(gray_sum, b_mul);

    gray_sum = vshrq_n_u16(gray_sum, 8);

    // jest throw away the high bits, >= 256 will be 0
    uint8x8_t gray_vec = vmovn_u16(gray_sum);
    vst1_u8(output + index, gray_vec);
  }

  for (; index < pixel_count; ++index) {
    // Each pixel in RGB is represented by three consecutive bytes: R, G, B
    unsigned char r = input[index * 3];     // Red
    unsigned char g = input[index * 3 + 1]; // Green
    unsigned char b = input[index * 3 + 2]; // Blue
    // Convert to grayscale using the luminosity method
    unsigned char gray = (unsigned char)(0.299 * r + 0.587 * g + 0.114 * b);

    output[index] = gray; // Set the output pixel to the grayscale value
  }

#else
  assert(false && "SIMD not supported on this platform");
#endif
  return true;
}

} // namespace kernels

} // namespace color_convert
} // namespace image_processing
