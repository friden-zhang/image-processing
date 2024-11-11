#include "rgba2gray.hpp"
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
namespace image_processing {

namespace color_convert {

namespace kernels {

bool rgba_packed_2_gray_native(const unsigned char *input,
                               unsigned char *output, int width, int height) {
  int pixel_count = width * height;
  for (int i = 0; i < pixel_count; ++i) {
    // Each pixel in RGBA is represented by three consecutive bytes: R, G, B, A
    // We only need the R, G, and B values to convert to grayscale
    unsigned char r = input[i * 4];     // Red
    unsigned char g = input[i * 4 + 1]; // Green
    unsigned char b = input[i * 4 + 2]; // Blue

    // Convert to grayscale using the luminosity method
    unsigned char gray = (unsigned char)(0.299 * r + 0.587 * g + 0.114 * b);

    output[i] = gray; // Set the output pixel to the grayscale value
  }
  return true; // Successful conversion
}

bool rgba_packed_2_gray_parallel(const unsigned char *input,
                                 unsigned char *output, int width, int height) {
  const auto *input_vec = reinterpret_cast<const uchar4 *>(input);
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

bool rgba_packed_2_gray_simd(const unsigned char *input, unsigned char *output,
                             int width, int height) {
  int pixel_count = width * height;
  int index = 0;

#ifdef __ARM_NEON__
  // std::unimplemented();
#elif defined(__AVX__)

  for (; index <= pixel_count - 8; index += 8) {
    std::array<uint8_t, 32> r_mask = {0,   255, 4,   255, 8,   255, 12,  255,
                                      255, 255, 255, 255, 255, 255, 255, 255,
                                      0,   255, 4,   255, 8,   255, 12,  255,
                                      255, 255, 255, 255, 255, 255, 255, 255};
    std::array<uint8_t, 32> g_mask = {1,   255, 5,   255, 9,   255, 13,  255,
                                      255, 255, 255, 255, 255, 255, 255, 255,
                                      1,   255, 5,   255, 9,   255, 13,  255,
                                      255, 255, 255, 255, 255, 255, 255, 255,};
    std::array<uint8_t, 32> b_mask = {2,   255, 6,   255, 10,   255, 14,  255,
                                      255, 255, 255, 255, 255, 255, 255, 255,
                                      2,   255, 6,   255, 10,   255, 14,  255,
                                      255, 255, 255, 255, 255, 255, 255, 255,};
    auto r_mask_vec = _mm256_loadu_si256(reinterpret_cast<__m256i*>(r_mask.data()));
    auto g_mask_vec = _mm256_loadu_si256(reinterpret_cast<__m256i*>(g_mask.data()));
    auto b_mask_vec = _mm256_loadu_si256(reinterpret_cast<__m256i*>(b_mask.data()));

    auto data = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(input + index * 4));
    
    __m256i r_temp_res = _mm256_shuffle_epi8(data, r_mask_vec);
    __m256i g_temp_res = _mm256_shuffle_epi8(data, g_mask_vec);
    __m256i b_temp_res = _mm256_shuffle_epi8(data, b_mask_vec);

    // uint16 x 8
    __m128i r_vec = _mm256_extracti128_si256(r_temp_res, 0);
    __m128i g_vec = _mm256_extracti128_si256(g_temp_res, 0);
    __m128i b_vec = _mm256_extracti128_si256(b_temp_res, 0);

    __m128i r_mul = _mm_mullo_epi16(r_vec, _mm_set1_epi16(77));
    __m128i g_mul = _mm_mullo_epi16(g_vec, _mm_set1_epi16(150));
    __m128i b_mul = _mm_mullo_epi16(b_vec, _mm_set1_epi16(29));

    __m128i gray_sum = _mm_add_epi16(r_mul, g_mul);
    gray_sum = _mm_add_epi16(gray_sum, b_mul);
    gray_sum = _mm_srli_epi16(gray_sum, 8);

    _mm_storeu_si128(reinterpret_cast<__m128i *>(output + index), gray_sum);
  }

#else
  assert(false && "SIMD not supported on this platform");
#endif

  for (; index < pixel_count; ++index) {
    // Each pixel in RGBA is represented by three consecutive bytes: R, G, B, A
    // We only need the R, G, and B values to convert to grayscale
    unsigned char r = input[index * 4];     // Red
    unsigned char g = input[index * 4 + 1]; // Green
    unsigned char b = input[index * 4 + 2]; // Blue
    // Convert to grayscale using the luminosity method
    unsigned char gray = (unsigned char)(0.299 * r + 0.587 * g + 0.114 * b);

    output[index] = gray; // Set the output pixel to the grayscale value
  }

  return true;
}

bool rgba_planar_2_gray_native(const unsigned char *input,
                               unsigned char *output, int width, int height) {
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

bool rgba_planar_2_gray_parallel(const unsigned char *input,
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

bool rgba_planar_2_gray_simd(const unsigned char *input, unsigned char *output,
                             int width, int height) {
  int pixel_count = width * height;
  int index = 0;

#ifdef __ARM_NEON__
  // std::unimplemented();
#elif defined(__AVX__)
  for (; index <= pixel_count - 16; index += 16) {
    __m128i r_vec_temp =
        _mm_loadu_si128(reinterpret_cast<const __m128i *>(input + index));
    __m128i g_vec_temp = _mm_loadu_si128(
        reinterpret_cast<const __m128i *>(input + index + pixel_count));
    __m128i b_vec_temp = _mm_loadu_si128(
        reinterpret_cast<const __m128i *>(input + index + 2 * pixel_count));

    __m256i r_vec = _mm256_cvtepu8_epi16(r_vec_temp);
    __m256i g_vec = _mm256_cvtepu8_epi16(g_vec_temp);
    __m256i b_vec = _mm256_cvtepu8_epi16(b_vec_temp);

    __m256i r_mul = _mm256_mullo_epi16(r_vec, _mm256_set1_epi16(77));
    __m256i g_mul = _mm256_mullo_epi16(g_vec, _mm256_set1_epi16(150));
    __m256i b_mul = _mm256_mullo_epi16(b_vec, _mm256_set1_epi16(29));

    __m256i gray_sum = _mm256_add_epi16(r_mul, g_mul);
    gray_sum = _mm256_add_epi16(gray_sum, b_mul);
    gray_sum = _mm256_srli_epi16(gray_sum, 8);

    __m256i packed = _mm256_packus_epi16(gray_sum, gray_sum);
    __m128i result = _mm256_extracti128_si256(packed, 0);
    // if intput is aligned to 32 bytes, we can use _mm_store_si128
    // otherwise, we need to use _mm_storeu_si128
    _mm_storeu_si128(reinterpret_cast<__m128i *>(output + index), result);
  }
#else
  assert(false && "SIMD not supported on this platform");
#endif

  for (; index < pixel_count; ++index) {
    // Each pixel in RGB is represented by three consecutive bytes: R, G, B, A
    // We only need the R, G, and B values to convert to grayscale
    unsigned char r = input[index];                   // Red
    unsigned char g = input[index + pixel_count];     // Green
    unsigned char b = input[index + 2 * pixel_count]; // Blue
    // Convert to grayscale using the luminosity method
    unsigned char gray = (unsigned char)(0.299 * r + 0.587 * g + 0.114 * b);

    output[index] = gray; // Set the output pixel to the grayscale value
  }
  return true;
}

} // namespace kernels

} // namespace color_convert

} // namespace image_processing
