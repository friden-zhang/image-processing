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

#elif defined(__AVX__)
  for (; index <= pixel_count - 16; index += 16) {

    __m256i r_vec = _mm256_set_epi16(
        input[(index + 15) * 3], input[(index + 14) * 3],
        input[(index + 13) * 3], input[(index + 12) * 3],
        input[(index + 11) * 3], input[(index + 10) * 3],
        input[(index + 9) * 3], input[(index + 8) * 3], input[(index + 7) * 3],
        input[(index + 6) * 3], input[(index + 5) * 3], input[(index + 4) * 3],
        input[(index + 3) * 3], input[(index + 2) * 3], input[(index + 1) * 3],
        input[(index + 0) * 3]);

    __m256i g_vec = _mm256_set_epi16(
        input[(index + 15) * 3 + 1], input[(index + 14) * 3 + 1],
        input[(index + 13) * 3 + 1], input[(index + 12) * 3 + 1],
        input[(index + 11) * 3 + 1], input[(index + 10) * 3 + 1],
        input[(index + 9) * 3 + 1], input[(index + 8) * 3 + 1],
        input[(index + 7) * 3 + 1], input[(index + 6) * 3 + 1],
        input[(index + 5) * 3 + 1], input[(index + 4) * 3 + 1],
        input[(index + 3) * 3 + 1], input[(index + 2) * 3 + 1],
        input[(index + 1) * 3 + 1], input[(index + 0) * 3 + 1]);

    __m256i b_vec = _mm256_set_epi16(
        input[(index + 15) * 3 + 2], input[(index + 14) * 3 + 2],
        input[(index + 13) * 3 + 2], input[(index + 12) * 3 + 2],
        input[(index + 11) * 3 + 2], input[(index + 10) * 3 + 2],
        input[(index + 9) * 3 + 2], input[(index + 8) * 3 + 2],
        input[(index + 7) * 3 + 2], input[(index + 6) * 3 + 2],
        input[(index + 5) * 3 + 2], input[(index + 4) * 3 + 2],
        input[(index + 3) * 3 + 2], input[(index + 2) * 3 + 2],
        input[(index + 1) * 3 + 2], input[(index + 0) * 3 + 2]);

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

  /*
  // see:
  http://giantpandacv.com/project/%E9%83%A8%E7%BD%B2%E4%BC%98%E5%8C%96/AI%20PC%E7%AB%AF%E4%BC%98%E5%8C%96/%E3%80%90AI%20PC%E7%AB%AF%E7%AE%97%E6%B3%95%E4%BC%98%E5%8C%96%E3%80%91%E4%B8%80%EF%BC%8C%E4%B8%80%E6%AD%A5%E6%AD%A5%E4%BC%98%E5%8C%96RGB%E8%BD%AC%E7%81%B0%E5%BA%A6%E5%9B%BE%E7%AE%97%E6%B3%95/#9-rgb2gray
  // but run benchmark not faster than native code
  for (int Y = 0; Y < height; Y++) {
    const unsigned char *LinePS = input + Y * width * 3;
    unsigned char *LinePD = output + Y * width;
    int X = 0;
    for (; X < width - 12; X += 12, LinePS += 36) {
      __m128i p1aL = _mm_mullo_epi16(
          _mm_cvtepu8_epi16(_mm_loadu_si128((__m128i *)(LinePS + 0))),
          _mm_setr_epi16(R_WT, G_WT, B_WT, R_WT, G_WT, B_WT, R_WT, G_WT)); // 1
      __m128i p2aL = _mm_mullo_epi16(
          _mm_cvtepu8_epi16(_mm_loadu_si128((__m128i *)(LinePS + 1))),
          _mm_setr_epi16(G_WT, B_WT, R_WT, G_WT, B_WT, R_WT, G_WT, B_WT)); // 2
      __m128i p3aL = _mm_mullo_epi16(
          _mm_cvtepu8_epi16(_mm_loadu_si128((__m128i *)(LinePS + 2))),
          _mm_setr_epi16(B_WT, R_WT, G_WT, B_WT, R_WT, G_WT, B_WT, R_WT)); // 3

      __m128i p1aH = _mm_mullo_epi16(
          _mm_cvtepu8_epi16(_mm_loadu_si128((__m128i *)(LinePS + 8))),
          _mm_setr_epi16(B_WT, R_WT, G_WT, B_WT, R_WT, G_WT, B_WT, R_WT)); // 4
      __m128i p2aH = _mm_mullo_epi16(
          _mm_cvtepu8_epi16(_mm_loadu_si128((__m128i *)(LinePS + 9))),
          _mm_setr_epi16(R_WT, G_WT, B_WT, R_WT, G_WT, B_WT, R_WT, G_WT)); // 5
      __m128i p3aH = _mm_mullo_epi16(
          _mm_cvtepu8_epi16(_mm_loadu_si128((__m128i *)(LinePS + 10))),
          _mm_setr_epi16(G_WT, B_WT, R_WT, G_WT, B_WT, R_WT, G_WT, B_WT)); // 6

      __m128i p1bL = _mm_mullo_epi16(
          _mm_cvtepu8_epi16(_mm_loadu_si128((__m128i *)(LinePS + 18))),
          _mm_setr_epi16(B_WT, G_WT, R_WT, B_WT, G_WT, R_WT, B_WT, G_WT)); // 7
      __m128i p2bL = _mm_mullo_epi16(
          _mm_cvtepu8_epi16(_mm_loadu_si128((__m128i *)(LinePS + 19))),
          _mm_setr_epi16(G_WT, R_WT, B_WT, G_WT, R_WT, B_WT, G_WT, R_WT)); // 8
      __m128i p3bL = _mm_mullo_epi16(
          _mm_cvtepu8_epi16(_mm_loadu_si128((__m128i *)(LinePS + 20))),
          _mm_setr_epi16(R_WT, B_WT, G_WT, R_WT, B_WT, G_WT, R_WT, B_WT)); // 9

      __m128i p1bH = _mm_mullo_epi16(
          _mm_cvtepu8_epi16(_mm_loadu_si128((__m128i *)(LinePS + 26))),
          _mm_setr_epi16(R_WT, B_WT, G_WT, R_WT, B_WT, G_WT, R_WT, B_WT)); // 10
      __m128i p2bH = _mm_mullo_epi16(
          _mm_cvtepu8_epi16(_mm_loadu_si128((__m128i *)(LinePS + 27))),
          _mm_setr_epi16(B_WT, G_WT, R_WT, B_WT, G_WT, R_WT, B_WT, G_WT)); // 11
      __m128i p3bH = _mm_mullo_epi16(
          _mm_cvtepu8_epi16(_mm_loadu_si128((__m128i *)(LinePS + 28))),
          _mm_setr_epi16(G_WT, R_WT, B_WT, G_WT, R_WT, B_WT, G_WT, R_WT)); // 12

      __m128i sumaL = _mm_add_epi16(p3aL, _mm_add_epi16(p1aL, p2aL)); // 13
      __m128i sumaH = _mm_add_epi16(p3aH, _mm_add_epi16(p1aH, p2aH)); // 14
      __m128i sumbL = _mm_add_epi16(p3bL, _mm_add_epi16(p1bL, p2bL)); // 15
      __m128i sumbH = _mm_add_epi16(p3bH, _mm_add_epi16(p1bH, p2bH)); // 16
      __m128i sclaL = _mm_srli_epi16(sumaL, 8);                       // 17
      __m128i sclaH = _mm_srli_epi16(sumaH, 8);                       // 18
      __m128i sclbL = _mm_srli_epi16(sumbL, 8);                       // 19
      __m128i sclbH = _mm_srli_epi16(sumbH, 8);                       // 20
      __m128i shftaL = _mm_shuffle_epi8(
          sclaL, _mm_setr_epi8(0, 6, 12, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
                               -1, -1, -1)); // 21
      __m128i shftaH = _mm_shuffle_epi8(
          sclaH, _mm_setr_epi8(-1, -1, -1, 18, 24, 30, -1, -1, -1, -1, -1, -1,
                               -1, -1, -1, -1)); // 22
      __m128i shftbL = _mm_shuffle_epi8(
          sclbL, _mm_setr_epi8(-1, -1, -1, -1, -1, -1, 0, 6, 12, -1, -1, -1, -1,
                               -1, -1, -1)); // 23
      __m128i shftbH = _mm_shuffle_epi8(
          sclbH, _mm_setr_epi8(-1, -1, -1, -1, -1, -1, -1, -1, -1, 18, 24, 30,
                               -1, -1, -1, -1));     // 24
      __m128i accumL = _mm_or_si128(shftaL, shftbL); // 25
      __m128i accumH = _mm_or_si128(shftaH, shftbH); // 26
      __m128i h3 = _mm_or_si128(accumL, accumH);     // 27
      //__m128i h3 = _mm_blendv_epi8(accumL, accumH, _mm_setr_epi8(0, 0, 0, -1,
      //-1, -1, 0, 0, 0, -1, -1, -1, 1, 1, 1, 1));
      _mm_storeu_si128((__m128i *)(LinePD + X), h3); // 28
    }
    for (; X < width; X++, LinePS += 3) { // 29
      LinePD[X] =
          (B_WT * LinePS[0] + G_WT * LinePS[1] + R_WT * LinePS[2]) >> 8; // 30
    }
  }
  */

#else
  assert(false && "SIMD not supported on this platform");
#endif
  for (; index < pixel_count; ++index) {
    // Each pixel in RGB is represented by three consecutive bytes: R, G, B
    unsigned char r = input[index * 3];     // Red
    unsigned char g = input[index * 3 + 1]; // Green
    unsigned char b = input[index * 3 + 2]; // Blue
    // Convert to grayscale using the luminosity method
    unsigned char gray = (unsigned char)(0.299 * r + 0.587 * g + 0.114 * b);

    output[index] = gray; // Set the output pixel to the grayscale value
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
    // Each pixel in RGB is represented by three consecutive bytes: R, G, B
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
