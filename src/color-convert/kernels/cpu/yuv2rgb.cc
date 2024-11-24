#pragma once

#include "yuv2rgb.hpp"

#include <tbb/blocked_range.h>
#include <tbb/parallel_for.h>

#include <string>

#include <exception>

#ifdef __ARM_NEON__
#include <arm_neon.h>
#elif defined(__AVX__)
#include <immintrin.h>
#endif

namespace image_processing {

namespace color_convert {

namespace kernels {

#define CLAMP(val) (val < 0 ? 0 : (val > 255 ? 255 : val))

template <int Y0_INDEX, int Y1_INDEX, int U_INDEX, int V_INDEX>
bool yuv422_2_rgb_native(const unsigned char *input, unsigned char *output,
                         int width, int height) {

  int uyuv_frame_size = width * height * 2;
  int output_index = 0;

  for (int i = 0; i < uyuv_frame_size; i += 4) {
    auto u = input[i + U_INDEX];
    auto y0 = input[i + Y0_INDEX];
    auto v = input[i + V_INDEX];
    auto y1 = input[i + Y1_INDEX];

    int C0 = y0 - 16;
    int C1 = y1 - 16;
    int D = u - 128;
    int E = v - 128;

    int R0 = CLAMP((298 * C0 + 409 * E + 128) >> 8);
    int G0 = CLAMP((298 * C0 - 100 * D - 208 * E + 128) >> 8);
    int B0 = CLAMP((298 * C0 + 516 * D + 128) >> 8);

    int R1 = CLAMP((298 * C1 + 409 * E + 128) >> 8);
    int G1 = CLAMP((298 * C1 - 100 * D - 208 * E + 128) >> 8);
    int B1 = CLAMP((298 * C1 + 516 * D + 128) >> 8);

    output[output_index++] = static_cast<unsigned char>(R0);
    output[output_index++] = static_cast<unsigned char>(G0);
    output[output_index++] = static_cast<unsigned char>(B0);
    output[output_index++] = static_cast<unsigned char>(R1);
    output[output_index++] = static_cast<unsigned char>(G1);
    output[output_index++] = static_cast<unsigned char>(B1);
  }

  return true;
}

bool yuv420p_2_rgb_native(const unsigned char *input, unsigned char *output,
                          int width, int height) {

  const unsigned char *y_plane = input;
  const unsigned char *u_plane = y_plane + width * height;
  const unsigned char *v_plane = u_plane + (width * height) / 4;

  for (int y = 0; y < height; ++y) {
    for (int x = 0; x < width; ++x) {
      int Y = y_plane[y * width + x];

      int u_index = (y / 2) * (width / 2) + (x / 2);
      int v_index = (y / 2) * (width / 2) + (x / 2);
      int U = u_plane[u_index];
      int V = v_plane[v_index];

      int C = Y - 16;
      int D = U - 128;
      int E = V - 128;

      int R = CLAMP((298 * C + 409 * E + 128) >> 8);
      int G = CLAMP((298 * C - 100 * D - 208 * E + 128) >> 8);
      int B = CLAMP((298 * C + 516 * D + 128) >> 8);

      int rgb_index = (y * width + x) * 3;
      output[rgb_index] = (unsigned char)R;
      output[rgb_index + 1] = (unsigned char)G;
      output[rgb_index + 2] = (unsigned char)B;
    }
  }

  return true;
}

template <int Y0_INDEX, int Y1_INDEX, int U_INDEX, int V_INDEX>
bool yuv422_2_rgb_parallel(const unsigned char *input, unsigned char *output,
                           int width, int height) {

  int uyuv_frame_size = width * height * 2;
  int output_index = 0;

  tbb::parallel_for(tbb::blocked_range<int>(0, uyuv_frame_size),
                    [&](const tbb::blocked_range<int> &range) {
                      for (int i = range.begin(); i != range.end(); i += 4) {
                        auto u = input[i + U_INDEX];
                        auto y0 = input[i + Y0_INDEX];
                        auto v = input[i + V_INDEX];
                        auto y1 = input[i + Y1_INDEX];
                        int C0 = y0 - 16;
                        int C1 = y1 - 16;
                        int D = u - 128;
                        int E = v - 128;

                        int R0 = CLAMP((298 * C0 + 409 * E + 128) >> 8);
                        int G0 =
                            CLAMP((298 * C0 - 100 * D - 208 * E + 128) >> 8);
                        int B0 = CLAMP((298 * C0 + 516 * D + 128) >> 8);

                        int R1 = CLAMP((298 * C1 + 409 * E + 128) >> 8);
                        int G1 =
                            CLAMP((298 * C1 - 100 * D - 208 * E + 128) >> 8);
                        int B1 = CLAMP((298 * C1 + 516 * D + 128) >> 8);

                        output[output_index++] = static_cast<unsigned char>(R0);
                        output[output_index++] = static_cast<unsigned char>(G0);
                        output[output_index++] = static_cast<unsigned char>(B0);
                        output[output_index++] = static_cast<unsigned char>(R1);
                        output[output_index++] = static_cast<unsigned char>(G1);
                        output[output_index++] = static_cast<unsigned char>(B1);
                      }
                    });

  return true;
}

bool yuv420p_2_rgb_parallel(const unsigned char *input, unsigned char *output,
                            int width, int height) {

  const unsigned char *y_plane = input;
  const unsigned char *u_plane = y_plane + width * height;
  const unsigned char *v_plane = u_plane + (width * height) / 4;

  tbb::parallel_for(tbb::blocked_range<int>(0, height),
                    [&](const tbb::blocked_range<int> &range) {
                      for (int y = range.begin(); y != range.end(); ++y) {
                        for (int x = 0; x < width; ++x) {
                          int Y = y_plane[y * width + x];

                          int u_index = (y / 2) * (width / 2) + (x / 2);
                          int v_index = (y / 2) * (width / 2) + (x / 2);
                          int U = u_plane[u_index];
                          int V = v_plane[v_index];

                          int C = Y - 16;
                          int D = U - 128;
                          int E = V - 128;

                          int R = CLAMP((298 * C + 409 * E + 128) >> 8);
                          int G =
                              CLAMP((298 * C - 100 * D - 208 * E + 128) >> 8);
                          int B = CLAMP((298 * C + 516 * D + 128) >> 8);

                          int rgb_index = (y * width + x) * 3;
                          output[rgb_index] = (unsigned char)R;
                          output[rgb_index + 1] = (unsigned char)G;
                          output[rgb_index + 2] = (unsigned char)B;
                        }
                      }
                    });

  return true;
}

bool yuv_2_rgb_native(const unsigned char *input, unsigned char *output,
                      int width, int height, ImageFormat yuv_format) {
  if (yuv_format == ImageFormat::IMAGE_YUYV) {
    return yuv422_2_rgb_native<0, 2, 1, 3>(input, output, width, height);
  } else if (yuv_format == ImageFormat::IMAGE_UYVY) {
    return yuv422_2_rgb_native<1, 3, 0, 2>(input, output, width, height);
  } else if (yuv_format == ImageFormat::IMAGE_YVYU) {
    return yuv422_2_rgb_native<0, 2, 3, 1>(input, output, width, height);
  } else if (yuv_format == ImageFormat::IMAGE_I420) {
    return yuv420p_2_rgb_native(input, output, width, height);
  }
  return false;
}

bool yuv_2_rgb_parallel(const unsigned char *input, unsigned char *output,
                        int width, int height, ImageFormat yuv_format) {
  if (yuv_format == ImageFormat::IMAGE_YUYV) {
    return yuv422_2_rgb_parallel<0, 2, 1, 3>(input, output, width, height);
  } else if (yuv_format == ImageFormat::IMAGE_UYVY) {
    return yuv422_2_rgb_parallel<1, 3, 0, 2>(input, output, width, height);
  } else if (yuv_format == ImageFormat::IMAGE_YVYU) {
    return yuv422_2_rgb_parallel<0, 2, 3, 1>(input, output, width, height);
  } else if (yuv_format == ImageFormat::IMAGE_I420) {
    return yuv420p_2_rgb_parallel(input, output, width, height);
  }
  return false;
}

template <int Y0_INDEX, int Y1_INDEX, int U_INDEX, int V_INDEX>
bool yuv422_2_rgb_simd(const unsigned char *input, unsigned char *output,
                       int width, int height) {

  int uyuv_frame_size = width * height * 2;
  int output_index = 0;

  int i = 0;

#ifdef __ARM_NEON__

  for (; i < uyuv_frame_size; i += 16) {
    uint8x8_t y0 = vld1_u8(input + i + Y0_INDEX);
    uint8x8_t y1 = vld1_u8(input + i + Y1_INDEX);
    uint8x8_t u = vld1_u8(input + i + U_INDEX);
    uint8x8_t v = vld1_u8(input + i + V_INDEX);

    int16x8_t u_signed = vreinterpretq_s16_u16(vsubl_u8(u, vdup_n_u8(128)));
    int16x8_t v_signed = vreinterpretq_s16_u16(vsubl_u8(v, vdup_n_u8(128)));

    int16x8_t y0_signed = vreinterpretq_s16_u16(vsubl_u8(y0, vdup_n_u8(16)));
    int16x8_t y1_signed = vreinterpretq_s16_u16(vsubl_u8(y1, vdup_n_u8(16)));

    // int C0 = y0 - 16;
    // int C1 = y1 - 16;
    // int D = u - 128;
    // int E = v - 128;

    // int R0 = CLAMP((298 * C0 + 409 * E + 128) >> 8);
    // int G0 = CLAMP((298 * C0 - 100 * D - 208 * E + 128) >> 8);
    // int B0 = CLAMP((298 * C0 + 516 * D + 128) >> 8);

    // int R1 = CLAMP((298 * C1 + 409 * E + 128) >> 8);
    // int G1 = CLAMP((298 * C1 - 100 * D - 208 * E + 128) >> 8);
    // int B1 = CLAMP((298 * C1 + 516 * D + 128) >> 8);

    int16x8_t r0 = vmlaq_n_s16(vqrdmulhq_n_s16(y0_signed, 298), v_signed, 409);
    r0 = vaddq_s16(r0, vdupq_n_s16(128));

    int16x8_t g0 = vmlaq_n_s16(
        vmlaq_n_s16(vqrdmulhq_n_s16(y0_signed, 298), u_signed, -100), v_signed,
        -208);
    g0 = vaddq_s16(g0, vdupq_n_s16(128));

    int16x8_t b0 = vmlaq_n_s16(vqrdmulhq_n_s16(y0_signed, 298), u_signed, 516);
    b0 = vaddq_s16(b0, vdupq_n_s16(128));

    int16x8_t r1 = vmlaq_n_s16(vqrdmulhq_n_s16(y1_signed, 298), v_signed, 409);
    r1 = vaddq_s16(r1, vdupq_n_s16(128));

    int16x8_t g1 = vmlaq_n_s16(
        vmlaq_n_s16(vqrdmulhq_n_s16(y1_signed, 298), u_signed, -100), v_signed,
        -208);
    g1 = vaddq_s16(g1, vdupq_n_s16(128));

    int16x8_t b1 = vmlaq_n_s16(vqrdmulhq_n_s16(y1_signed, 298), u_signed, 516);
    b1 = vaddq_s16(b1, vdupq_n_s16(128));

    // uint16_t -> uint8_t
    uint8x8_t r0_clamped = vqmovun_s16(r0);
    uint8x8_t g0_clamped = vqmovun_s16(g0);
    uint8x8_t b0_clamped = vqmovun_s16(b0);
    uint8x8_t r1_clamped = vqmovun_s16(r1);
    uint8x8_t g1_clamped = vqmovun_s16(g1);
    uint8x8_t b1_clamped = vqmovun_s16(b1);

    for (int j = 0; j < 8; ++j) {
      output[output_index++] = r0_clamped[j];
      output[output_index++] = g0_clamped[j];
      output[output_index++] = b0_clamped[j];
      output[output_index++] = r1_clamped[j];
      output[output_index++] = g1_clamped[j];
      output[output_index++] = b1_clamped[j];
    }
  }

#elif defined(__AVX__)

#endif

  for (; i < uyuv_frame_size; i += 4) {
    auto u = input[i + U_INDEX];
    auto y0 = input[i + Y0_INDEX];
    auto v = input[i + V_INDEX];
    auto y1 = input[i + Y1_INDEX];

    int C0 = y0 - 16;
    int C1 = y1 - 16;
    int D = u - 128;
    int E = v - 128;

    int R0 = CLAMP((298 * C0 + 409 * E + 128) >> 8);
    int G0 = CLAMP((298 * C0 - 100 * D - 208 * E + 128) >> 8);
    int B0 = CLAMP((298 * C0 + 516 * D + 128) >> 8);

    int R1 = CLAMP((298 * C1 + 409 * E + 128) >> 8);
    int G1 = CLAMP((298 * C1 - 100 * D - 208 * E + 128) >> 8);
    int B1 = CLAMP((298 * C1 + 516 * D + 128) >> 8);

    output[output_index++] = static_cast<unsigned char>(R0);
    output[output_index++] = static_cast<unsigned char>(G0);
    output[output_index++] = static_cast<unsigned char>(B0);
    output[output_index++] = static_cast<unsigned char>(R1);
    output[output_index++] = static_cast<unsigned char>(G1);
    output[output_index++] = static_cast<unsigned char>(B1);
  }

  return true;
}

bool yuv420p_2_rgb_simd(const unsigned char *input, unsigned char *output,
                        int width, int height) {

  const unsigned char *y_plane = input;
  const unsigned char *u_plane = y_plane + width * height;
  const unsigned char *v_plane = u_plane + (width * height) / 4;

  int y = 0;
  int x = 0;

#ifdef __ARM_NEON__

  for (; y < height; y += 2) {
    for (x = 0; x < width; x += 8) {
      uint8x8_t y_row0 = vld1_u8(&y_plane[y * width + x]);
      uint8x8_t y_row1 = vld1_u8(&y_plane[(y + 1) * width + x]);

      int uv_offset = (y / 2) * (width / 2) + (x / 2);
      uint8x8_t u_data = vld1_u8(&u_plane[uv_offset]);
      uint8x8_t v_data = vld1_u8(&v_plane[uv_offset]);

      int16x8_t u_signed =
          vreinterpretq_s16_u16(vsubl_u8(u_data, vdup_n_u8(128)));
      int16x8_t v_signed =
          vreinterpretq_s16_u16(vsubl_u8(v_data, vdup_n_u8(128)));

      int16x8_t y0_signed =
          vreinterpretq_s16_u16(vsubl_u8(y_row0, vdup_n_u8(16)));
      int16x8_t y1_signed =
          vreinterpretq_s16_u16(vsubl_u8(y_row1, vdup_n_u8(16)));

      // R = 1.164 * (Y - 16) + 1.596 * (V - 128)
      // G = 1.164 * (Y - 16) - 0.392 * (U - 128) - 0.813 * (V - 128)
      // B = 1.164 * (Y - 16) + 2.017 * (U - 128)

      int16x8_t r0 =
          vmlaq_n_s16(vqrdmulhq_n_s16(y0_signed, 298), v_signed, 409);
      int16x8_t g0 = vmlsq_n_s16(
          vmlsq_n_s16(vqrdmulhq_n_s16(y0_signed, 298), u_signed, 100), v_signed,
          208);
      int16x8_t b0 =
          vmlaq_n_s16(vqrdmulhq_n_s16(y0_signed, 298), u_signed, 516);

      int16x8_t r1 =
          vmlaq_n_s16(vqrdmulhq_n_s16(y1_signed, 298), v_signed, 409);
      int16x8_t g1 = vmlsq_n_s16(
          vmlsq_n_s16(vqrdmulhq_n_s16(y1_signed, 298), u_signed, 100), v_signed,
          208);
      int16x8_t b1 =
          vmlaq_n_s16(vqrdmulhq_n_s16(y1_signed, 298), u_signed, 516);

      uint8x8_t r0_clamped = vqmovun_s16(r0);
      uint8x8_t g0_clamped = vqmovun_s16(g0);
      uint8x8_t b0_clamped = vqmovun_s16(b0);

      uint8x8_t r1_clamped = vqmovun_s16(r1);
      uint8x8_t g1_clamped = vqmovun_s16(g1);
      uint8x8_t b1_clamped = vqmovun_s16(b1);

      for (int i = 0; i < 8; ++i) {
        int idx1 = ((y * width) + x + i) * 3;
        output[idx1] = r0_clamped[i];
        output[idx1 + 1] = g0_clamped[i];
        output[idx1 + 2] = b0_clamped[i];

        int idx2 = (((y + 1) * width) + x + i) * 3;
        output[idx2] = r1_clamped[i];
        output[idx2 + 1] = g1_clamped[i];
        output[idx2 + 2] = b1_clamped[i];
      }
    }
  }

#elif defined(__AVX__)

#endif

  for (; y < height; ++y) {
    for (; x < width; ++x) {
      int Y = y_plane[y * width + x];

      int u_index = (y / 2) * (width / 2) + (x / 2);
      int v_index = (y / 2) * (width / 2) + (x / 2);
      int U = u_plane[u_index];
      int V = v_plane[v_index];

      int C = Y - 16;
      int D = U - 128;
      int E = V - 128;

      int R = CLAMP((298 * C + 409 * E + 128) >> 8);
      int G = CLAMP((298 * C - 100 * D - 208 * E + 128) >> 8);
      int B = CLAMP((298 * C + 516 * D + 128) >> 8);

      int rgb_index = (y * width + x) * 3;
      output[rgb_index] = (unsigned char)R;
      output[rgb_index + 1] = (unsigned char)G;
      output[rgb_index + 2] = (unsigned char)B;
    }
  }

  return true;
}

} // namespace kernels

} // namespace color_convert

} // namespace image_processing