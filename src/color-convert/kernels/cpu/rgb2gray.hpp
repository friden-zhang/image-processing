#pragma once

namespace image_processing {

namespace color_convert {

namespace kernels {

bool rgb_packed_2_gray_native(const unsigned char *input, unsigned char *output,
                              int width, int height);

bool rgb_packed_2_gray_parallel(const unsigned char *input,
                                unsigned char *output, int width, int height);

bool rgb_packed_2_gray_simd(const unsigned char *input, unsigned char *output,
                            int width, int height);

bool rgb_planar_2_gray_native(const unsigned char *input, unsigned char *output,
                              int width, int height);

bool rgb_planar_2_gray_parallel(const unsigned char *input,
                                unsigned char *output, int width, int height);

bool rgb_planar_2_gray_simd(const unsigned char *input, unsigned char *output,
                            int width, int height);
} // namespace kernels

} // namespace color_convert

} // namespace image_processing