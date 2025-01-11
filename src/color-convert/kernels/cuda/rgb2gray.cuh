#pragma once

namespace image_processing {

namespace color_convert {

namespace kernels {

bool launch_rgb_packed_2_gray_cuda(const unsigned char *input, unsigned char *output,
                       int width, int height);

bool launch_rgb_planar_2_gray_cuda(const unsigned char *input, unsigned char *output,
                       int width, int height);

} // namespace kernels
} // namespace color_convert
} // namespace image_processing