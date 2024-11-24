#pragma once

#include "image-processing/color-convert/common/image-format.hpp"

namespace image_processing {

namespace color_convert {

namespace kernels {

bool yuv_2_rgb_native(const unsigned char *input, unsigned char *output,
                      int width, int height, ImageFormat yuv_format);

}

} // namespace color_convert

} // namespace image_processing