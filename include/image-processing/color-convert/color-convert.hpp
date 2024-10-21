#pragma once

#include "image-processing/color-convert/common/algo-type.hpp"
#include "image-processing/color-convert/common/image-format.hpp"

namespace image_processing {
namespace color_convert {

bool color_convert(const unsigned char *input_buffer,
                   unsigned char *output_buffer,
                   const ImageFormat &input_format,
                   const ImageFormat &output_format, const AlgoType &algo_type);

}
} // namespace image_processing