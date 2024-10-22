#pragma once

#include "image-processing/color-convert/common/algo-type.hpp"
#include "image-processing/color-convert/common/mem-layout.hpp"

namespace image_processing {

namespace color_convert {

namespace kernels {

bool rgb_2_gray(const unsigned char *input, unsigned char *output, int width,
                int height, AlgoType algo_type, MemLayout mem_layout);

}
} // namespace color_convert
} // namespace image_processing