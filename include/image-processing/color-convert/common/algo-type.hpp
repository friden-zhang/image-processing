#pragma once

namespace image_processing {
namespace color_convert {

enum class AlgoType { kNativeCpu, kParallelCpu, kSimdCpu, kCuda };

}
} // namespace image_processing