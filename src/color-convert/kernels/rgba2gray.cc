#include "image-processing/color-convert/kernels/rgba2gray.hpp"
#include "cpu/rgba2gray.hpp"
#include "cuda/rgba2gray.cuh"
#include <assert.h>
#include <iostream>
namespace image_processing {

namespace color_convert {

namespace kernels {
bool rgba_2_gray(const unsigned char *input, unsigned char *output, int width,
                 int height, AlgoType algo_type, MemLayout mem_layout) {
  if (input == nullptr || output == nullptr || width <= 0 || height <= 0) {
    return false;
  }
  bool ret = false;
  switch (algo_type) {
  case AlgoType::kNativeCpu:
    switch (mem_layout) {
    case MemLayout::Packed:
      ret = rgba_packed_2_gray_native(input, output, width, height);
      break;
    case MemLayout::Planar:
      ret = rgba_planar_2_gray_native(input, output, width, height);
      break;
    default:
      assert(false);
      break;
    }
    break;

  case AlgoType::kParallelCpu:
    switch (mem_layout) {
    case MemLayout::Packed:
      ret = rgba_packed_2_gray_parallel(input, output, width, height);
      break;
    case MemLayout::Planar:
      ret = rgba_planar_2_gray_parallel(input, output, width, height);
      break;
    default:
      assert(false);
      break;
    }
    break;
  case AlgoType::kSimdCpu:
    switch (mem_layout) {
    case MemLayout::Packed:
      ret = rgba_packed_2_gray_simd(input, output, width, height);
      break;
    case MemLayout::Planar:
      ret = rgba_planar_2_gray_simd(input, output, width, height);
      break;
    default:
      assert(false);
      break;
    }
    break;
#if HAS_CUDA
  case AlgoType::kCuda:
    switch (mem_layout) {
    case MemLayout::Packed:
      ret = launch_rgba_packed_2_gray_cuda(input, output, width, height);
      break;
    case MemLayout::Planar:
      ret = launch_rgba_planar_2_gray_cuda(input, output, width, height);
      break;
    }
    break;
#else
    throw std::runtime_error("Cuda not supported in this build.");
#endif
  default:
    assert(false);
    break;
  }

  return ret;
}
} // namespace kernels
} // namespace color_convert
} // namespace image_processing