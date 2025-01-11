#include <cub/block/block_reduce.cuh>


#include <cuda/atomic>

#include <cstdio>

#include "rgb2gray.cuh"

namespace image_processing {

namespace color_convert {

namespace kernels {

namespace detail {

// TODO: use cudax::span<T> (in cccl)
// see https://github.com/NVIDIA/cccl/tree/main/examples/cudax/vector_add
__global__ void rgb_packed_2_gray_kernel(const unsigned char *input,
                                         unsigned char *output, int width,
                                         int height) {

  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x < width && y < height) {
    int index = (y * width + x) * 3;
    unsigned char r = input[index];
    unsigned char g = input[index + 1];
    unsigned char b = input[index + 2];

    output[y * width + x] =
        static_cast<unsigned char>(0.299f * r + 0.587f * g + 0.114f * b);
  }
}
} // namespace detail

bool launch_rgb_packed_2_gray_cuda(const unsigned char *input,
                                   unsigned char *output, int width,
                                   int height) {

  dim3 blockSize(32, 32);
  dim3 gridSize((width + blockSize.x - 1) / blockSize.x,
                (height + blockSize.y - 1) / blockSize.y);
  detail::rgb_packed_2_gray_kernel<<<gridSize, blockSize>>>(input, output,
                                                            width, height);
  // TODO: hanle error
  cudaDeviceSynchronize();
  return true;
}
namespace detail {
// TODO: use cudax::span<T> (in cccl)
// see https://github.com/NVIDIA/cccl/tree/main/examples/cudax/vector_add
__global__ void rgb_planar_2_gray_kernel(const unsigned char *input,
                                         unsigned char *output, int width,
                                         int height) {

  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  int index = (y * width + x);
  int planar_size = width * height;

  if (x < width && y < height) {
    unsigned char r = input[index];
    unsigned char g = input[index + planar_size];
    unsigned char b = input[index + 2 * planar_size];

    output[y * width + x] =
        static_cast<unsigned char>(0.299f * r + 0.587f * g + 0.114f * b);
  }
}

} // namespace detail
bool launch_rgb_planar_2_gray_cuda(const unsigned char *input,
                                   unsigned char *output, int width,
                                   int height) {

  dim3 blockSize(32, 32);
  dim3 gridSize((width + blockSize.x - 1) / blockSize.x,
                (height + blockSize.y - 1) / blockSize.y);
  detail::rgb_planar_2_gray_kernel<<<gridSize, blockSize>>>(input, output,
                                                            width, height);
  // TODO: hanle error
  cudaDeviceSynchronize();
  return true;
}

} // namespace kernels
} // namespace color_convert
} // namespace image_processing