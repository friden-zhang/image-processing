#include <cub/block/block_reduce.cuh>

#include <thrust/device_vector.h>

#include <cuda/atomic>

#include <cstdio>

#include "rgba2gray.cuh"

namespace image_processing {

namespace color_convert {

namespace kernels {

namespace detail {

// TODO: use cudax::span<T> (in cccl)
// see https://github.com/NVIDIA/cccl/tree/main/examples/cudax/vector_add
__global__ void rgba_packed_2_gray_kernel(const unsigned char *input,
                                          unsigned char *output, int width,
                                          int height) {

  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  int index = (y * width + x) * 4;

  if (x < width && y < height) {
    unsigned char r = input[index];
    unsigned char g = input[index + 1];
    unsigned char b = input[index + 2];

    output[y * width + x] =
        static_cast<unsigned char>(0.299f * r + 0.587f * g + 0.114f * b);
  }
}
} // namespace detail

bool rgba_packed_2_gray_cuda(const unsigned char *input, unsigned char *output,
                             int width, int height) {

  // use device_trust vecrtor
  thrust::device_vector<unsigned char> input_dev(input,
                                                 input + 4 * width * height);
  thrust::device_vector<unsigned char> output_dev(width * height);

  unsigned char *input_dev_raw_ptr = thrust::raw_pointer_cast(input_dev.data());
  unsigned char *output_dev_raw_ptr =
      thrust::raw_pointer_cast(output_dev.data());
  dim3 blockSize(32, 32);
  dim3 gridSize((width + blockSize.x - 1) / blockSize.x,
                (height + blockSize.y - 1) / blockSize.y);
                
  detail::rgba_packed_2_gray_kernel<<<gridSize, blockSize>>>(input_dev_raw_ptr, output_dev_raw_ptr,
                                                             width, height);

  cudaDeviceSynchronize();
  thrust::copy(output_dev.begin(), output_dev.end(), output);
  return true;
  return true;
}

namespace detail {
// TODO: use cudax::span<T> (in cccl)
// see https://github.com/NVIDIA/cccl/tree/main/examples/cudax/vector_add
__global__ void rgba_planar_2_gray_kernel(const unsigned char *input,
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

bool rgba_planar_2_gray_cuda(const unsigned char *input, unsigned char *output,
                             int width, int height) {

  // use device_trust vecrtor
  thrust::device_vector<unsigned char> input_dev(input,
                                                 input + 4 * width * height);
  thrust::device_vector<unsigned char> output_dev(width * height);

  unsigned char *input_dev_raw_ptr = thrust::raw_pointer_cast(input_dev.data());
  unsigned char *output_dev_raw_ptr =
      thrust::raw_pointer_cast(output_dev.data());

  dim3 blockSize(32, 32);
  dim3 gridSize((width + blockSize.x - 1) / blockSize.x,
                (height + blockSize.y - 1) / blockSize.y);
  detail::rgba_planar_2_gray_kernel<<<gridSize, blockSize>>>(
      input_dev_raw_ptr, output_dev_raw_ptr, width, height);

  cudaDeviceSynchronize();
  thrust::copy(output_dev.begin(), output_dev.end(), output);
  return true;
  return true;
}

} // namespace kernels
} // namespace color_convert
} // namespace image_processing