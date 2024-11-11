#include "image-processing/color-convert/kernels/rgba2gray.hpp"
#include "gtest/gtest.h"
#include <fstream>
#include <stdexcept>
#include <vector>

static std::vector<unsigned char> read_raw_image(const std::string &filename,
                                                 int width, int height) {

  size_t size = width * height * 4;

  std::vector<unsigned char> data(size);

  std::ifstream file(filename, std::ios::binary);
  if (!file) {
    throw std::runtime_error("Failed to open file: " + filename);
  }

  file.read(reinterpret_cast<char *>(data.data()), size);
  if (!file) {
    throw std::runtime_error("Failed to read data from file: " + filename);
  }

  return data;
}

TEST(RGBA2GrayTest, PackedNativeConversion) {
  int width = 1920;
  int height = 1080;
  auto input_image = read_raw_image("/tmp/geometric_image.rgba", width, height);
  std::vector<unsigned char> output_image(width * height);

  ASSERT_TRUE(image_processing::color_convert::kernels::rgba_2_gray(
      input_image.data(), output_image.data(), width, height,
      image_processing::color_convert::AlgoType::kNativeCpu,
      image_processing::color_convert::MemLayout::Packed))
      << "Expected rgba_2_gray to return true but it returned false.";

  EXPECT_EQ(output_image[100 * width + 100], 76);
  EXPECT_EQ(output_image[50 * width + 400], 29);
}

TEST(RGBA2GrayTest, PlanarNativeConversion) {
  int width = 1920;
  int height = 1080;
  auto input_image = read_raw_image("/tmp/geometric_image.rgba", width, height);
  auto input_image_planar = input_image;

  for (int index = 0; index < height * width; index++) {
    input_image_planar[index] = input_image[index * 4];
    input_image_planar[index + height * width] = input_image[index * 4 + 1];
    input_image_planar[index + 2 * height * width] = input_image[index * 4 + 2];
  }

  std::vector<unsigned char> output_image(width * height);

  ASSERT_TRUE(image_processing::color_convert::kernels::rgba_2_gray(
      input_image_planar.data(), output_image.data(), width, height,
      image_processing::color_convert::AlgoType::kNativeCpu,
      image_processing::color_convert::MemLayout::Planar))
      << "Expected rgba_2_gray to return true but it returned false.";

  EXPECT_EQ(output_image[100 * width + 100], 76);
  EXPECT_EQ(output_image[50 * width + 400], 29);
}

TEST(RGBA2GrayTest, PackedParallelConversion) {
  int width = 1920;
  int height = 1080;
  auto input_image = read_raw_image("/tmp/geometric_image.rgba", width, height);
  std::vector<unsigned char> output_image(width * height);

  ASSERT_TRUE(image_processing::color_convert::kernels::rgba_2_gray(
      input_image.data(), output_image.data(), width, height,
      image_processing::color_convert::AlgoType::kParallelCpu,
      image_processing::color_convert::MemLayout::Packed))
      << "Expected rgba_2_gray to return true but it returned false.";

  EXPECT_EQ(output_image[100 * width + 100], 76);
  EXPECT_EQ(output_image[50 * width + 400], 29);
}

TEST(RGBA2GrayTest, PlanarParallelConversion) {
  int width = 1920;
  int height = 1080;
  auto input_image = read_raw_image("/tmp/geometric_image.rgba", width, height);
  std::vector<unsigned char> output_image(width * height);
  auto input_image_planar = input_image;

  for (int index = 0; index < height * width; index++) {
    input_image_planar[index] = input_image[index * 4];
    input_image_planar[index + height * width] = input_image[index * 4 + 1];
    input_image_planar[index + 2 * height * width] = input_image[index * 4 + 2];
  }

  ASSERT_TRUE(image_processing::color_convert::kernels::rgba_2_gray(
      input_image_planar.data(), output_image.data(), width, height,
      image_processing::color_convert::AlgoType::kParallelCpu,
      image_processing::color_convert::MemLayout::Planar))
      << "Expected rgba_2_gray to return true but it returned false.";

  EXPECT_EQ(output_image[100 * width + 100], 76);
  EXPECT_EQ(output_image[50 * width + 400], 29);
}

TEST(RGBA2GrayTest, PackedSIMDConversion) {
  int width = 1920;
  int height = 1080;
  auto input_image = read_raw_image("/tmp/geometric_image.rgba", width, height);
  std::vector<unsigned char> output_image(width * height);

  ASSERT_TRUE(image_processing::color_convert::kernels::rgba_2_gray(
      input_image.data(), output_image.data(), width, height,
      image_processing::color_convert::AlgoType::kSimdCpu,
      image_processing::color_convert::MemLayout::Packed))
      << "Expected rgba_2_gray to return true but it returned false.";

  EXPECT_TRUE(std::abs(output_image[100 * width + 100] - 76) <= 2)
      << "Expected " << static_cast<int>(output_image[100 * width + 100])
      << " to be equal to 76 but it was not.";
  EXPECT_TRUE(std::abs(output_image[50 * width + 400] - 29) <= 2)
      << "Expected " << static_cast<int>(output_image[50 * width + 400])
      << " to be equal to 29 but it was not.";
}

TEST(RGBA2GrayTest, PlanarSIMDConversion) {
  int width = 1920;
  int height = 1080;
  auto input_image = read_raw_image("/tmp/geometric_image.rgba", width, height);
  std::vector<unsigned char> output_image(width * height);
  auto input_image_planar = input_image;

  for (int index = 0; index < height * width; index++) {
    input_image_planar[index] = input_image[index * 4];
    input_image_planar[index + height * width] = input_image[index * 4 + 1];
    input_image_planar[index + 2 * height * width] = input_image[index * 4 + 2];
  }

  ASSERT_TRUE(image_processing::color_convert::kernels::rgba_2_gray(
      input_image_planar.data(), output_image.data(), width, height,
      image_processing::color_convert::AlgoType::kSimdCpu,
      image_processing::color_convert::MemLayout::Planar))
      << "Expected rgba_2_gray to return true but it returned false.";

  EXPECT_TRUE(std::abs(output_image[100 * width + 100] - 76) <= 2)
      << "Expected " << static_cast<int>(output_image[100 * width + 100])
      << " to be equal to 76 but it was not.";
  EXPECT_TRUE(std::abs(output_image[50 * width + 400] - 29) <= 2)
      << "Expected " << static_cast<int>(output_image[50 * width + 400])
      << " to be equal to 29 but it was not.";
}

#if HAS_CUDA

TEST(RGBA2GrayTest, PackedCUDAConversion) {
  int width = 1920;
  int height = 1080;
  auto input_image = read_raw_image("/tmp/geometric_image.rgba", width, height);
  std::vector<unsigned char> output_image(width * height);

  ASSERT_TRUE(image_processing::color_convert::kernels::rgba_2_gray(
      input_image.data(), output_image.data(), width, height,
      image_processing::color_convert::AlgoType::kCuda,
      image_processing::color_convert::MemLayout::Packed))
      << "Expected rgba_2_gray to return true but it returned false.";

  EXPECT_TRUE(std::abs(output_image[100 * width + 100] - 76) <= 2)
      << "Expected " << static_cast<int>(output_image[100 * width + 100])
      << " to be equal to 76 but it was not.";
  EXPECT_TRUE(std::abs(output_image[50 * width + 400] - 29) <= 2)
      << "Expected " << static_cast<int>(output_image[50 * width + 400])
      << " to be equal to 29 but it was not.";
}

TEST(RGBA2GrayTest, PlanarCUDAConversion) {
  int width = 1920;
  int height = 1080;
  auto input_image = read_raw_image("/tmp/geometric_image.rgba", width, height);
  std::vector<unsigned char> output_image(width * height);
  auto input_image_planar = input_image;

  for (int index = 0; index < height * width; index++) {
    input_image_planar[index] = input_image[index * 4];
    input_image_planar[index + height * width] = input_image[index * 4 + 1];
    input_image_planar[index + 2 * height * width] = input_image[index * 4 + 2];
  }

  ASSERT_TRUE(image_processing::color_convert::kernels::rgba_2_gray(
      input_image_planar.data(), output_image.data(), width, height,
      image_processing::color_convert::AlgoType::kCuda,
      image_processing::color_convert::MemLayout::Planar))
      << "Expected rgba_2_gray to return true but it returned false.";

  EXPECT_TRUE(std::abs(output_image[100 * width + 100] - 76) <= 2)
      << "Expected " << static_cast<int>(output_image[100 * width + 100])
      << " to be equal to 76 but it was not.";
  EXPECT_TRUE(std::abs(output_image[50 * width + 400] - 29) <= 2)
      << "Expected " << static_cast<int>(output_image[50 * width + 400])
      << " to be equal to 29 but it was not.";
}

#endif