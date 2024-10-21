#include "image-processing/color-convert/common/image-format.hpp"
#include <gtest/gtest.h>

using namespace image_processing::color_convert;

TEST(ImageFormatTest, EnumToString) {
  EXPECT_STREQ(image_format_to_str(ImageFormat::IMAGE_RGB8), "rgb8");
  EXPECT_STREQ(image_format_to_str(ImageFormat::IMAGE_RGBA8), "rgba8");
  EXPECT_STREQ(image_format_to_str(ImageFormat::IMAGE_GRAY8), "gray8");
  EXPECT_STREQ(image_format_to_str(ImageFormat::IMAGE_UNKNOWN), "unknown");
}

TEST(ImageFormatTest, StringToEnum) {
  EXPECT_EQ(image_format_from_str("rgb8"), ImageFormat::IMAGE_RGB8);
  EXPECT_EQ(image_format_from_str("rgba8"), ImageFormat::IMAGE_RGBA8);
  EXPECT_EQ(image_format_from_str("gray8"), ImageFormat::IMAGE_GRAY8);
  EXPECT_EQ(image_format_from_str("unknown"), ImageFormat::IMAGE_UNKNOWN);
}

TEST(ImageFormatTest, BaseType) {
  EXPECT_EQ(image_format_to_base_type(ImageFormat::IMAGE_RGB8),
            ImageBaseType::IMAGE_UINT8);
  EXPECT_EQ(image_format_to_base_type(ImageFormat::IMAGE_RGB32F),
            ImageBaseType::IMAGE_FLOAT);
}

TEST(ImageFormatTest, ChannelCount) {
  EXPECT_EQ(image_format_channels(ImageFormat::IMAGE_RGB8), 3);
  EXPECT_EQ(image_format_channels(ImageFormat::IMAGE_RGBA8), 4);
  EXPECT_EQ(image_format_channels(ImageFormat::IMAGE_GRAY8), 1);
  EXPECT_EQ(image_format_channels(ImageFormat::IMAGE_I420), 3);
}

TEST(ImageFormatTest, Depth) {
  EXPECT_EQ(image_format_depth(ImageFormat::IMAGE_RGB8), sizeof(uchar3) * 8);
  EXPECT_EQ(image_format_depth(ImageFormat::IMAGE_RGBA8), sizeof(uchar4) * 8);
  EXPECT_EQ(image_format_depth(ImageFormat::IMAGE_RGB32F), sizeof(float3) * 8);
  EXPECT_EQ(image_format_depth(ImageFormat::IMAGE_GRAY8),
            sizeof(unsigned char) * 8);
}

TEST(ImageFormatTest, Size) {
  size_t width = 640;
  size_t height = 480;
  EXPECT_EQ(image_format_size(ImageFormat::IMAGE_RGB8, width, height),
            (width * height * image_format_depth(ImageFormat::IMAGE_RGB8)) / 8);
}

TEST(ImageFormatTest, ImageFormatFromType) {
  EXPECT_EQ(image_format_from_type<uchar3>(), ImageFormat::IMAGE_RGB8);
  EXPECT_EQ(image_format_from_type<uchar4>(), ImageFormat::IMAGE_RGBA8);
  EXPECT_EQ(image_format_from_type<float3>(), ImageFormat::IMAGE_RGB32F);
  EXPECT_EQ(image_format_from_type<float4>(), ImageFormat::IMAGE_RGBA32F);
}
