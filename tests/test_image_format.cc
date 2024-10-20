#include "image-processing/color-convert/format/image-format.hpp"
#include <gtest/gtest.h>

using namespace image_processing::color_convert;

TEST(ImageFormatTest, EnumToString) {
  EXPECT_STREQ(ImageFormatToStr(ImageFormat::IMAGE_RGB8), "rgb8");
  EXPECT_STREQ(ImageFormatToStr(ImageFormat::IMAGE_RGBA8), "rgba8");
  EXPECT_STREQ(ImageFormatToStr(ImageFormat::IMAGE_GRAY8), "gray8");
  EXPECT_STREQ(ImageFormatToStr(ImageFormat::IMAGE_UNKNOWN), "unknown");
}

TEST(ImageFormatTest, StringToEnum) {
  EXPECT_EQ(ImageFormatFromStr("rgb8"), ImageFormat::IMAGE_RGB8);
  EXPECT_EQ(ImageFormatFromStr("rgba8"), ImageFormat::IMAGE_RGBA8);
  EXPECT_EQ(ImageFormatFromStr("gray8"), ImageFormat::IMAGE_GRAY8);
  EXPECT_EQ(ImageFormatFromStr("unknown"), ImageFormat::IMAGE_UNKNOWN);
}

TEST(ImageFormatTest, BaseType) {
  EXPECT_EQ(ImageFormatBaseType(ImageFormat::IMAGE_RGB8),
            ImageBaseType::IMAGE_UINT8);
  EXPECT_EQ(ImageFormatBaseType(ImageFormat::IMAGE_RGB32F),
            ImageBaseType::IMAGE_FLOAT);
}

TEST(ImageFormatTest, ChannelCount) {
  EXPECT_EQ(ImageFormatChannels(ImageFormat::IMAGE_RGB8), 3);
  EXPECT_EQ(ImageFormatChannels(ImageFormat::IMAGE_RGBA8), 4);
  EXPECT_EQ(ImageFormatChannels(ImageFormat::IMAGE_GRAY8), 1);
  EXPECT_EQ(ImageFormatChannels(ImageFormat::IMAGE_I420), 3);
}

TEST(ImageFormatTest, Depth) {
  EXPECT_EQ(ImageFormatDepth(ImageFormat::IMAGE_RGB8), sizeof(uchar3) * 8);
  EXPECT_EQ(ImageFormatDepth(ImageFormat::IMAGE_RGBA8), sizeof(uchar4) * 8);
  EXPECT_EQ(ImageFormatDepth(ImageFormat::IMAGE_RGB32F), sizeof(float3) * 8);
  EXPECT_EQ(ImageFormatDepth(ImageFormat::IMAGE_GRAY8),
            sizeof(unsigned char) * 8);
}

TEST(ImageFormatTest, Size) {
  size_t width = 640;
  size_t height = 480;
  EXPECT_EQ(ImageFormatSize(ImageFormat::IMAGE_RGB8, width, height),
            (width * height * ImageFormatDepth(ImageFormat::IMAGE_RGB8)) / 8);
}

TEST(ImageFormatTest, ImageFormatFromType) {
  EXPECT_EQ(ImageFormatFromType<uchar3>(), ImageFormat::IMAGE_RGB8);
  EXPECT_EQ(ImageFormatFromType<uchar4>(), ImageFormat::IMAGE_RGBA8);
  EXPECT_EQ(ImageFormatFromType<float3>(), ImageFormat::IMAGE_RGB32F);
  EXPECT_EQ(ImageFormatFromType<float4>(), ImageFormat::IMAGE_RGBA32F);
}
