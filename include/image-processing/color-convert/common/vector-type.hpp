#pragma once

#include "image-processing/color-convert/common/marco.hpp"

namespace image_processing {
namespace color_convert {

// define float4, float3, uchar4, uchar3

struct float4 {
  float x, y, z, w;
};

struct float3 {
  float x, y, z;
};

struct uchar4 {
  unsigned char x, y, z, w;
};

struct uchar3 {
  unsigned char x, y, z;
};

template <typename T> HOST_DEVICE float4 make_float4(T x, T y, T z, T w) {
  return {static_cast<float>(x), static_cast<float>(y), static_cast<float>(z),
          static_cast<float>(w)};
}

template <typename T> HOST_DEVICE float3 make_float3(T x, T y, T z) {
  return {static_cast<float>(x), static_cast<float>(y), static_cast<float>(z)};
}

template <typename T> HOST_DEVICE uchar4 make_uchar4(T x, T y, T z, T w) {
  return {static_cast<unsigned char>(x), static_cast<unsigned char>(y),
          static_cast<unsigned char>(z), static_cast<unsigned char>(w)};
}

template <typename T> HOST_DEVICE uchar3 make_uchar3(T x, T y, T z) {
  return {static_cast<unsigned char>(x), static_cast<unsigned char>(y),
          static_cast<unsigned char>(z)};
}

} // namespace color_convert
} // namespace image_processing