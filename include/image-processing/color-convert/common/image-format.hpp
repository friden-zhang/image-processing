#pragma once

#include <cstddef>
#include <cstdint>

#include <strings.h>
#include <type_traits>

#include "image-processing/color-convert/common/vector-type.hpp"

namespace image_processing {

namespace color_convert {

/**
 * The ImageFormat enum is used to identify the pixel format and colorspace
 * of an image.  Supported data types are based on `uint8` and `float`, with
 * colorspaces including RGB/RGBA, BGR/BGRA, grayscale, YUV, and Bayer.
 *
 * There are also a variety of helper functions available that provide info
 * about each format at runtime - for example, the pixel bit depth
 * (image_format_depth()) the number of image channels
 * (image_format_channels()), and computing the size of an image from it's
 * dimensions (@see image_format_size()).  To convert between image formats
 * using the GPU, there is also the cudaConvertColor() function.
 *
 * In addition to the enums below, each format can also be identified by a
 * string. The string corresponding to each format is included in the
 * documentation below. These strings are more commonly used from Python, but
 * can also be used from C++ with the image_format_from_str() and
 * image_format_to_str() functions.
 *
 * @ingroup ImageFormat
 */
enum class ImageFormat {
  // RGB
  IMAGE_RGB8 = 0, /**< uchar3 RGB8    (`'rgb8'`) */
  IMAGE_RGBA8,    /**< uchar4 RGBA8   (`'rgba8'`) */
  IMAGE_RGB32F,   /**< float3 RGB32F  (`'rgb32f'`) */
  IMAGE_RGBA32F,  /**< float4 RGBA32F (`'rgba32f'`) */

  // BGR
  IMAGE_BGR8,    /**< uchar3 BGR8    (`'bgr8'`) */
  IMAGE_BGRA8,   /**< uchar4 BGRA8   (`'bgra8'`) */
  IMAGE_BGR32F,  /**< float3 BGR32F  (`'bgr32f'`) */
  IMAGE_BGRA32F, /**< float4 BGRA32F (`'bgra32f'`) */

  // YUV
  IMAGE_YUYV,              /**< YUV YUYV 4:2:2 packed (`'yuyv'`) */
  IMAGE_YUY2 = IMAGE_YUYV, /**< Duplicate of YUYV     (`'yuy2'`) */
  IMAGE_YVYU,              /**< YUV YVYU 4:2:2 packed (`'yvyu'`) */
  IMAGE_UYVY,              /**< YUV UYVY 4:2:2 packed (`'uyvy'`) */
  IMAGE_I420,              /**< YUV I420 4:2:0 planar (`'i420'`) */
  IMAGE_YV12,              /**< YUV YV12 4:2:0 planar (`'yv12'`) */
  IMAGE_NV12,              /**< YUV NV12 4:2:0 planar (`'nv12'`) */

  // Bayer
  IMAGE_BAYER_BGGR, /**< 8-bit Bayer BGGR (`'bayer-bggr'`) */
  IMAGE_BAYER_GBRG, /**< 8-bit Bayer GBRG (`'bayer-gbrg'`) */
  IMAGE_BAYER_GRBG, /**< 8-bit Bayer GRBG (`'bayer-grbg'`) */
  IMAGE_BAYER_RGGB, /**< 8-bit Bayer RGGB (`'bayer-rggb'`) */

  // grayscale
  IMAGE_GRAY8,   /**< uint8 grayscale  (`'gray8'`)   */
  IMAGE_GRAY32F, /**< float grayscale  (`'gray32f'`) */

  // extras
  IMAGE_COUNT,                  /**< The number of image formats */
  IMAGE_UNKNOWN = 999,          /**< Unknown/undefined format */
  IMAGE_DEFAULT = IMAGE_RGBA32F /**< Default format (IMAGE_RGBA32F) */
};

/**
 * The imageBaseType enum is used to identify the base data type of an
 * ImageFormat - either uint8 or float.  For example, the IMAGE_RGB8
 * format has a base type of uint8, while IMAGE_RGB32F is float.
 *
 * You can retrieve the base type of each format with
 * image_format_to_base_type()
 *
 * @ingroup ImageFormat
 */
enum class ImageBaseType { IMAGE_UINT8, IMAGE_FLOAT };

/**
 * Get the base type of an image format (uint8 or float).
 * @see ImageBaseType
 * @ingroup ImageFormat
 */
inline ImageBaseType image_format_to_base_type(ImageFormat format) {
  switch (format) {
  case ImageFormat::IMAGE_GRAY32F:
  case ImageFormat::IMAGE_RGB32F:
  case ImageFormat::IMAGE_BGR32F:
  case ImageFormat::IMAGE_RGBA32F:
  case ImageFormat::IMAGE_BGRA32F:
    return ImageBaseType::IMAGE_FLOAT;
  }

  return ImageBaseType::IMAGE_UINT8;
}

/**
 * Convert an ImageFormat enum to a string.
 * @see ImageFormat for the strings that correspond to each format.
 * @ingroup ImageFormat
 */
inline const char *image_format_to_str(ImageFormat format) {
  switch (format) {
  case ImageFormat::IMAGE_RGB8:
    return "rgb8";
  case ImageFormat::IMAGE_RGBA8:
    return "rgba8";
  case ImageFormat::IMAGE_RGB32F:
    return "rgb32f";
  case ImageFormat::IMAGE_RGBA32F:
    return "rgba32f";
  case ImageFormat::IMAGE_BGR8:
    return "bgr8";
  case ImageFormat::IMAGE_BGRA8:
    return "bgra8";
  case ImageFormat::IMAGE_BGR32F:
    return "bgr32f";
  case ImageFormat::IMAGE_BGRA32F:
    return "bgra32f";
  case ImageFormat::IMAGE_I420:
    return "i420";
  case ImageFormat::IMAGE_YV12:
    return "yv12";
  case ImageFormat::IMAGE_NV12:
    return "nv12";
  case ImageFormat::IMAGE_UYVY:
    return "uyvy";
  case ImageFormat::IMAGE_YUYV:
    return "yuyv";
  case ImageFormat::IMAGE_YVYU:
    return "yvyu";
  case ImageFormat::IMAGE_BAYER_BGGR:
    return "bayer-bggr";
  case ImageFormat::IMAGE_BAYER_GBRG:
    return "bayer-gbrg";
  case ImageFormat::IMAGE_BAYER_GRBG:
    return "bayer-grbg";
  case ImageFormat::IMAGE_BAYER_RGGB:
    return "bayer-rggb";
  case ImageFormat::IMAGE_GRAY8:
    return "gray8";
  case ImageFormat::IMAGE_GRAY32F:
    return "gray32f";
  case ImageFormat::IMAGE_UNKNOWN:
    return "unknown";
  };

  return "unknown";
}

/**
 * Convert a string to an ImageFormat enum.
 * @see ImageFormat for the strings that correspond to each format.
 * @ingroup ImageFormat
 */
inline ImageFormat image_format_from_str(const char *str) {
  if (!str)
    return ImageFormat::IMAGE_UNKNOWN;

  for (uint32_t n = 0; n < static_cast<uint32_t>(ImageFormat::IMAGE_COUNT);
       n++) {
    const ImageFormat fmt = (ImageFormat)n;

    if (strcasecmp(str, image_format_to_str(fmt)) == 0)
      return fmt;
  }

  if (strcasecmp(str, "yuy2") == 0)
    return ImageFormat::IMAGE_YUY2;
  else if (strcasecmp(str, "rgb32") == 0)
    return ImageFormat::IMAGE_RGB32F;
  else if (strcasecmp(str, "rgba32") == 0)
    return ImageFormat::IMAGE_RGBA32F;
  else if (strcasecmp(str, "grey8") == 0)
    return ImageFormat::IMAGE_GRAY8;
  else if (strcasecmp(str, "grey32f") == 0)
    return ImageFormat::IMAGE_GRAY32F;

  return ImageFormat::IMAGE_UNKNOWN;
}

/**
 * @brief Get the number of bits per pixel for a given image format.
 *
 * @param format
 * @return size_t
 */
inline size_t image_format_channels(ImageFormat format) {
  switch (format) {
  case ImageFormat::IMAGE_RGB8:
  case ImageFormat::IMAGE_RGB32F:
  case ImageFormat::IMAGE_BGR8:
  case ImageFormat::IMAGE_BGR32F:
    return 3;
  case ImageFormat::IMAGE_RGBA8:
  case ImageFormat::IMAGE_RGBA32F:
  case ImageFormat::IMAGE_BGRA8:
  case ImageFormat::IMAGE_BGRA32F:
    return 4;
  case ImageFormat::IMAGE_GRAY8:
  case ImageFormat::IMAGE_GRAY32F:
    return 1;
  case ImageFormat::IMAGE_I420:
  case ImageFormat::IMAGE_YV12:
  case ImageFormat::IMAGE_NV12:
  case ImageFormat::IMAGE_UYVY:
  case ImageFormat::IMAGE_YUYV:
  case ImageFormat::IMAGE_YVYU:
    return 3;
  case ImageFormat::IMAGE_BAYER_BGGR:
  case ImageFormat::IMAGE_BAYER_GBRG:
  case ImageFormat::IMAGE_BAYER_GRBG:
  case ImageFormat::IMAGE_BAYER_RGGB:
    return 1;
  }

  return 0;
}

/**
 * @brief Check if an image format is a RGB format.
 *
 * @param format
 * @return true
 * @return false
 */
inline bool image_format_is_rgb(ImageFormat format) {
  // if( format == IMAGE_RGB8 || format == IMAGE_RGBA8 || format == IMAGE_RGB32F
  // || format == IMAGE_RGBA32F ) 	return true;
  if (format >= ImageFormat::IMAGE_RGB8 && format <= ImageFormat::IMAGE_RGBA32F)
    return true;

  return false;
}

/**
 * @brief Check if an image format is a BGR format.
 *
 * @param format
 * @return true
 * @return false
 */
inline bool image_format_is_bgr(ImageFormat format) {
  if (format >= ImageFormat::IMAGE_BGR8 && format <= ImageFormat::IMAGE_BGRA32F)
    return true;

  return false;
}

/**
 * @brief Check if an image format is a YUV format.
 *
 * @param format
 * @return true
 * @return false
 */
inline bool image_format_is_yuv(ImageFormat format) {
  if (format >= ImageFormat::IMAGE_YUYV && format <= ImageFormat::IMAGE_NV12)
    return true;

  return false;
}

/**
 * @brief Check if an image format is a Grayscale format.
 *
 * @param format
 * @return true
 * @return false
 */
inline bool image_format_is_gray(ImageFormat format) {
  if (format == ImageFormat::IMAGE_GRAY8 ||
      format == ImageFormat::IMAGE_GRAY32F)
    return true;

  return false;
}

/**
 * @brief Check if an image format is a Bayer format.
 *
 * @param format
 * @return true
 * @return false
 */
inline bool image_format_is_bayer(ImageFormat format) {
  if (format >= ImageFormat::IMAGE_BAYER_BGGR &&
      format <= ImageFormat::IMAGE_BAYER_RGGB)
    return true;

  return false;
}

/**
 * @brief Get the number of bits per pixel for a given image format.
 *
 * @param format
 * @return size_t
 */
inline size_t image_format_depth(ImageFormat format) {

  switch (format) {
  case ImageFormat::IMAGE_RGB8:
  case ImageFormat::IMAGE_BGR8:
    return sizeof(uchar3) * 8;
  case ImageFormat::IMAGE_RGBA8:
  case ImageFormat::IMAGE_BGRA8:
    return sizeof(uchar4) * 8;
  case ImageFormat::IMAGE_RGB32F:
  case ImageFormat::IMAGE_BGR32F:
    return sizeof(float3) * 8;
  case ImageFormat::IMAGE_RGBA32F:
  case ImageFormat::IMAGE_BGRA32F:
    return sizeof(float4) * 8;
  case ImageFormat::IMAGE_GRAY8:
    return sizeof(unsigned char) * 8;
  case ImageFormat::IMAGE_GRAY32F:
    return sizeof(float) * 8;
  case ImageFormat::IMAGE_I420:
  case ImageFormat::IMAGE_YV12:
  case ImageFormat::IMAGE_NV12:
    return 12;
  case ImageFormat::IMAGE_UYVY:
  case ImageFormat::IMAGE_YUYV:
  case ImageFormat::IMAGE_YVYU:
    return 16;
  case ImageFormat::IMAGE_BAYER_BGGR:
  case ImageFormat::IMAGE_BAYER_GBRG:
  case ImageFormat::IMAGE_BAYER_GRBG:
  case ImageFormat::IMAGE_BAYER_RGGB:
    return sizeof(unsigned char) * 8;
  }

  return 0;
}

/**
 * @brief Get the size of an image in bytes for a given image format and
 * dimensions.
 *
 * @param format
 * @param width
 * @param height
 * @return size_t
 */
inline size_t image_format_size(ImageFormat format, size_t width,
                                size_t height) {
  return (width * height * image_format_depth(format)) / 8;
}

template <typename T> struct __image_format_assert_false : std::false_type {};

template <typename T> inline ImageFormat image_format_from_type() {
  static_assert(__image_format_assert_false<T>::value,
                "invalid image format type - supported types are uchar3, "
                "uchar4, float3, float4");
}

template <> inline ImageFormat image_format_from_type<uchar3>() {
  return ImageFormat::IMAGE_RGB8;
}

template <> inline ImageFormat image_format_from_type<uchar4>() {
  return ImageFormat::IMAGE_RGBA8;
}

template <> inline ImageFormat image_format_from_type<float3>() {
  return ImageFormat::IMAGE_RGB32F;
}

template <> inline ImageFormat image_format_from_type<float4>() {
  return ImageFormat::IMAGE_RGBA32F;
}

template <ImageFormat format> struct ImageFormatType {
  static_assert(format == ImageFormat::IMAGE_RGB8 ||
                    format == ImageFormat::IMAGE_RGBA8 ||
                    format == ImageFormat::IMAGE_RGB32F ||
                    format == ImageFormat::IMAGE_RGBA32F,
                "invalid image format type - supported types are IMAGE_RGB8, "
                "IMAGE_RGBA8, IMAGE_RGB32F, IMAGE_RGBA32F");
};

template <> struct ImageFormatType<ImageFormat::IMAGE_RGB8> {
  typedef uint8_t Base;
  typedef uchar3 Vector;
};
template <> struct ImageFormatType<ImageFormat::IMAGE_RGBA8> {
  typedef uint8_t Base;
  typedef uchar4 Vector;
};

template <> struct ImageFormatType<ImageFormat::IMAGE_RGB32F> {
  typedef float Base;
  typedef float3 Vector;
};
template <> struct ImageFormatType<ImageFormat::IMAGE_RGBA32F> {
  typedef float Base;
  typedef float4 Vector;
};

} // namespace color_convert

} // namespace image_processing
