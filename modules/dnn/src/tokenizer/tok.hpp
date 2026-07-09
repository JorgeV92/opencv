// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#ifndef __OPENCV_DNN_TOKENIZER_TOK_HPP__
#define __OPENCV_DNN_TOKENIZER_TOK_HPP__

#include <opencv2/core/cvdef.h>
#include <opencv2/dnn/version.hpp>

#include <cstddef>
#include <cstdint>
#include <string>
#include <utility>

namespace cv { namespace dnn {
CV__DNN_INLINE_NS_BEGIN

/**
 * @brief A token produced by a tokenizer model.
 *
 * Offsets form a half-open byte range `[begin, end)` in the input sequence.
 */
struct Token
{
    Token() = default;

    Token(std::uint32_t tokenId,
          std::string tokenValue,
          std::pair<std::size_t, std::size_t> tokenOffsets)
        : id(tokenId),
          value(std::move(tokenValue)),
          offsets(tokenOffsets)
    {
    }

    std::uint32_t id = 0;
    std::string value;
    std::pair<std::size_t, std::size_t> offsets{0, 0};
};

CV__DNN_INLINE_NS_END
}} // namespace cv::dnn

#endif // __OPENCV_DNN_TOKENIZER_TOK_HPP__
