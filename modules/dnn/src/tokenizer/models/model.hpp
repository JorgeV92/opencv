// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#ifndef __OPENCV_DNN_TOKENIZER_MODELS_MODEL_HPP__
#define __OPENCV_DNN_TOKENIZER_MODELS_MODEL_HPP__

#include "../tok.hpp"

#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

namespace cv { namespace dnn {

class CV_EXPORTS TokenizerModel
{
public:
    virtual ~TokenizerModel() = default;

    virtual std::vector<Token> tokenize(const std::string& sequence) const = 0;
    virtual std::optional<std::uint32_t> tokenToId(const std::string& token) const = 0;
    virtual std::optional<std::string> idToToken(std::uint32_t id) const = 0;
    virtual std::unordered_map<std::string, std::uint32_t> getVocab() const = 0;
    virtual std::size_t getVocabSize() const = 0;
};

}} // namespace cv::dnn

#endif // __OPENCV_DNN_TOKENIZER_MODELS_MODEL_HPP__
