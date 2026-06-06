
#ifndef __OPENCV_DNN_TOKENIZER_NORMALIZER_HPP__
#define __OPENCV_DNN_TOKENIZER_NORMALIZER_HPP__

#include <string>
#include <string_view>
#include <vector>
#include <optional>

namespace cv { namespace dnn {

struct offset {
    std::size_t begin;
    std::size_t end;
};

class NormalizedString {
public:
    explicit NormalizedString(std::string org) 
        : org_(std::move(org)), normalized_(org_) {
        rebuildIdentityAlignment();
    }

private:
    void rebuildIdentityAlignment() {
        aligment_.clear();
        aligment_.reserve(normalized_.size());
        for (std::size_t i{0}; i < (int)normalized_.size(); ++i) {
            aligment_.push_back(offset{i,i+1});
        }
    }

    const std::string& orginal() const {
        return org_;
    }

    const std::string& nomalized() const {
        return normalized_;
    }

    void lowercase() { 
        // TODO
    }

    void collapseWhitespace() {
        // TODO
    }

    std::optional<offset> normalzedToOriginal(offset nomalizedOffset) const {
        // TODO
        // conver normalzied span back to original span.
        return std::nullopt;
    }

    std::optional<offset> originalToNormalized(offset originalOffset) const {
        // TODO
        // convert original span to normalized span
        return std::nullopt;
    }


private:
    std::string org_;
    std::string normalized_;
    std::vector<offset> aligment_;


};

class Normalizer {
public:
    virtual ~Normalizer() = default;
    virtual std::string normalize(std::string_view input) const = 0;
    virtual void normalize(NormalizedString&) const = 0;
};

}} // namespace cv::dnn

#endif //__OPENCV_DNN_TOKENIZER_NORMALIZER_HPP__ 