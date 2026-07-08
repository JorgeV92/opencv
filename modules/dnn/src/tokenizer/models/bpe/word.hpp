// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#ifndef __OPENCV_DNN_TOKENIZER_MODELS_BPE_WORD_HPP__
#define __OPENCV_DNN_TOKENIZER_MODELS_BPE_WORD_HPP__

#include <opencv2/core/cvdef.h>

#include <cstddef>
#include <cstdint>
#include <optional>
#include <unordered_map>
#include <utility>
#include <vector>

namespace cv { namespace dnn {

using Pair = std::pair<std::uint32_t, std::uint32_t>;

struct PairHash
{
    std::size_t operator()(const Pair& p) const
    {
        return (static_cast<std::size_t>(p.first) << 32) ^ p.second;
    }
};

struct MergeRule
{
    std::uint32_t rank;
    std::uint32_t newId;
};

using MergeMap = std::unordered_map<Pair, MergeRule, PairHash>;

struct MergeCandidate
{
    std::size_t pos;
    std::uint32_t rank;
    std::uint32_t new_id;
};

struct MergeCandidateComp
{
    bool operator()(const MergeCandidate& lhs, const MergeCandidate& rhs) const noexcept
    {
        if (lhs.rank != rhs.rank)
            return lhs.rank > rhs.rank;
        return lhs.pos > rhs.pos;
    }
};

struct Symbol
{
    std::uint32_t c{};
    int prev{-1};
    int next{-1};
    std::size_t len{};

    void mergeWith(const Symbol& o, std::uint32_t new_c)  noexcept
    {
        c = new_c;
        len += o.len;
        next = o.next;
    }
};

class CV_EXPORTS Word
{
public:
    Word() = default;
    explicit Word(int capacity);

    void add(std::uint32_t c, std::size_t bytelen);

    void mergeAll(const MergeMap& merges, std::optional<float> dropout);

    std::vector<std::pair<Pair, std::int32_t>>
    merge(std::uint32_t c1, std::uint32_t c2, std::uint32_t replacement, std::size_t maxLength);

    std::vector<std::uint32_t> get_chars() const;
    std::vector<std::uint32_t> get_chars_iter() const;
    std::vector<std::pair<std::size_t, std::size_t>> get_offsets_iter() const;

private:
    std::vector<Symbol> symbols_;
};

}} // namespace cv::dnn

#endif // __OPENCV_DNN_TOKENIZER_MODELS_BPE_WORD_HPP__
