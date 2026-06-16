// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#ifndef __OPENCV_DNN_TOKENIZER_WORD_HPP__
#define __OPENCV_DNN_TOKENIZER_WORD_HPP__

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <queue>
#include <random>
#include <stdexcept>
#include <unordered_map>
#include <utility>
#include <vector>

#include "bpe.hpp"

namespace cv { namespace dnn {

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

// TODO
class Word
{
public:

    Word() =default;
    explicit Word(int capacity) { symbols_.reserve(capacity); }

    void add(std::uint32_t c, std::size_t bytelen);

    void mergeAll(const MergeMap& merges, std::optional<float> dropout);

    std::vector<std::pair<Pair, std::uint32_t>>
    merge(std::uint32_t c1, std::uint32_t c2, std::uint32_t replacement, std::size_t maxLength);

private:
    std::vector<Symbol> symbols_;
};

inline void Word::add(std::uint32_t c, std::size_t bytelen)
{
    int prev = -1;
    int next = -1;

    int len = (int)symbols_.size();

    if (!symbols_.empty())
    {
        Symbol& last = symbols_.back();
        last.next = len;
        prev = len-1;

    }

    symbols_.push_back(Symbol{
        .c = c,
        .prev = prev,
        .next = next,
        .len = bytelen,
    });
}

/// Merges adjacent (c1, c2) pairs into `replacement`.
/// Returns neighboring pair-frequency changes as {pair, delta}.
inline std::vector<std::pair<Pair, std::int32_t>>
Word::merge(
    std::uint32_t c1,
    std::uint32_t c2,
    std::uint32_t replacement,
    std::size_t maxLength)
{
    std::vector<std::pair<Pair, std::int32_t>> changes;
    std::size_t i = 0;
    while (i < symbols_.size())
    {
        if (symbols_[i].c == c1 && i + 1 < symbols_.size() && symbols_[i+1].c == c2)
        {
            auto first = symbols_[i];
            auto second = symbols_[i+1];

            Symbol nS{
                .c = replacement,
                .prev = first.prev,
                .next = second.next,
                .len = first.len + second.len,
            };

            if (i > 0)
            {
                changes.emplace_back(Pair{symbols_[i-1].c, first.c}, -1);

                if (symbols_[i-1].len + nS.len < maxLength)
                {
                    changes.emplace_back(Pair{symbols_[i-1].c, replacement}, 1);
                }
            }

            symbols_.insert(symbols_.begin()+i, nS);
            symbols_.erase(symbols_.begin()+i+1);
            symbols_.erase(symbols_.begin()+i+1);

            if (i + 1 < symbols_.size())
            {
                changes.emplace_back(Pair{second.c, symbols_[i+1].c}, -1);

                if (symbols_[i+1].len + nS.len < maxLength)
                {
                    changes.emplace_back(Pair{replacement, symbols_[i+1].c}, -1);
                }
            }
        }
        ++i;
    }

    return changes;
}

inline void Word::mergeAll(const MergeMap& merges, std::optional<float> dropout)
{
    if (dropout && (*dropout < 0.0f || *dropout > 1.0f))
        throw std::invalid_argument("error");

        using Queue = std::priority_queue<MergeCandidate, std::vector<MergeCandidate>, MergeCandidateComp>;

        Queue queue;

        std::vector<MergeCandidate> skipped;
        skipped.reserve(symbols_.size());

        for (std::size_t i = 0; i + 1 < symbols_.size(); ++i)
        {
            const Pair pp{ symbols_[i].c, symbols_[i+1].c};

            const auto rule = merges.find(pp);
            if (rule != merges.end())
            {
                queue.push(MergeCandidate{
                    .pos = i,
                    .rank = rule->second.rank,
                    .new_id = rule->second.newId
                });
            }
        }

        static thread_local std::mt19937 generator{std::random_device{}()};
        std::uniform_real_distribution<float> distribution(0.0f, 1.0f);

        while (!queue.empty())
        {
            MergeCandidate top = queue.top();
            queue.pop();

            if (dropout && distribution(generator) < *dropout)
            {
                skipped.push_back(top);
                continue;
            }

            for (const MergeCandidate& candidate : skipped)
            {
                queue.push(candidate);
            }
            skipped.clear();

            if (top.pos >= symbols_.size()) 
                continue;

            Symbol& left = symbols_[top.pos];

            if (left.len == 0) continue;
            if (left.next < 0) continue;

            const std::size_t rightPos = static_cast<std::size_t>(left.next);
            if (rightPos >= symbols_.size()) continue;

            const Symbol right = symbols_[rightPos];
            const Pair currPair{left.c, right.c};

            const auto currRule = merges.find(currPair);
            if (currRule == merges.end() || currRule->second.newId != top.new_id) 
                continue;
            
            left.mergeWith(right, top.new_id);
            symbols_[rightPos].len = 0;

            if (right.next >= 0) 
            {
                const std::size_t followingPos = static_cast<std::size_t>(right.next);
                if (followingPos < symbols_.size()) 
                {
                    symbols_[followingPos].prev = static_cast<int>(top.pos);
                }
            }
            
            if (left.prev >= 0) 
            {
                const std::size_t prevPos =
                    static_cast<std::size_t>(left.prev);

                const Pair newLeftPair{symbols_[prevPos].c, left.c};
                const auto rule = merges.find(newLeftPair);
                if (rule != merges.end()) 
                {
                    queue.push(MergeCandidate{
                        .pos = prevPos,
                        .rank = rule->second.rank,
                        .new_id = rule->second.newId,
                    });
                }
            }

            if (left.next >= 0) 
            {
                const std::size_t followingPos = 
                    static_cast<std::size_t>(left.next);
                if (followingPos < symbols_.size()) 
                {
                    const Pair newRightPair{left.c, symbols_[followingPos].c};
                    const auto rule = megres.find(newRightPair);
                    if (rule != megres.end()) 
                    {
                        queue.push(MergeCandidate{
                            .pos = top.pos,
                            .rank = rule->second.rank,
                            .new_id = rule->second.newId,
                        });
                    }
                }
            }

        }

        std::erase_if(
            symbols_,
            [](const Symbol& sym) 
            {
                return sym.len == 0;
            }
        );

        for (std::size_t i = 0; i < symbols_.size(); ++i) 
        {
            symbols_[i].prev = 
                i == 0
                ? -1 
                : static_cast<int>(i-1);
            
            symbols_[i].next = 
                i + 1 < symbols_.size()
                ? static_cast<int>(i + 1)
                : -1;
        }
}

}} // namespace cv::dnn

#endif // __OPENCV_DNN_TOKENIZER_WORD_HPP__
