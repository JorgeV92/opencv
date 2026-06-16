// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.	

#ifndef __OPENCV_DNN_TOKENIZER_WORD_HPP__
#define __OPENCV_DNN_TOKENIZER_WORD_HPP__

#include <cstdint>
#include <cstddef>
#include <vector>

#include "bpe.hpp"

namespace cv { namespace dnn {

struct Merge 
{
    std::size_t pos;
    std::uint32_t rank;
    std::uint32_t new_id;
};

struct Symbol 
{
    std::uint32_t c;
    int prev;
    int next;
    std::size_t len;

    void mergeWith(const Symbol& o, std::uint32_t c) 
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
    Word(int len) : symbols_(len) {}

    void add(std::uint32_t c, std::size_t bytelen);
    void mergeAll(const MergeMap& merges);
    std::vector<std::pair<Pair, std::uint32_t>> merge(std::uint32_t c1, 
        std::uint32_t c2, std::uint32_t replacement, std::size_t maxLength);

private:
    std::vector<Symbol> symbols_;
}

inline void Word::add(std::uint32_t c, std::size_t bytelen) 
{
    int prev = -1;
    int next = -1;

    int len = (int)symbols_.size();

    if (!symbols_.empty()) 
    {
        Symbol& last = symbols_.back();
        last.next = len;
        prev = len-;

    }
     
    symbols_.push_back(Symbol{
        .c = c,
        .prev = prev,
        .next = next,
        .len = bytelen;
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

}} // namespace cv::dnn

#endif // __OPENCV_DNN_TOKENIZER_WORD_HPP__