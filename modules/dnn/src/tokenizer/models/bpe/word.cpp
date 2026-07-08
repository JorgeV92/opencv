// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "../../../precomp.hpp"
#include "word.hpp"

#include <algorithm>
#include <queue>
#include <random>

namespace cv { namespace dnn {

Word::Word(int capacity)
{
    symbols_.reserve(capacity);
}

void Word::add(std::uint32_t c, std::size_t bytelen)
{
    int prev = -1;
    const int next = -1;
    const int len = static_cast<int>(symbols_.size());

    if (!symbols_.empty())
    {
        Symbol& last = symbols_.back();
        last.next = len;
        prev = len - 1;
    }

    symbols_.push_back(Symbol{c, prev, next, bytelen});
}

std::vector<std::pair<Pair, std::int32_t>> Word::merge(
    std::uint32_t c1,
    std::uint32_t c2,
    std::uint32_t replacement,
    std::size_t maxLength)
{
    std::vector<std::pair<Pair, std::int32_t>> changes;
    std::size_t i = 0;
    while (i < symbols_.size())
    {
        if (symbols_[i].c == c1 && i + 1 < symbols_.size() && symbols_[i + 1].c == c2)
        {
            const Symbol first = symbols_[i];
            const Symbol second = symbols_[i + 1];
            const Symbol newSymbol{replacement, first.prev, second.next, first.len + second.len};

            if (i > 0)
            {
                changes.emplace_back(Pair{symbols_[i - 1].c, first.c}, -1);
                if (symbols_[i - 1].len + newSymbol.len < maxLength)
                    changes.emplace_back(Pair{symbols_[i - 1].c, replacement}, 1);
            }

            symbols_.insert(symbols_.begin() + i, newSymbol);
            symbols_.erase(symbols_.begin() + i + 1);
            symbols_.erase(symbols_.begin() + i + 1);

            if (i + 1 < symbols_.size())
            {
                changes.emplace_back(Pair{second.c, symbols_[i + 1].c}, -1);
                if (symbols_[i + 1].len + newSymbol.len < maxLength)
                    changes.emplace_back(Pair{replacement, symbols_[i + 1].c}, 1);
            }
        }
        ++i;
    }
    return changes;
}

void Word::mergeAll(const MergeMap& merges, std::optional<float> dropout)
{
    using Queue = std::priority_queue<MergeCandidate, std::vector<MergeCandidate>, MergeCandidateComp>;
    Queue queue;
    std::vector<MergeCandidate> skipped;
    skipped.reserve(symbols_.size());

    for (std::size_t i = 0; i + 1 < symbols_.size(); ++i)
    {
        const auto rule = merges.find(Pair{symbols_[i].c, symbols_[i + 1].c});
        if (rule != merges.end())
            queue.push(MergeCandidate{i, rule->second.rank, rule->second.newId});
    }

    static thread_local std::mt19937 generator{std::random_device{}()};
    std::uniform_real_distribution<float> distribution(0.0f, 1.0f);

    while (!queue.empty())
    {
        const MergeCandidate top = queue.top();
        queue.pop();

        if (dropout && distribution(generator) < *dropout)
        {
            skipped.push_back(top);
            continue;
        }

        for (const MergeCandidate& candidate : skipped)
            queue.push(candidate);
        skipped.clear();

        if (top.pos >= symbols_.size())
            continue;

        Symbol& left = symbols_[top.pos];
        if (left.len == 0 || left.next < 0)
            continue;

        const std::size_t rightPos = static_cast<std::size_t>(left.next);
        if (rightPos >= symbols_.size())
            continue;

        const Symbol right = symbols_[rightPos];
        const auto currRule = merges.find(Pair{left.c, right.c});
        if (currRule == merges.end() || currRule->second.newId != top.new_id)
            continue;

        left.mergeWith(right, top.new_id);
        symbols_[rightPos].len = 0;

        if (right.next >= 0)
        {
            const std::size_t followingPos = static_cast<std::size_t>(right.next);
            if (followingPos < symbols_.size())
                symbols_[followingPos].prev = static_cast<int>(top.pos);
        }

        if (left.prev >= 0)
        {
            const std::size_t prevPos = static_cast<std::size_t>(left.prev);
            const auto rule = merges.find(Pair{symbols_[prevPos].c, left.c});
            if (rule != merges.end())
                queue.push(MergeCandidate{prevPos, rule->second.rank, rule->second.newId});
        }

        if (left.next >= 0)
        {
            const std::size_t followingPos = static_cast<std::size_t>(left.next);
            if (followingPos < symbols_.size())
            {
                const auto rule = merges.find(Pair{left.c, symbols_[followingPos].c});
                if (rule != merges.end())
                    queue.push(MergeCandidate{top.pos, rule->second.rank, rule->second.newId});
            }
        }
    }

    symbols_.erase(
        std::remove_if(symbols_.begin(), symbols_.end(),
                       [](const Symbol& symbol) { return symbol.len == 0; }),
        symbols_.end());

    for (std::size_t i = 0; i < symbols_.size(); ++i)
    {
        symbols_[i].prev = i == 0 ? -1 : static_cast<int>(i - 1);
        symbols_[i].next = i + 1 < symbols_.size() ? static_cast<int>(i + 1) : -1;
    }
}

std::vector<std::uint32_t> Word::get_chars() const
{
    std::vector<std::uint32_t> chars;
    chars.reserve(symbols_.size());
    for (const Symbol& symbol : symbols_)
        chars.push_back(symbol.c);
    return chars;
}

std::vector<std::uint32_t> Word::get_chars_iter() const
{
    return get_chars();
}

std::vector<std::pair<std::size_t, std::size_t>> Word::get_offsets_iter() const
{
    std::vector<std::pair<std::size_t, std::size_t>> offsets;
    offsets.reserve(symbols_.size());

    std::size_t pos = 0;
    for (const Symbol& symbol : symbols_)
    {
        const std::size_t newPos = pos + symbol.len;
        offsets.emplace_back(pos, newPos);
        pos = newPos;
    }
    return offsets;
}

}} // namespace cv::dnn
