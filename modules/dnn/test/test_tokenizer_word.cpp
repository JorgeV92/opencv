// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "test_precomp.hpp"

#include "../src/tokenizer/models/bpe/word.hpp"

namespace opencv_test { namespace {

using cv::dnn::MergeMap;
using cv::dnn::MergeRule;
using cv::dnn::Pair;
using cv::dnn::Word;

static Word makeHelloWord()
{
    Word word;
    word.add(0, 1); // h
    word.add(1, 1); // e
    word.add(2, 1); // l
    word.add(2, 1); // l
    word.add(3, 1); // o
    return word;
}

TEST(Tokenizer_BPE_Word, Merge)
{
    Word word = makeHelloWord();

    const std::vector<std::pair<Pair, std::int32_t>> changes =
            word.merge(2, 2, 4, std::numeric_limits<std::size_t>::max());

    EXPECT_EQ(word.get_chars(), (std::vector<std::uint32_t>{0, 1, 4, 3}));
    EXPECT_EQ(changes, (std::vector<std::pair<Pair, std::int32_t> >{
        {Pair{1, 2}, -1},
        {Pair{1, 4}, 1},
        {Pair{2, 3}, -1},
        {Pair{4, 3}, 1},
    }));
}

TEST(Tokenizer_BPE_Word, MergeMaxLength)
{
    Word word = makeHelloWord();

    const std::vector<std::pair<Pair, std::int32_t>> changes =
            word.merge(2, 2, 4, 2);

    EXPECT_EQ(word.get_chars(), (std::vector<std::uint32_t>{0, 1, 4, 3}));
    EXPECT_EQ(changes, (std::vector<std::pair<Pair, std::int32_t> >{
        {Pair{1, 2}, -1},
        {Pair{2, 3}, -1},
    }));
}

}} // namespace
