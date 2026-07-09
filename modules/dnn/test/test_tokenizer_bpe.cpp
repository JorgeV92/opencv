// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "test_precomp.hpp"
#include "../src/tokenizer/models/bpe/bpe.hpp"

#include <cstdio>
#include <fstream>

namespace opencv_test { namespace {

using cv::dnn::BPE;
using cv::dnn::BpeBuilder;
using cv::dnn::Merges;
using cv::dnn::Vocab;

TEST(Tokenizer_BPE_Model, ReadFile)
{
    const std::string vocabPath = cv::tempfile(".json");
    const std::string mergesPath = cv::tempfile(".txt");

    {
        std::ofstream vocabFile(vocabPath);
        ASSERT_TRUE(vocabFile.is_open());
        vocabFile << R"({"h": 0, "e": 1, "he": 2})";
    }
    {
        std::ofstream mergesFile(mergesPath);
        ASSERT_TRUE(mergesFile.is_open());
        mergesFile << "#version: 0.2\nh e\n";
    }

    const auto vocabAndMerges = cv::dnn::BPE::readFile(vocabPath, mergesPath);

    EXPECT_EQ(3u, vocabAndMerges.first.size());
    EXPECT_EQ(0u, vocabAndMerges.first.at("h"));
    EXPECT_EQ(1u, vocabAndMerges.first.at("e"));
    EXPECT_EQ(2u, vocabAndMerges.first.at("he"));
    ASSERT_EQ(1u, vocabAndMerges.second.size());
    EXPECT_EQ("h", vocabAndMerges.second[0].first);
    EXPECT_EQ("e", vocabAndMerges.second[0].second);

    EXPECT_EQ(0, std::remove(vocabPath.c_str()));
    EXPECT_EQ(0, std::remove(mergesPath.c_str()));
}

TEST(Tokenizer_BPE_Model, MergeWord)
{
    BpeBuilder builder(Vocab{{"a", 0}, {"b", 1}, {"ab", 2}}, Merges{{"a", "b"}});
    BPE model = builder.build();

    const std::vector<cv::dnn::Token> tokens = model.tokenize("ab");

    ASSERT_EQ(1u, tokens.size());
    EXPECT_EQ(2u, tokens[0].id);
    EXPECT_EQ("ab", tokens[0].value);
    EXPECT_EQ((std::pair<std::size_t, std::size_t>{0, 2}), tokens[0].offsets);
}

TEST(Tokenizer_BPE_Model, MergeWordPrefixAndSuffix)
{
    BpeBuilder builder(Vocab{{"a", 0}, {"##b</w>", 1}}, Merges{});
    builder.setConSubwordPrefix("##");
    builder.setEndOfWordSuffix("</w>");
    BPE model = builder.build();

    const std::vector<cv::dnn::Token> tokens = model.tokenize("ab");

    ASSERT_EQ(2u, tokens.size());
    EXPECT_EQ(0u, tokens[0].id);
    EXPECT_EQ("a", tokens[0].value);
    EXPECT_EQ((std::pair<std::size_t, std::size_t>{0, 1}), tokens[0].offsets);
    EXPECT_EQ(1u, tokens[1].id);
    EXPECT_EQ("##b</w>", tokens[1].value);
    EXPECT_EQ((std::pair<std::size_t, std::size_t>{1, 2}), tokens[1].offsets);
}

TEST(Tokenizer_BPE_Model, MergeWordFusedUnknowns)
{
    BpeBuilder builder(Vocab{{"a", 0}, {"[UNK]", 1}}, Merges{});
    builder.setUnkToken("[UNK]");
    builder.setFuseUnk(true);
    BPE model = builder.build();

    const std::vector<cv::dnn::Token> tokens = model.tokenize("xxa?");

    ASSERT_EQ(3u, tokens.size());
    EXPECT_EQ(1u, tokens[0].id);
    EXPECT_EQ("[UNK]", tokens[0].value);
    EXPECT_EQ((std::pair<std::size_t, std::size_t>{0, 2}), tokens[0].offsets);
    EXPECT_EQ(0u, tokens[1].id);
    EXPECT_EQ((std::pair<std::size_t, std::size_t>{2, 3}), tokens[1].offsets);
    EXPECT_EQ(1u, tokens[2].id);
    EXPECT_EQ((std::pair<std::size_t, std::size_t>{3, 4}), tokens[2].offsets);
}

TEST(Tokenizer_BPE_Model, MergeWordByteFallback)
{
    BpeBuilder builder(Vocab{{"<0xC3>", 0}, {"<0xA9>", 1}}, Merges{});
    builder.setByteFallback(true);
    BPE model = builder.build();

    const std::vector<cv::dnn::Token> tokens = model.tokenize("\xC3\xA9");

    ASSERT_EQ(2u, tokens.size());
    EXPECT_EQ(0u, tokens[0].id);
    EXPECT_EQ("<0xC3>", tokens[0].value);
    EXPECT_EQ((std::pair<std::size_t, std::size_t>{0, 1}), tokens[0].offsets);
    EXPECT_EQ(1u, tokens[1].id);
    EXPECT_EQ("<0xA9>", tokens[1].value);
    EXPECT_EQ((std::pair<std::size_t, std::size_t>{1, 2}), tokens[1].offsets);
}

TEST(Tokenizer_BPE_Model, MergeWordRejectsMissingUnknownToken)
{
    BpeBuilder builder(Vocab{}, Merges{});
    builder.setUnkToken("[UNK]");
    BPE model = builder.build();

    EXPECT_THROW(model.tokenize("?"), cv::Exception);
}

TEST(Tokenizer_BPE_Model, VocabularyLookup)
{
    BpeBuilder builder(Vocab{{"a", 7}}, Merges{});
    BPE model = builder.build();

    ASSERT_TRUE(model.tokenToId("a").has_value());
    EXPECT_EQ(7u, model.tokenToId("a").value());
    EXPECT_FALSE(model.tokenToId("missing").has_value());
    ASSERT_TRUE(model.idToToken(7).has_value());
    EXPECT_EQ("a", model.idToToken(7).value());
    EXPECT_FALSE(model.idToToken(9).has_value());
    EXPECT_EQ(1u, model.getVocabSize());
}

}} // namespace
