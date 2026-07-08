// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "test_precomp.hpp"
#include "../src/tokenizer/models/bpe/bpe.hpp"

#include <cstdio>
#include <fstream>

namespace opencv_test { namespace {

TEST(Tokenizer_BPE, ReadFile)
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

}} // namespace
