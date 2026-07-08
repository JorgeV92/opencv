// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "../../../precomp.hpp"
#include "bpe.hpp"

#include "../../unicode.hpp"

namespace cv { namespace dnn {

BpeBuilder::BpeBuilder(Vocab vocab, Merges merges)
    : vocab_(std::move(vocab)), merges_(std::move(merges))
{
}

void BpeBuilder::vocabAndMerges(Vocab vocab, Merges merges)
{
    vocab_ = std::move(vocab);
    merges_ = std::move(merges);
}

void BpeBuilder::files(const std::string& vocab, const std::string& merges)
{
    files_ = std::make_pair(vocab, merges);
}

void BpeBuilder::setDropout(float dropout)
{
    dropout_ = dropout;
}

void BpeBuilder::setUnkToken(const std::string& un)
{
    unk_token_ = un;
}

void BpeBuilder::setConSubwordPrefix(const std::string& prefix)
{
    continuing_subword_prefix_ = prefix;
}

BPE BpeBuilder::build()
{
    if (dropout_.has_value())
    {
        const float value = dropout_.value();
        if (value > 1.0f || value < 0.0f)
            CV_Error(cv::Error::StsBadArg, "BPE dropout must be in [0, 1].");
    }

    if (files_.has_value())
    {
        auto vocabAndMerges = BPE::readFile(files_->first, files_->second);
        vocab_ = std::move(vocabAndMerges.first);
        merges_ = std::move(vocabAndMerges.second);
    }

    RVocab reverseVocab;
    reverseVocab.reserve(vocab_.size());
    for (const auto& tokenAndId : vocab_)
        reverseVocab[tokenAndId.second] = tokenAndId.first;

    const std::size_t prefixLength = continuing_subword_prefix_.has_value()
            ? continuing_subword_prefix_->size()
            : 0;

    MergeMap mergeMap;
    mergeMap.reserve(merges_.size());
    for (std::size_t i = 0; i < merges_.size(); ++i)
    {
        const std::string& first = merges_[i].first;
        const std::string& second = merges_[i].second;

        const auto firstIt = vocab_.find(first);
        if (firstIt == vocab_.end())
            CV_Error(cv::Error::StsBadArg, "BPE merge token out of vocabulary: " + first);

        const auto secondIt = vocab_.find(second);
        if (secondIt == vocab_.end())
            CV_Error(cv::Error::StsBadArg, "BPE merge token out of vocabulary: " + second);

        if (second.size() < prefixLength)
            CV_Error(cv::Error::StsBadArg,
                     "BPE merge token is shorter than continuing subword prefix: " + second);

        std::string newToken;
        newToken.reserve(first.size() + second.size() - prefixLength);
        newToken += first;
        newToken.append(second, prefixLength, std::string::npos);

        const auto newIt = vocab_.find(newToken);
        if (newIt == vocab_.end())
            CV_Error(cv::Error::StsBadArg, "BPE merged token out of vocabulary: " + newToken);

        mergeMap.emplace(Pair{firstIt->second, secondIt->second},
                         MergeRule{static_cast<std::uint32_t>(i), newIt->second});
    }

    return BPE(std::move(vocab_), std::move(reverseVocab), std::move(mergeMap), dropout_,
               unk_token_, continuing_subword_prefix_, end_of_word_suffix_, fuse_unk_,
               byte_fallback_);
}

BPE::BPE(Vocab vocab, Merges merges)
    : builder_(std::make_shared<BpeBuilder>())
{
    builder_->vocabAndMerges(std::move(vocab), std::move(merges));
}

BPE::BPE(Vocab vocab,
         RVocab reverseVocab,
         MergeMap mergeMap,
         std::optional<float> dropout,
         std::optional<std::string> unkToken,
         std::optional<std::string> continuingSubwordPrefix,
         std::optional<std::string> endOfWordSuffix,
         bool fuseUnk,
         bool byteFallback)
    : vocab_(std::move(vocab)),
      rev_vocab_(std::move(reverseVocab)),
      merge_map_(std::move(mergeMap)),
      dropout_(std::move(dropout)),
      unk_token_(std::move(unkToken)),
      continuing_subword_prefix_(std::move(continuingSubwordPrefix)),
      end_of_word_suffix_(std::move(endOfWordSuffix)),
      fuse_unk_(fuseUnk),
      byte_fallback_(byteFallback)
{
}

std::pair<Vocab, Merges> BPE::readFile(const std::string&, const std::string&)
{
    CV_Error(cv::Error::StsNotImplemented, "BPE vocabulary and merges file loading is not implemented.");
}

BpeBuilder BPE::fromFile(const std::string& vocab, const std::string& merges)
{
    if (!builder_)
        builder_ = std::make_shared<BpeBuilder>();
    builder_->files(vocab, merges);
    return *builder_;
}

Vocab BPE::getVocab() const
{
    return vocab_;
}

std::optional<std::string> BPE::getUnkToken() const
{
    return unk_token_;
}

std::optional<std::string> BPE::getConSubwordPrefix() const
{
    return continuing_subword_prefix_;
}

Word BPE::mergeWord(const std::string& wordValue)
{
    std::vector<std::size_t> indices;
    std::size_t offset = 0;
    while (offset < wordValue.size())
    {
        indices.push_back(offset);
        const std::size_t before = offset;
        unicode_cpt_from_utf8(wordValue, offset);
        if (offset <= before)
            ++offset;
    }

    Word word(static_cast<int>(wordValue.size()));

    for (std::size_t i = 0; i < indices.size(); ++i)
    {
        const std::size_t begin = indices[i];
        const std::size_t end = i + 1 < indices.size() ? indices[i + 1] : wordValue.size();
        std::string token = wordValue.substr(begin, end - begin);

        if (i > 0 && continuing_subword_prefix_.has_value())
            token = continuing_subword_prefix_.value() + token;

        if (i + 1 == indices.size() && end_of_word_suffix_.has_value())
            token += end_of_word_suffix_.value();

        // TODO: map token IDs and unknown-token behavior.
    }

    // TODO: apply merge_map_ after token IDs have been added.
    return word;
}

}} // namespace cv::dnn
