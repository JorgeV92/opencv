// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

/**
 * Initialization:
    load tokenizer files:
        vocab.json
        merges.txt
        special_tokens_map.json maybe
        tokenizer_config.json maybe
        tokenizer.json maybe

    Encoding:
        raw text
         normalizer
         pre-tokenizer
         BPE / WordPiece / Unigram model using loaded vocab/merges
         post-processor
         token IDs

    Decoding:
        token IDs
         reverse vocab lookup
         decoder
         output text 
 * 
 */

#ifndef __OPENCV_DNN_TOKENIZER_MODELS_BPE_BPE_HPP__
#define __OPENCV_DNN_TOKENIZER_MODELS_BPE_BPE_HPP__

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "word.hpp"

namespace cv { namespace dnn {

using Vocab = std::unordered_map<std::string, std::uint32_t>;
using RVocab = std::unordered_map<std::uint32_t, std::string>;
using Merges = std::vector<std::pair<std::string, std::string>>;

class BPE;

class CV_EXPORTS BpeBuilder
{
public:
    BpeBuilder() = default;
    BpeBuilder(Vocab vocab, Merges merges);

    void vocabAndMerges(Vocab vocab, Merges merges);
    void files(const std::string& vocab, const std::string& merges);
    void setDropout(float dropout);
    void setUnkToken(const std::string& un);
    void setConSubwordPrefix(const std::string& prefix);
    BPE build();

private:
    std::optional<std::pair<std::string, std::string>> files_;
    Vocab vocab_;
    Merges merges_;
    std::optional<float> dropout_;
    std::optional<std::string> unk_token_;
    std::optional<std::string> continuing_subword_prefix_;
    std::optional<std::string> end_of_word_suffix_;
    bool fuse_unk_ = false;
    bool byte_fallback_ = false;
};

class CV_EXPORTS BPE
{
public:
    BPE() = default;

    explicit BPE(Vocab vo, Merges me);
    
    BpeBuilder fromFile(const std::string& vocab, const std::string& merges);
    // read file with cv::FileStorage 
    static std::pair<Vocab, Merges> readFile(const std::string& vocab, const std::string& mergs);

    Vocab getVocab() const;
    std::optional<std::string> getUnkToken() const;
    std::optional<std::string> getConSubwordPrefix() const;

    Word mergeWord(const std::string& w);

private:
    friend class BpeBuilder;

    BPE(Vocab vocab,
        RVocab rev_vocab,
        MergeMap merge_map,
        std::optional<float> dropout,
        std::optional<std::string> unk_token,
        std::optional<std::string> continuing_subword_prefix,
        std::optional<std::string> end_of_word_suffix,
        bool fuse_unk,
        bool byte_fallback);

    Vocab vocab_;
    RVocab rev_vocab_;
    MergeMap merge_map_;
    std::optional<float> dropout_;
    std::optional<std::string> unk_token_;
    std::optional<std::string> continuing_subword_prefix_;
    std::optional<std::string> end_of_word_suffix_;
    bool fuse_unk_ = false;
    bool byte_fallback_ = false;

    std::shared_ptr<BpeBuilder> builder_;
};

template<typename Iter>
Merges mergesToMap(Iter begin, Iter end, const Vocab& vocab) 
{

}

}} // namespace cv::dnn

#endif // __OPENCV_DNN_TOKENIZER_MODELS_BPE_BPE_HPP__
