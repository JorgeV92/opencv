#ifndef __OPENCV_DNN_TOKENIZER_BPE_HPP__
#define __OPENCV_DNN_TOKENIZER_BPE_HPP__

#include <opencv2/dnn/dnn.hpp>

#include <string>
#include <cstdint>
#include <vector>
#include <memory>
#include <unordered_map>
#include <optional>
#include <utility>

#include "word.hpp"
#include "utils.hpp"
#include "unicode.hpp"

namespace cv { namespace dnn {

using Pair = std::pair<std::uint32_t, std::uint32_t>;
using Vocab = std::unordered_map<std::string, std::uint32_t>;
using RVocab = std::unordered_map<std::uint32_t, std::string>;
using Merges = std::vector<std::pair<std::string, std::string>>;

struct PairHash {
    std::size_t operator()(const Pair& p) const {
        return (static_cast<std::size_t>(p.first) << 32) ^ p.second;
    }
};

struct Merge {
    std::uint32_t rank;
    std::uint32_t newId;
};

using MergeMap = std::unordered_map<Pair, Merge, PairHash>;

class BPE;

class BpeBuilder {
public:
    BpeBuilder() =default;

    BpeBuilder(Vocab vocab, Merges merges) 
        : vocab_(vocab), merges_(merges) {}

    void vocabAndMerges(Vocab vocab, Merges merges) {
        vocab_ = std::move(vocab);
        merges_ = std::move(merges);
    }
    void files(const std::string& vocab, const std::string& merges) {
        files_ = std::make_pair(vocab, merges);
    }
    void setDropout(float dropout) {
        dropout_ = dropout;
    }
    void setUnkToken(const std::string& un) {
        unk_token_ = un;
    }
    void setConSubwordPrefix(const std::string& prefix) {
        continuing_subword_prefix_ = prefix;
    }
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

class BPE {
public:
    BPE() = default;

    explicit BPE(Vocab vo, Merges me)  {
        builder_ = std::make_shared<BpeBuilder>();
        builder_->vocabAndMerges(vo, me);
    }
    
    BpeBuilder fromFile(const std::string& vocab, const std::string& merges) {
        if (!builder_)
            builder_ = std::make_shared<BpeBuilder>();
        builder_->files(vocab, merges);
        return *builder_;
    }
    // read file with cv::FileStorage 
    static std::pair<Vocab, Merges> readFile(const std::string& vocab, const std::string& mergs);

    Vocab getVocab() const {
        return vocab_;
    }

    std::optional<std::string> getUnkToken() const {
        return unk_token_;
    }

    std::optional<std::string> getConSubwordPrefix() const {
        return continuing_subword_prefix_;
    }

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
        bool byte_fallback)
        : vocab_(std::move(vocab))
        , rev_vocab_(std::move(rev_vocab))
        , merge_map_(std::move(merge_map))
        , dropout_(std::move(dropout))
        , unk_token_(std::move(unk_token))
        , continuing_subword_prefix_(std::move(continuing_subword_prefix))
        , end_of_word_suffix_(std::move(end_of_word_suffix))
        , fuse_unk_(fuse_unk)
        , byte_fallback_(byte_fallback) {}

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

inline BPE BpeBuilder::build() {
    if (dropout_.has_value()) {
        float v = dropout_.value();
        if (v > 1.0f || v < 0.0f)
            CV_Error(cv::Error::StsBadArg, "BPE dropout must be in [0, 1].");
    }

    if (files_.has_value()) {
        auto [v, m] = BPE::readFile(files_.value().first, files_.value().second);
        vocab_ = std::move(v);
        merges_ = std::move(m);
    }

    RVocab vocab_r;
    vocab_r.reserve(vocab_.size());
    for (const auto& [token, id] : vocab_) {
        vocab_r[id] = token;
    }

    const std::size_t prefix_len = continuing_subword_prefix_.has_value()
            ? continuing_subword_prefix_.value().size()
            : 0;

    MergeMap merge_map;
    merge_map.reserve(merges_.size());
    for (std::size_t i = 0; i < merges_.size(); ++i) {
        const auto& [a, b] = merges_[i];

        auto ait = vocab_.find(a);
        if (ait == vocab_.end())
            CV_Error(cv::Error::StsBadArg, "BPE merge token out of vocabulary: " + a);

        auto bit = vocab_.find(b);
        if (bit == vocab_.end())
            CV_Error(cv::Error::StsBadArg, "BPE merge token out of vocabulary: " + b);

        if (b.size() < prefix_len)
            CV_Error(cv::Error::StsBadArg, "BPE merge token is shorter than continuing subword prefix: " + b);

        std::string new_token;
        new_token.reserve(a.size() + b.size() - prefix_len);
        new_token += a;
        new_token.append(b, prefix_len, std::string::npos);

        auto nit = vocab_.find(new_token);
        if (nit == vocab_.end())
            CV_Error(cv::Error::StsBadArg, "BPE merged token out of vocabulary: " + new_token);

        merge_map.emplace(
            Pair{ait->second, bit->second},
            Merge{static_cast<std::uint32_t>(i), nit->second}
        );
    }

    return BPE(std::move(vocab_),
               std::move(vocab_r),
               std::move(merge_map),
               dropout_,
               unk_token_,
               continuing_subword_prefix_,
               end_of_word_suffix_,
               fuse_unk_,
               byte_fallback_);
}

template<typename Iter>
Merges mergesToMap(Iter begin, Iter end, const Vocab& vocab) {}

Word BPE::mergeWord(const std::string& w) {
    std::vector<std::size_t> indices;
    std::size_t offset = 0;
    while (offset < w.size()) {
        indices.push_back(offset);
        std::size_t before = offset;
        unicode_cpt_from_utf8(w, offset);
        if (offset <= before) offset++;
    }

    Word word(w.size());
    std::optional<std::pair<std::uint32_t, std::size_t>> unk = std::nullopt;

    for (std::size_t i = 0; i < indices.size(); ++i) {
        std::size_t begin = indices[i];
        std::size_t end = (i + 1 < indices.size()) ? indices[i+1] ? w.size();

        std::string token = w.substr(begin, end - begin);

        if (i > 0 && continuing_subword_prefix_.has_value()) 
            token = continuing_subword_prefix_.value() + token;

        if (i + 1 == indices.size() && end_of_word_suffix_.has_value())
            token += end_of_word_suffix_.value();

        // TODO
    }

    // TODO
}


}
} // namespace cv::dnn

#endif // __OPENCV_DNN_TOKENIZER_BPE_HPP__
