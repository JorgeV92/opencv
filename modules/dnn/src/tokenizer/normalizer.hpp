
#ifndef __OPENCV_DNN_TOKENIZER_NORMALIZER_HPP__
#define __OPENCV_DNN_TOKENIZER_NORMALIZER_HPP__

#include <string>
#include <string_view>
#include <vector>
#include <optional>
#include <functional>

namespace cv { namespace dnn {

struct Offsets {
    std::size_t begin{0};
    std::size_t end{0};
};

using Alignment = std::pair<std::size_t, std::size_t>;

enum class OffsetReferential {
    Original,
    Normalized,
};

enum class BoundKind {
    Unbounded,
    Included,
    Excluded,
};

struct RangeBound {
    BoundKind kind = BoundKind::Unbounded;
    std::size_t value = 0;

    static RangeBound unbounded();
    static RangeBound included(std::size_t value);
    static RangeBound excluded(std::size_t value);
};

class Range {
public:
    static Range original(RangeBound start, RangeBound end);
    static Range normalized(RangeBound start, RangeBound end);
    static Range original(std::size_t start, std::size_t end);
    static Range normalized(std::size_t start, std::size_t end);

    OffsetReferential referential() const;
    const RangeBound& start_bound() const;
    const RangeBound& end_bound() const;

    std::optional<std::size_t> len() const;
    Offsets into_full_range(std::size_t max_len) const;

private:
    Range(OffsetReferential referential, RangeBound start, RangeBound end);

    OffsetReferential referential_;
    RangeBound start_;
    RangeBound end_;
};

enum class SplitDelimiterBehavior {
    Removed,
    Isolated,
    MergedWithPrevious,
    MergedWithNext,
    Contiguous,
};

struct Match {
    Offsets offsets;
    bool is_match{false};
};

class Pattern {
public:
    virtual ~Pattern() = default;
    virtual std::vector<Match> find_matches(std::string_view normalized) const = 0;
};

struct Transform {
    char32_t value = U'\0';
    std::ptrdiff_t change = 0;
};

class NormalizedString {
public:
    NormalizedString() = default;
    explicit NormalizedString(std::string value);
    explicit NormalizedString(std::string_view value);

    static NormalizedString from(std::string value);
    static NormalizedString from(std::string_view value);

    const std::string& get() const;
    const std::string& get_original() const;
    Offsets offsets_original() const;

    std::optional<Offsets> convert_offsets(const Range& range) const;
    std::optional<std::string_view> get_range(const Range& range) const;
    std::optional<std::string_view> get_range_original(const Range& range) const;
    std::optional<NormalizedString> slice(const Range& range) const;

    NormalizedString& transform_range(const Range& range, 
                                      const std::vector<Transform>& dest,
                                      std::size_t initial_offset);
    NormalizedString& transform(const std::vector<Transform>& dest, std::size_t initial_offset);

    NormalizedString& nfd();
    NormalizedString& nfkd();
    NormalizedString& nfc();
    NormalizedString& nfkc();

    NormalizedString& filter(const std::function<bool(char32_t)>& keep);
    NormalizedString& prepend(std::string_view value);
    NormalizedString& append(std::string_view value);
    NormalizedString& map(const std::function<char32_t(char32_t)>& mapper);
    const NormalizedString& for_each(const std::function<void(char32_t)>& visitor) const;
    NormalizedString& lowercase();
    NormalizedString& uppercase();
    void replace(const Pattern& pattern, std::string_view content);
    std::size_t clear();

    std::vector<NormalizedString> split(const Pattern& pattern, 
                                    SplitDelimiterBehavior behavior) const;

    NormalizedString& lstrip();
    NormalizedString& rstrip();
    NormalizedString& strip();

    std::size_t len() const;
    std::size_t len_original() const;
    bool is_empty() const;

    const std::vector<Alignment>& alignments() const;
    std::size_t original_shift() const;
    std::vector<Alignment> alignments_original() const;

private:
    NormalizedString(std::string original,
                     std::string normalized, 
                     std::vector<Alignment> alignments,
                     std::size_t original_shift);
    
    std::optional<Range> validate_range(const Range& range) const;
    NormalizedString& lrstrip(bool left, bool right);

    std::string original_;
    std::string normalized_;
    std::vector<Alignment> alignments_;
    std::size_t original_shift_ = 0;
};

std::optional<Offsets> expand_alignments(const std::vector<Alignment>& alignments);
std::optional<std::string_view> get_range_of(std::string_view value, const Range& range);
std::optional<Offsets> bytes_to_char(std::string_view value, Offsets range);
std::optional<Offsets> char_to_bytes(std::string_view value, Offsets range);

class Normalizer {
public:
    virtual ~Normalizer() = default;
    virtual std::string normalize(std::string_view input) const = 0;
    virtual void normalize(NormalizedString&) const = 0;
};

}} // namespace cv::dnn

#endif //__OPENCV_DNN_TOKENIZER_NORMALIZER_HPP__

