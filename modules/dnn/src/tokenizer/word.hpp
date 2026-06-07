// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.	

#include <cstdint>
#include <cstddef>
#include <vector>

#include "bpe.hpp"

namespace cv { namespace dnn {

struct Symbol 
{
    std::uint32_t id;
    int prev;
    int next;
    std::size_t len;
};

// TODO 
class Word 
{
public:
    Word() =default;
    Word(int len) : symbols_(len) {}
    void add(std::uint32_t id, std::size_t bytelen);
    void mergeAll(const MergeMap& merges);
private:
    std::vector<Symbol> symbols_;
}

}
} // namespace cv::dnn