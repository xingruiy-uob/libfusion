#ifndef __INTRINSIC_MATRIX__
#define __INTRINSIC_MATRIX__

#include <memory>
#include <vector>
#include <iostream>

namespace fusion
{

struct IntrinsicMatrix;
struct IntrinsicMatrixPyramid;
typedef std::shared_ptr<IntrinsicMatrix> IntrinsicMatrixPtr;
typedef std::shared_ptr<IntrinsicMatrixPyramid> IntrinsicMatrixPyramidPtr;

struct IntrinsicMatrix
{
    IntrinsicMatrix() = default;
    IntrinsicMatrix(const IntrinsicMatrix &);
    IntrinsicMatrix(int cols, int rows, float fx, float fy, float cx, float cy);
    IntrinsicMatrix pyr_down() const;

    size_t width, height;
    float fx, fy, cx, cy, invfx, invfy;
};

struct IntrinsicMatrixPyramid
{
    IntrinsicMatrixPyramid(const IntrinsicMatrix &base_intrinsic_matrix, const int &max_level);
    IntrinsicMatrixPtr operator[](int level) const;
    int get_max_level() const;
    IntrinsicMatrix get_intrinsic_matrix_at(const int &level) const;

    std::vector<IntrinsicMatrixPtr> pyramid_;
};

std::ostream &operator<<(std::ostream &o, IntrinsicMatrix &K);

} // namespace fusion

#endif
