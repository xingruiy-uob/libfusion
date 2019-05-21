#include "intrinsic_matrix.h"
#include <iostream>

namespace fusion
{

IntrinsicMatrix::IntrinsicMatrix(int cols, int rows, float fx, float fy, float cx, float cy)
    : width(cols), height(rows), fx(fx), fy(fy), cx(cx), cy(cy), invfx(1.0f / fx), invfy(1.0f / fy)
{
}

IntrinsicMatrix::IntrinsicMatrix(const IntrinsicMatrix &other)
    : width(other.width), height(other.height), fx(other.fx), fy(other.fy),
      cx(other.cx), cy(other.cy), invfx(other.invfx), invfy(other.invfy)
{
}

IntrinsicMatrix IntrinsicMatrix::pyr_down() const
{
    float s = 0.5f;
    return IntrinsicMatrix(s * width, s * height, s * fx, s * fy, s * cx, s * cy);
}

IntrinsicMatrixPyramid::IntrinsicMatrixPyramid(const IntrinsicMatrix &base_intrinsic_matrix, const int &max_level)
{
    pyramid_.resize(max_level);
    pyramid_[0] = std::make_shared<IntrinsicMatrix>(base_intrinsic_matrix);
    for (int i = 0; i < pyramid_.size() - 1; ++i)
    {
        pyramid_[i + 1] = std::make_shared<IntrinsicMatrix>(pyramid_[i]->pyr_down());
    }
}

IntrinsicMatrixPtr IntrinsicMatrixPyramid::operator[](int level) const
{
    return pyramid_[level];
}

int IntrinsicMatrixPyramid::get_max_level() const
{
    return pyramid_.size();
}

IntrinsicMatrix IntrinsicMatrixPyramid::get_intrinsic_matrix_at(const int &level) const
{
    return *pyramid_[level];
}

std::ostream &operator<<(std::ostream &o, IntrinsicMatrix &K)
{
    o << "fx : " << K.fx << " , "
      << "fy : " << K.fy << " , "
      << "cx : " << K.cx << " , "
      << "cy : " << K.cy << " , "
      << "invfx : " << K.invfx << " , "
      << "invfy : " << K.invfy << " , "
      << "cols : " << K.width << " , "
      << "rows : " << K.height;

    return o;
}

}