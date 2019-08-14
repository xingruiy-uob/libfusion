#include <cuda_runtime_api.h>
#include <opencv2/opencv.hpp>
#include "math/vector_type.h"

template <bool Debug>
struct ComputeResidualFunctor
{
    int NP, cols, rows;
    fusion::Vector4f *reference;
    fusion::Vector4f *source;
    fusion::Vector2f *corresp;
    float *sq_residual;

    __device__ inline bool find_corresp(const int &x, const int &y, fusion::Vector2f &pos, float &residual)
    {
    }

    __device__ inline compute_residual(const int &k)
    {
        const int y = k / cols;
        const int x = k - y * cols;

        fusion::Vector2f corr;
        float residual;
        if (find_corresp(x, y, corr, residual))
        {
            corresp[k] = corr;
            sq_residual[k] = residual * residual;
        }
    }

    __device__ inline operator()()
    {
        for (int k = threadIdx.x + blockDim.x * blockIdx.x; k < NP; k += gridDim.x * blockDim.x)
        {
            compute_residual(k);
        }
    }
};

void compute_residual()
{
    ComputeResidualFunctor functor;
}

void compute_derivatives(

)
{
}