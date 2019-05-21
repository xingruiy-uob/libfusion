#ifndef __REDUCE_SUM__
#define __REDUCE_SUM__

#include <cuda_runtime_api.h>

#define MAX_WARP_SIZE 32

template <typename T, int size>
__device__ __forceinline__ void WarpReduce(T *val)
{
#pragma unroll
    for (int offset = MAX_WARP_SIZE / 2; offset > 0; offset /= 2)
    {
#pragma unroll
        for (int i = 0; i < size; ++i)
        {
            val[i] += __shfl_down_sync(0xffffffff, val[i], offset);
        }
    }
}

template <typename T, int size>
__device__ __forceinline__ void BlockReduce(T *val)
{
    static __shared__ T shared[32 * size];
    int lane = threadIdx.x % MAX_WARP_SIZE;
    int wid = threadIdx.x / MAX_WARP_SIZE;

    WarpReduce<T, size>(val);

    if (lane == 0)
        memcpy(&shared[wid * size], val, sizeof(T) * size);

    __syncthreads();

    if (threadIdx.x < blockDim.x / MAX_WARP_SIZE)
        memcpy(val, &shared[lane * size], sizeof(T) * size);
    else
        memset(val, 0, sizeof(T) * size);

    if (wid == 0)
        WarpReduce<T, size>(val);
}

#endif