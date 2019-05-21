#include "map_proc.h"
#include "vector_math.h"
#include "cuda_utils.h"
#include <thrust/device_vector.h>
#include <opencv2/cudaarithm.hpp>
#include <opencv2/cudev/common.hpp>

namespace slam
{
namespace map
{

__global__ void export_kernel(int3 *visible_block_pos, float3 *vertex, float3 *normal, uchar3 *colour)
{
}

void export_(MapStruct map_struct,
             cv::cuda::GpuMat vertex,
             cv::cuda::GpuMat normal,
             cv::cuda::GpuMat colour)
{
}

} // namespace map
} // namespace slam