#include "pose_estimator.h"
#include "vector_math.h"
#include "cuda_utils.h"
#include "reduce_sum.h"
#include <thrust/device_vector.h>

namespace fusion
{

// TODO : Robust RGB Estimation

class RGBSelection
{
public:
    __device__ inline bool find_corresp(
        const int &x,
        const int &y,
        float &curr_val,
        float &last_val,
        float &dx,
        float &dy,
        float4 &pt) const
    {
        if (x >= cols || y >= rows)
            return false;

        // reference point
        pt = last_vmap.ptr(y)[x];
        if (isnan(pt.x) || pt.w < 0)
            return false;

        // reference point in curr frame
        pt = T_last_curr(pt);

        // reference intensity
        last_val = last_intensity.ptr(y)[x];

        if (!isfinite(last_val))
            return false;

        auto u = fx * pt.x / pt.z + cx;
        auto v = fy * pt.y / pt.z + cy;
        if (u >= 1 && v >= 1 && u <= cols - 2 && v <= rows - 2)
        {
            curr_val = interpolate_bilinear(curr_intensity, u, v);
            dx = interpolate_bilinear(curr_intensity_dx, u, v);
            dy = interpolate_bilinear(curr_intensity_dy, u, v);

            // point selection criteria
            // TODO : Optimise this
            return (dx > 2 || dy > 2) &&
                   isfinite(curr_val) &&
                   isfinite(dx) && isfinite(dy);
        }

        return false;
    }

    __device__ float interpolate_bilinear(cv::cuda::PtrStep<float> image, float &x, float &y) const
    {
        int u = floor(x), v = floor(y);
        float coeff_x = x - u, coeff_y = y - v;
        return (image.ptr(v)[u] * (1 - coeff_x) + image.ptr(v)[u + 1] * coeff_x) * (1 - coeff_y) +
               (image.ptr(v + 1)[u] * (1 - coeff_x) + image.ptr(v + 1)[u + 1] * coeff_x) * coeff_y;
    }

    __device__ __inline__ void operator()() const
    {
        for (int k = blockIdx.x * blockDim.x + threadIdx.x; k < N; k += blockDim.x * gridDim.x)
        {
            const int y = k / cols;
            const int x = k - y * cols;

            if (y >= cols || x >= rows)
                return;

            float4 pt;
            float curr_val, last_val, dx, dy;
            bool corresp_found = find_corresp(x, y, curr_val, last_val, dx, dy, pt);

            if (corresp_found)
            {
                uint index = atomicAdd(num_corresp, 1);
                array_image[index] = make_float4(last_val, curr_val, dx, dy);
                array_point[index] = pt;
                error_term[index] = last_val - curr_val;
            }
        }
    }

public:
    cv::cuda::PtrStep<float4> last_vmap;
    cv::cuda::PtrStep<float> last_intensity;
    cv::cuda::PtrStep<float> curr_intensity;
    cv::cuda::PtrStep<float> curr_intensity_dx;
    cv::cuda::PtrStep<float> curr_intensity_dy;
    float fx, fy, cx, cy;
    DeviceMatrix3x4 T_last_curr;
    int N, cols, rows;

    int *num_corresp;
    float4 *array_image;
    float4 *array_point;

    float *error_term;
};

__global__ void compute_rgb_corresp_kernel(RGBSelection delegate)
{
    delegate();
}

__global__ void compute_variance_kernel(float *error_term, float *variance_term, float mean, uint max_idx)
{
    uint x = threadIdx.x + blockDim.x * blockIdx.x;
    if (x >= max_idx)
        return;

    variance_term[x] = pow(error_term[x] - mean, 2);
}

void compute_rgb_corresp(
    const cv::cuda::GpuMat last_vmap,
    const cv::cuda::GpuMat last_intensity,
    const cv::cuda::GpuMat curr_intensity,
    const cv::cuda::GpuMat curr_intensity_dx,
    const cv::cuda::GpuMat curr_intensity_dy,
    const Sophus::SE3d &frame_pose,
    const IntrinsicMatrix K,
    float4 *transformed_points,
    float4 *image_corresp_data,
    float *error_term_array,
    float *variance_term_array,
    float &mean_estimate,
    float &variance_estimate,
    uint &num_corresp)
{
    auto cols = last_vmap.cols;
    auto rows = last_vmap.rows;

    RGBSelection delegate;
    delegate.last_vmap = last_vmap;
    delegate.last_intensity = last_intensity;
    delegate.curr_intensity = curr_intensity;
    delegate.curr_intensity_dx = curr_intensity_dx;
    delegate.curr_intensity_dy = curr_intensity_dy;
    delegate.T_last_curr = frame_pose;
    delegate.array_image = image_corresp_data;
    delegate.array_point = transformed_points;
    delegate.error_term = error_term_array;
    delegate.fx = K.fx;
    delegate.fy = K.fy;
    delegate.cx = K.cx;
    delegate.cy = K.cy;
    delegate.cols = cols;
    delegate.rows = rows;
    delegate.N = cols * rows;

    safe_call(cudaMalloc(&delegate.num_corresp, sizeof(uint)));
    safe_call(cudaMemset(delegate.num_corresp, 0, sizeof(uint)));

    compute_rgb_corresp_kernel<<<96, 224>>>(delegate);

    safe_call(cudaMemcpy(&num_corresp, delegate.num_corresp, sizeof(uint), cudaMemcpyDeviceToHost));

    if (num_corresp <= 1)
        return;

    thrust::device_ptr<float> error_term(error_term_array);
    thrust::device_ptr<float> variance_term(variance_term_array);

    float sum_error = thrust::reduce(error_term, error_term + num_corresp);
    mean_estimate = 0; // sum_error / num_corresp;

    dim3 thread(MAX_THREAD);
    dim3 block(div_up(num_corresp, thread.x));

    compute_variance_kernel<<<block, thread>>>(error_term_array, variance_term_array, mean_estimate, num_corresp);
    float sum_variance = thrust::reduce(variance_term, variance_term + num_corresp);
    variance_estimate = sqrt(sum_variance / (num_corresp - 1));

    std::cout << "mean : " << mean_estimate << " variance : " << variance_estimate << " num_corresp : " << num_corresp << std::endl;

    safe_call(cudaFree(delegate.num_corresp));
}

struct RGBLeastSquares
{
    cv::cuda::PtrStep<float> out;

    float4 *transformed_points;
    float4 *image_corresp_data;
    float mean_estimated;
    float stdev_estimated;
    uint num_corresp;
    float fx, fy;
    size_t N;

    __device__ void compute_jacobian(const int &k, float *sum)
    {
        float row[7] = {0, 0, 0, 0, 0, 0, 0};

        if (k < num_corresp)
        {
            float3 p_transformed = make_float3(transformed_points[k]);
            float4 image = image_corresp_data[k];

            float z_inv = 1.0 / p_transformed.z;
            float3 left;
            left.x = image.z * fx * z_inv;
            left.y = image.w * fy * z_inv;
            left.z = -(left.x * p_transformed.x + left.y * p_transformed.y) * z_inv;

            float residual = image.x - image.y; // last_val - curr_val
            float res_normalized = (residual - mean_estimated) / stdev_estimated;
            float threshold_huber = 1.345 * stdev_estimated;
            float weight = 0;

            if (fabs(res_normalized) < threshold_huber)
                weight = 1;
            else
                weight = threshold_huber / fabs(res_normalized);

            if (weight < 10e-3 || res_normalized < 10e-4)
                weight = 1;

            row[6] = weight * res_normalized;
            // printf("%f, %f\n", res_normalized, threshold_huber);
            *(float3 *)&row[0] = weight * left;
            *(float3 *)&row[3] = weight * cross(p_transformed, left);
        }

        int count = 0;
#pragma unroll
        for (int i = 0; i < 7; ++i)
        {
#pragma unroll
            for (int j = i; j < 7; ++j)
                sum[count++] = row[i] * row[j];
        }
    }

    __device__ void operator()()
    {
        float sum[29] = {0, 0, 0, 0, 0,
                         0, 0, 0, 0, 0,
                         0, 0, 0, 0, 0,
                         0, 0, 0, 0, 0,
                         0, 0, 0, 0, 0,
                         0, 0, 0, 0};

        float val[29];
        for (int k = blockIdx.x * blockDim.x + threadIdx.x; k < N; k += blockDim.x * gridDim.x)
        {
            compute_jacobian(k, val);

#pragma unroll
            for (int i = 0; i < 29; ++i)
                sum[i] += val[i];
        }

        BlockReduce<float, 29>(sum);

        if (threadIdx.x == 0)
        {
#pragma unroll
            for (int i = 0; i < 29; ++i)
                out.ptr(blockIdx.x)[i] = sum[i];
        }
    }
};

__global__ void compute_least_square_RGB_kernel(RGBLeastSquares delegate)
{
    delegate();
}

void compute_least_square_RGB(
    const uint num_corresp,
    float4 *transformed_points,
    float4 *image_corresp_data,
    const float mean_estimated,
    const float stdev_estimated,
    const IntrinsicMatrix K,
    cv::cuda::GpuMat sum,
    cv::cuda::GpuMat out,
    float *hessian_estimated,
    float *residual_estimated,
    float *residual)
{
    RGBLeastSquares delegate;
    delegate.fx = K.fx;
    delegate.fy = K.fy;
    delegate.out = sum;
    delegate.N = num_corresp;
    delegate.num_corresp = num_corresp;
    delegate.image_corresp_data = image_corresp_data;
    delegate.transformed_points = transformed_points;
    delegate.mean_estimated = mean_estimated;
    delegate.stdev_estimated = stdev_estimated;

    compute_least_square_RGB_kernel<<<96, 224>>>(delegate);
    cv::cuda::reduce(sum, out, 0, cv::REDUCE_SUM);

    cv::Mat host_data;
    out.download(host_data);
    create_jtjjtr<6, 7>(host_data, hessian_estimated, residual_estimated);
    residual[0] = host_data.ptr<float>()[27];
    residual[1] = num_corresp;

    std::cout << residual[0] << " : " << residual[1] << std::endl;
}

}