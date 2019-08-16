#include <opencv2/opencv.hpp>
#include <opencv2/cudawarping.hpp>
#include "tracking/cuda_imgproc.h"
#include "utils/safe_call.h"
#include "math/matrix_type.h"
#include "math/vector_type.h"
#include "data_struct/intrinsic_matrix.h"

namespace fusion
{

FUSION_HOST inline dim3 createGrid(dim3 block, int cols, int rows)
{
    return dim3(div_up(cols, block.x), div_up(rows, block.y));
}

FUSION_DEVICE inline Vector4c renderPoint(
    const Vector3f &point,
    const Vector3f &normal,
    const Vector3f &image,
    const Vector3f &light_pos)
{
    Vector3f colour(4.f / 255.f, 2.f / 255.f, 2.f / 255.f);
    if (!isnan(point.x))
    {
        const float Ka = 0.3f; //ambient coeff
        const float Kd = 0.5f; //diffuse coeff
        const float Ks = 0.2f; //specular coeff
        const float n = 20.f;  //specular power

        const float Ax = image.x; //ambient color,  can be RGB
        const float Dx = image.y; //diffuse color,  can be RGB
        const float Sx = image.z; //specular color, can be RGB
        const float Lx = 1.f;     //light color

        Vector3f L = normalised(light_pos - point);
        Vector3f V = normalised(Vector3f(0.f, 0.f, 0.f) - point);
        Vector3f R = normalised(2 * normal * (normal * L) - L);

        float Ix = Ax * Ka * Dx + Lx * Kd * Dx * fmax(0.f, (normal * L)) + Lx * Ks * Sx * pow(fmax(0.f, (R * V)), n);
        colour = Vector3f(Ix, Ix, Ix);
    }

    return Vector4c(
        static_cast<unsigned char>(__saturatef(colour.x) * 255.f),
        static_cast<unsigned char>(__saturatef(colour.y) * 255.f),
        static_cast<unsigned char>(__saturatef(colour.z) * 255.f),
        255);
}

FUSION_KERNEL void renderSceneK(
    const cv::cuda::PtrStep<Vector4f> vmap,
    const cv::cuda::PtrStep<Vector4f> nmap,
    const Vector3f light_pos,
    cv::cuda::PtrStepSz<Vector4c> dst)
{
    const int x = threadIdx.x + blockIdx.x * blockDim.x;
    const int y = threadIdx.y + blockIdx.y * blockDim.y;
    if (x >= dst.cols || y >= dst.rows)
        return;

    Vector3f point = ToVector3(vmap.ptr(y)[x]);
    Vector3f normal = ToVector3(nmap.ptr(y)[x]);
    Vector3f pixel(1.f);

    dst.ptr(y)[x] = renderPoint(point, normal, pixel, light_pos);
}

void renderScene(const cv::cuda::GpuMat vmap, const cv::cuda::GpuMat nmap, cv::cuda::GpuMat &image)
{
    dim3 block(8, 8);
    dim3 grid = createGrid(block, vmap.cols, vmap.rows);

    if (image.empty())
        image.create(vmap.rows, vmap.cols, CV_8UC4);

    renderSceneK<<<grid, block>>>(vmap, nmap, Vector3f(5, 5, 5), image);
}

FUSION_KERNEL void renderSceneTexturedK(
    const cv::cuda::PtrStep<Vector4f> vmap,
    const cv::cuda::PtrStep<Vector4f> nmap,
    const cv::cuda::PtrStep<Vector3c> image,
    const Vector3f light_pos,
    cv::cuda::PtrStepSz<Vector4c> dst)
{
    const int x = threadIdx.x + blockIdx.x * blockDim.x;
    const int y = threadIdx.y + blockIdx.y * blockDim.y;
    if (x >= dst.cols || y >= dst.rows)
        return;

    Vector3f point = ToVector3(vmap.ptr(y)[x]);
    Vector3f normal = ToVector3(nmap.ptr(y)[x]);
    Vector3f pixel = ToVector3f(image.ptr(y)[x]) / 255.f;

    dst.ptr(y)[x] = renderPoint(point, normal, pixel, light_pos);
}

void renderSceneTextured(const cv::cuda::GpuMat vmap, const cv::cuda::GpuMat nmap, const cv::cuda::GpuMat image, cv::cuda::GpuMat &out)
{
    dim3 block(8, 8);
    dim3 grid = createGrid(block, vmap.cols, vmap.rows);

    if (out.empty())
        out.create(vmap.rows, vmap.cols, CV_8UC4);

    renderSceneTexturedK<<<grid, block>>>(vmap, nmap, image, Vector3f(5, 5, 5), out);
}

FUSION_KERNEL void ToSemiDenseImageK(
    const cv::cuda::PtrStepSz<float> image,
    const cv::cuda::PtrStepSz<float> intensity_dx,
    const cv::cuda::PtrStepSz<float> intensity_dy,
    cv::cuda::PtrStepSz<float> semi,
    float th_dx, float th_dy)
{
    const int x = threadIdx.x + blockIdx.x * blockDim.x;
    const int y = threadIdx.y + blockIdx.y * blockDim.y;
    if (x >= image.cols || y >= image.rows)
        return;

    semi.ptr(y)[x] = 255;

    auto dx = intensity_dx.ptr(y)[x];
    auto dy = intensity_dy.ptr(y)[x];

    if (dx > th_dx || dy > th_dy)
    {
        semi.ptr(y)[x] = image.ptr(y)[x];
    }
}

void convert_image_to_semi_dense(const cv::cuda::GpuMat image, const cv::cuda::GpuMat dx, const cv::cuda::GpuMat dy, cv::cuda::GpuMat &semi, float th_dx, float th_dy)
{
    if (semi.empty())
        semi.create(image.size(), image.type());

    dim3 block(8, 4);
    dim3 grid = createGrid(block, image.cols, image.rows);

    ToSemiDenseImageK<<<grid, block>>>(image, dx, dy, semi, th_dx, th_dy);
}

void build_semi_dense_pyramid(const std::vector<cv::cuda::GpuMat> image_pyr, const std::vector<cv::cuda::GpuMat> dx_pyr, const std::vector<cv::cuda::GpuMat> dy_pyr, std::vector<cv::cuda::GpuMat> &semi_pyr, float th_dx, float th_dy)
{
    if (semi_pyr.size() != image_pyr.size())
        semi_pyr.resize(image_pyr.size());

    for (int level = 0; level < image_pyr.size(); ++level)
    {
        convert_image_to_semi_dense(image_pyr[level], dx_pyr[level], dy_pyr[level], semi_pyr[level], th_dx, th_dy);
    }
}

FUSION_DEVICE inline Vector3c interpolate_bilinear(const cv::cuda::PtrStepSz<Vector3c> image, float x, float y)
{
    int u = std::floor(x), v = std::floor(y);
    float coeff_x = x - (float)u, coeff_y = y - (float)v;
    Vector3f result = ToVector3f((image.ptr(v)[u] * (1 - coeff_x) + image.ptr(v)[u + 1] * coeff_x) * (1 - coeff_y) +
                                 (image.ptr(v + 1)[u] * (1 - coeff_x) + image.ptr(v + 1)[u + 1] * coeff_x) * coeff_y);
    return ToVector3c(result);
}

FUSION_KERNEL void warp_image_kernel(const cv::cuda::PtrStepSz<Vector3c> src,
                                     const cv::cuda::PtrStep<Vector4f> vmap_dst,
                                     const Matrix3x4f pose,
                                     const IntrinsicMatrix K,
                                     cv::cuda::PtrStep<Vector3c> dst)
{
    const int x = threadIdx.x + blockIdx.x * blockDim.x;
    const int y = threadIdx.y + blockIdx.y * blockDim.y;
    if (x >= src.cols || y >= src.rows)
        return;

    dst.ptr(y)[x] = Vector3c(0);
    Vector3f dst_pt_src = pose(ToVector3(vmap_dst.ptr(y)[x]));

    float u = K.fx * dst_pt_src.x / dst_pt_src.z + K.cx;
    float v = K.fy * dst_pt_src.y / dst_pt_src.z + K.cy;
    if (u >= 1 && v >= 1 && u < src.cols - 1 && v < src.rows - 1)
    {
        dst.ptr(y)[x] = interpolate_bilinear(src, u, v);
    }
}

void warp_image(const cv::cuda::GpuMat src, const cv::cuda::GpuMat vmap_dst, const Sophus::SE3d pose, const IntrinsicMatrix K, cv::cuda::GpuMat &dst)
{
    if (dst.empty())
        dst.create(src.size(), src.type());

    dim3 block(8, 4);
    dim3 grid = createGrid(block, src.cols, src.rows);

    warp_image_kernel<<<grid, block>>>(src, vmap_dst, pose.cast<float>().matrix3x4(), K, dst);
}

FUSION_HOST void filterDepthBilateral(const cv::cuda::GpuMat src, cv::cuda::GpuMat &dst)
{
    cv::cuda::bilateralFilter(src, dst, 5, 1, 1);
}

FUSION_HOST void pyrDownDepth(const cv::cuda::GpuMat src, cv::cuda::GpuMat &dst)
{
    cv::cuda::resize(src, dst, cv::Size(0, 0), 0.5, 0.5);
}

FUSION_HOST void pyrDownImage(const cv::cuda::GpuMat src, cv::cuda::GpuMat &dst)
{
    cv::cuda::pyrDown(src, dst);
}

FUSION_HOST void pyrDownVMap(const cv::cuda::GpuMat src, cv::cuda::GpuMat &dst)
{
    cv::cuda::resize(src, dst, cv::Size(0, 0), 0.5, 0.5);
}

FUSION_KERNEL void computeDerivativeK(
    cv::cuda::PtrStepSz<float> image,
    cv::cuda::PtrStep<float> dx,
    cv::cuda::PtrStep<float> dy)
{
    const int x = threadIdx.x + blockDim.x * blockIdx.x;
    const int y = threadIdx.y + blockDim.y * blockIdx.y;
    if (x >= image.cols - 1 || y >= image.rows - 1)
        return;

    int x10 = max(x - 1, 0);
    int x01 = min(x + 1, image.cols);
    int y10 = max(y - 1, 0);
    int y01 = min(y + 1, image.rows);

    dx.ptr(y)[x] = (image.ptr(y)[x01] - image.ptr(y)[x10]) * 0.5;
    dy.ptr(y)[x] = (image.ptr(y01)[x] - image.ptr(y10)[x]) * 0.5;
}

FUSION_HOST void computeDerivative(const cv::cuda::GpuMat image, cv::cuda::GpuMat &dx, cv::cuda::GpuMat &dy)
{
    if (dx.empty())
        dx.create(image.size(), image.type());
    if (dy.empty())
        dy.create(image.size(), image.type());

    dim3 block(8, 8);
    dim3 grid(div_up(image.cols, block.x), div_up(image.rows, block.y));

    computeDerivativeK<<<grid, block>>>(image, dx, dy);
}

FUSION_KERNEL void backProjectDepthK(const cv::cuda::PtrStepSz<float> depth, cv::cuda::PtrStep<Vector4f> vmap, IntrinsicMatrix intrinsics)
{
    const int x = threadIdx.x + blockDim.x * blockIdx.x;
    const int y = threadIdx.y + blockDim.y * blockIdx.y;
    if (x > depth.cols - 1 || y > depth.rows - 1)
        return;

    vmap.ptr(y)[x] = Vector4f(nanf("NAN"), nanf("NAN"), nanf("NAN"), -1.0f);
    float z = depth.ptr(y)[x];
    // z = (z == z) ? z : nanf("NAN");
    if (z > 0.3f && z < 5.0f)
    {
        vmap.ptr(y)[x] = Vector4f(z * (x - intrinsics.cx) * intrinsics.invfx, z * (y - intrinsics.cy) * intrinsics.invfy, z, 1.0f);
    }
}

FUSION_HOST void backProjectDepth(const cv::cuda::GpuMat depth, cv::cuda::GpuMat &vmap, const IntrinsicMatrix &K)
{
    if (vmap.empty())
        vmap.create(depth.size(), CV_32FC4);

    dim3 block(8, 8);
    dim3 grid = createGrid(block, depth.cols, depth.rows);

    backProjectDepthK<<<grid, block>>>(depth, vmap, K);
}

FUSION_KERNEL void computeNMapK(cv::cuda::PtrStepSz<Vector4f> vmap, cv::cuda::PtrStep<Vector4f> nmap)
{
    const int x = threadIdx.x + blockDim.x * blockIdx.x;
    const int y = threadIdx.y + blockDim.y * blockIdx.y;
    if (x >= vmap.cols - 1 || y >= vmap.rows - 1)
        return;

    int x10 = max(x - 1, 0);
    int x01 = min(x + 1, vmap.cols);
    int y10 = max(y - 1, 0);
    int y01 = min(y + 1, vmap.rows);

    Vector3f v00 = ToVector3(vmap.ptr(y)[x10]);
    Vector3f v01 = ToVector3(vmap.ptr(y)[x01]);
    Vector3f v10 = ToVector3(vmap.ptr(y10)[x]);
    Vector3f v11 = ToVector3(vmap.ptr(y01)[x]);

    nmap.ptr(y)[x] = Vector4f(normalised((v01 - v00).cross(v11 - v10)), 1.f);
}

FUSION_HOST void computeNMap(const cv::cuda::GpuMat vmap, cv::cuda::GpuMat &nmap)
{
    if (nmap.empty())
        nmap.create(vmap.size(), vmap.type());

    dim3 block(8, 8);
    dim3 grid = createGrid(block, vmap.cols, vmap.rows);

    computeNMapK<<<grid, block>>>(vmap, nmap);
}

__global__ void select_point_with_gradient_kernel(
    const cv::cuda::PtrStepSz<float> intensity,
    const cv::cuda::PtrStep<float> depth,
    const cv::cuda::PtrStep<float> dx,
    const cv::cuda::PtrStep<float> dy,
    cv::cuda::PtrStep<float> mask_out,
    Vector4f *selected_points)
{
    const int x = threadIdx.x + blockDim.x * blockIdx.x;
    const int y = threadIdx.y + blockDim.y * blockIdx.y;
    if (x >= intensity.cols || y >= intensity.rows)
        return;

    if (sqrt(pow(dx.ptr(y)[x], 2) + pow(dy.ptr(y)[x], 2)) > 2)
    {
        mask_out.ptr(y)[x] = intensity.ptr(y)[x];
    }
    else
    {
        mask_out.ptr(y)[x] = 0;
    }
}

void select_point(
    const cv::cuda::GpuMat intensity,
    const cv::cuda::GpuMat depth,
    const cv::cuda::GpuMat dx,
    const cv::cuda::GpuMat dy,
    Vector4f *selected_points)
{
    const auto cols = intensity.cols;
    const auto rows = intensity.rows;

    dim3 block(8, 8);
    dim3 grid(div_up(cols, block.x), div_up(rows, block.y));

    cv::cuda::GpuMat mask(intensity.size(), intensity.type());

    select_point_with_gradient_kernel<<<grid, block>>>(intensity, depth, dx, dy, mask, selected_points);

    cv::Mat img(mask);
    cv::imshow("img", img);
    cv::waitKey(1);
}

__global__ void check_covisibility_kernel(
    const cv::cuda::PtrStepSz<Vector4f> vmap,
    Matrix3x3f KRKinv, Vector3f Kt,
    uint *num_points)
{
    const int x = threadIdx.x + blockDim.x * blockIdx.x;
    const int y = threadIdx.y + blockDim.y * blockIdx.y;
    if (x >= vmap.cols || y >= vmap.rows)
        return;

    Vector3f vertex = ToVector3(vmap.ptr(y)[x]);
    Vector3f transformed_point = KRKinv(vertex) + Kt;
    if (transformed_point.x >= 0 &&
        transformed_point.y >= 0 &&
        transformed_point.x < vmap.cols &&
        transformed_point.y < vmap.rows)
    {
        atomicAdd(num_points, 1);
    }
}

float check_covisibility(
    const cv::cuda::GpuMat vmap,
    Eigen::Matrix3f R,
    Eigen::Vector3f t,
    IntrinsicMatrix &K)
{
    const int cols = vmap.cols;
    const int rows = vmap.rows;

    Eigen::Matrix3f Kmat;
    Kmat << K.fx, 0, K.cx,
        0, K.fy, K.cy,
        0, 0, 1;
    Eigen::Matrix3f KR = Kmat * R;
    Eigen::Vector3f Kt = Kmat * t;

    dim3 block(8, 8);
    dim3 grid(div_up(cols, block.x), div_up(rows, block.y));

    uint *num_points;

    cudaMalloc((void **)&num_points, sizeof(uint));
    cudaMemset(num_points, 0, sizeof(uint));

    check_covisibility_kernel<<<grid, block>>>(vmap, KR, Vector3f(Kt(0), Kt(1), Kt(2)), num_points);

    uint temp = 0;
    cudaMemcpy(&temp, num_points, sizeof(uint), cudaMemcpyDeviceToHost);

    cudaFree(num_points);
    return (float)temp / (cols * rows);
}

__device__ inline float interpolate(const cv::cuda::PtrStep<float> &map, float x, float y)
{
    float u = std::floor(x), v = std::floor(y);
    float coeff_x = x - u, coeff_y = y - v;

    float v00 = map.ptr((int)u)[(int)v];
    float v10 = map.ptr((int)u)[(int)v + 1];
    float v01 = map.ptr((int)u + 1)[(int)v];
    float v11 = map.ptr((int)u + 1)[(int)v + 1];

    return (v00 * (1 - coeff_x) * v10 * coeff_x) * (1 - coeff_y) + (v01 * (1 - coeff_x) + v11 * coeff_x) * coeff_y;
}

__global__ void compute_residual_kernel(
    const cv::cuda::PtrStep<float> ref_image,
    const cv::cuda::PtrStep<Vector4f> ref_vmap,
    const cv::cuda::PtrStep<float> src_image,
    Matrix3x3f KR_ref2src, Vector3f Kt_ref2src,
    cv::cuda::PtrStepSz<float> out_image)
{
    const int x = threadIdx.x + blockDim.x * blockIdx.x;
    const int y = threadIdx.y + blockDim.y * blockIdx.y;
    if (x >= out_image.cols || y >= out_image.rows)
        return;

    out_image.ptr(y)[x] = 255;

    Vector3f point = ToVector3(ref_vmap.ptr(y)[x]);
    Vector3f project = KR_ref2src(point) + Kt_ref2src;
    if (project.x >= 1 && project.y >= 1 &&
        project.x < out_image.cols - 1 &&
        project.y < out_image.rows - 1)
    {
        float val_interp = interpolate(src_image, project.x, project.y);
        out_image.ptr(y)[x] = abs(val_interp - ref_image.ptr(y)[x]);
    }
}

void compute_residual(
    const cv::cuda::GpuMat ref_image,
    const cv::cuda::GpuMat ref_vmap,
    const cv::cuda::GpuMat src_image,
    const IntrinsicMatrix &K,
    const Eigen::Matrix4d T_ref2src,
    cv::cuda::GpuMat &out_image)
{
    if (out_image.empty())
        out_image.create(ref_image.size(), ref_image.type());

    dim3 block(8, 8);
    dim3 grid(div_up(out_image.cols, block.x), div_up(out_image.rows, block.y));

    Eigen::Matrix3d Kmat;
    Kmat << K.fx, 0, K.cx,
        0, K.fy, K.cy,
        0, 0, 1;

    Eigen::Matrix3d KR = Kmat * T_ref2src.topLeftCorner(3, 3);
    Eigen::Vector3d Kt = Kmat * T_ref2src.topRightCorner(3, 1);

    compute_residual_kernel<<<grid, block>>>(ref_image, ref_vmap, src_image, KR, Kt, out_image);
}

} // namespace fusion
