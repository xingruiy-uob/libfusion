#ifndef __VECTOR_MATH__
#define __VECTOR_MATH__

#include <cmath>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <sophus/se3.hpp>

__host__ __device__ __forceinline__ uchar3 make_uchar3(int a)
{
    return make_uchar3(a, a, a);
}

__host__ __device__ __forceinline__ uchar4 make_uchar4(int a)
{
    return make_uchar4(a, a, a, a);
}

__host__ __device__ __forceinline__ uchar3 make_uchar3(float3 a)
{
    return make_uchar3((int)a.x, (int)a.y, (int)a.z);
}

__host__ __device__ __forceinline__ int3 make_int3(int a)
{
    return make_int3(a, a, a);
}

__host__ __device__ __forceinline__ int3 make_int3(float3 a)
{
    // return make_int3((int)a.x, (int)a.y, (int)a.z);
    int3 b = make_int3((int)a.x, (int)a.y, (int)a.z);
    b.x = b.x > a.x ? b.x - 1 : b.x;
    b.y = b.y > a.y ? b.y - 1 : b.y;
    b.z = b.z > a.z ? b.z - 1 : b.z;
    return b;
}

__host__ __device__ __forceinline__ float3 make_float3(uchar3 a)
{
    return make_float3(a.x, a.y, a.z);
}

__host__ __device__ __forceinline__ float3 make_float3(float a)
{
    return make_float3(a, a, a);
}

__host__ __device__ __forceinline__ float3 make_float3(float4 a)
{
    return make_float3(a.x, a.y, a.z);
}

__host__ __device__ __forceinline__ float4 make_float4(float a)
{
    return make_float4(a, a, a, a);
}

__host__ __device__ __forceinline__ float4 make_float4(float3 a, float b)
{
    return make_float4(a.x, a.y, a.z, b);
}

__host__ __device__ __forceinline__ float3 operator+(int3 a, float3 b)
{
    return make_float3(a.x + b.x, a.y + b.y, a.z + b.z);
}


__host__ __device__ __forceinline__ int3 operator+(int3 a, int3 b)
{
    return make_int3(a.x + b.x, a.y + b.y, a.z + b.z);
}

__host__ __device__ __forceinline__ float3 operator+(float3 a, float3 b)
{
    return make_float3(a.x + b.x, a.y + b.y, a.z + b.z);
}

__host__ __device__ __forceinline__ float4 operator+(float4 a, float4 b)
{
    return make_float4(a.x + b.x, a.y + b.y, a.z + b.z, a.w + b.w);
}

__host__ __device__ __forceinline__ void operator+=(float3 &a, float3 b)
{
    a.x += b.x;
    a.y += b.y;
    a.z += b.z;
}

__host__ __device__ __forceinline__ float3 operator-(float3 b)
{
    return make_float3(-b.x, -b.y, -b.z);
}

// __host__ __device__ __forceinline__ float4 operator-(float4 b)
// {
//     return make_float4(-b.x, -b.y, -b.z, -b.w);
// }

__host__ __device__ __forceinline__ float3 operator-(float3 a, float b)
{
    return make_float3(a.x - b, a.y - b, a.z - b);
}

// __host__ __device__ __forceinline__ float3 operator-(float a, float3 b)
// {
//     return make_float3(a - b.x, a - b.y, a - b.z);
// }

__host__ __device__ __forceinline__ int3 operator-(int3 a, int3 b)
{
    return make_int3(a.x - b.x, a.y - b.y, a.z - b.z);
}

__host__ __device__ __forceinline__ float3 operator-(float3 a, float3 b)
{
    return make_float3(a.x - b.x, a.y - b.y, a.z - b.z);
}

// __host__ __device__ __forceinline__ float4 operator-(float4 a, float3 b)
// {
//     return make_float4(a.x - b.x, a.y - b.y, a.z - b.z, a.w);
// }

__host__ __device__ __forceinline__ float4 operator-(float4 a, float4 b)
{
    return make_float4(a.x - b.x, a.y - b.y, a.z - b.z, a.w - b.w);
}

__host__ __device__ __forceinline__ float operator*(float3 a, float3 b)
{
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

// __host__ __device__ __forceinline__ float operator*(float3 a, float4 b)
// {
//     return a.x * b.x + a.y * b.y + a.z * b.z;
// }

// __host__ __device__ __forceinline__ float operator*(float4 a, float3 b)
// {
//     return a.x * b.x + a.y * b.y + a.z * b.z + a.w;
// }

__host__ __device__ __forceinline__ float operator*(float4 a, float4 b)
{
    return a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w;
}

// __host__ __device__ __forceinline__ uchar3 operator*(uchar3 a, unsigned short b)
// {
//     return make_uchar3(a.x * b, a.y * b, a.z * b);
// }

// __host__ __device__ __forceinline__ uchar3 operator*(uchar3 a, int b)
// {
//     return make_uchar3(a.x * b, a.y * b, a.z * b);
// }

__host__ __device__ __forceinline__ float3 operator*(uchar3 a, float b)
{
    return make_float3(a.x * b, a.y * b, a.z * b);
}

// __host__ __device__ __forceinline__ uchar3 operator*(int b, uchar3 a)
// {
//     return make_uchar3(a.x * b, a.y * b, a.z * b);
// }

__host__ __device__ __forceinline__ float3 operator*(float b, uchar3 a)
{
    return make_float3(a.x * b, a.y * b, a.z * b);
}

// __host__ __device__ __forceinline__ int3 operator*(int3 a, unsigned int b)
// {
//     return make_int3(a.x * b, a.y * b, a.z * b);
// }

__host__ __device__ __forceinline__ int3 operator*(int3 a, int b)
{
    return make_int3(a.x * b, a.y * b, a.z * b);
}

__host__ __device__ __forceinline__ int3 operator*(float3 a, int b)
{
    return make_int3(a.x * b, a.y * b, a.z * b);
}

__host__ __device__ __forceinline__ float3 operator*(int3 a, float b)
{
    return make_float3(a.x * b, a.y * b, a.z * b);
}

__host__ __device__ __forceinline__ float3 operator*(float3 a, float b)
{
    return make_float3(a.x * b, a.y * b, a.z * b);
}

__host__ __device__ __forceinline__ float3 operator*(float a, float3 b)
{
    return make_float3(a * b.x, a * b.y, a * b.z);
}

__host__ __device__ __forceinline__ float4 operator*(float4 a, float b)
{
    return make_float4(a.x * b, a.y * b, a.z * b, a.w * b);
}

__host__ __device__ __forceinline__ int3 operator/(int3 a, int3 b)
{
    return make_int3(a.x / b.x, a.y / b.y, a.z / b.z);
}

// __host__ __device__ __forceinline__ float3 operator/(float3 a, int3 b)
// {
//     return make_float3(a.x / b.x, a.y / b.y, a.z / b.z);
// }

__host__ __device__ __forceinline__ float3 operator/(float3 a, float3 b)
{
    return make_float3(a.x / b.x, a.y / b.y, a.z / b.z);
}

__host__ __device__ __forceinline__ float4 operator/(float4 a, float4 b)
{
    return make_float4(a.x / b.x, a.y / b.y, a.z / b.z, a.w / b.w);
}

// __host__ __device__ __forceinline__ int2 operator/(int2 a, int b)
// {
//     return make_int2(a.x / b, a.y / b);
// }

__host__ __device__ __forceinline__ float2 operator/(float2 a, int b)
{
    return make_float2(a.x / b, a.y / b);
}

// __host__ __device__ __forceinline__ uchar3 operator/(uchar3 a, int b)
// {
//     return make_uchar3(a.x / b, a.y / b, a.z / b);
// }

// __host__ __device__ __forceinline__ int3 operator/(int3 a, unsigned int b)
// {
//     return make_int3(a.x / (int)b, a.y / (int)b, a.z / (int)b);
// }

__host__ __device__ __forceinline__ int3 operator/(int3 a, int b)
{
    return make_int3(a.x / b, a.y / b, a.z / b);
}

__host__ __device__ __forceinline__ float3 operator/(float3 a, int b)
{
    return make_float3(a.x / b, a.y / b, a.z / b);
}

__host__ __device__ __forceinline__ float3 operator/(float3 a, float b)
{
    return make_float3(a.x / b, a.y / b, a.z / b);
}

// __host__ __device__ __forceinline__ float3 operator/(float a, float3 b)
// {
//     return make_float3(a / b.x, a / b.y, a / b.z);
// }

__host__ __device__ __forceinline__ float4 operator/(float4 a, float b)
{
    return make_float4(a.x / b, a.y / b, a.z / b, a.w / b);
}

__host__ __device__ __forceinline__ int3 operator%(int3 a, int b)
{
    return make_int3(a.x % b, a.y % b, a.z % b);
}

// __host__ __device__ __forceinline__ void operator%=(int3 &a, int b)
// {
//     a.x %= b;
//     a.y %= b;
//     a.z %= b;
// }

__host__ __device__ __forceinline__ bool operator==(int3 a, int3 b)
{
    return a.x == b.x && a.y == b.y && a.z == b.z;
}

// __device__ __forceinline__ bool operator==(float3 a, float3 b)
// {
//     return a.x == b.x && a.y == b.y && a.z == b.z;
// }

// __device__ __forceinline__ bool operator==(float4 a, float4 b)
// {
//     return a.x == b.x && a.y == b.y && a.z == b.z && a.w == b.w;
// }

// __device__ __forceinline__ bool operator==(float2 a, float2 b)
// {
//     return a.x == b.x && a.y == b.y;
// }

__host__ __device__ __forceinline__ float3 cross(float3 a, float3 b)
{
    return make_float3(a.y * b.z - a.z * b.y,
                       a.z * b.x - a.x * b.z,
                       a.x * b.y - a.y * b.x);
}

__host__ __device__ __forceinline__ float3 cross(float4 a, float4 b)
{
    return make_float3(a.y * b.z - a.z * b.y,
                       a.z * b.x - a.x * b.z,
                       a.x * b.y - a.y * b.x);
}

__host__ __device__ __forceinline__ float norm(float3 a)
{
    return sqrt(a * a);
}

__host__ __device__ __forceinline__ float norm(float4 a)
{
    return sqrt(a * a);
}

__host__ __device__ __forceinline__ float inv_norm(float3 a)
{
    return 1.0 / sqrt(a * a);
}

__host__ __device__ __forceinline__ float inv_norm(float4 a)
{
    return 1.0 / sqrt(a * a);
}

__host__ __device__ __forceinline__ float3 normalised(float3 a)
{
    return a / norm(a);
}

__host__ __device__ __forceinline__ float4 normalised(float4 a)
{
    return a / norm(a);
}

__host__ __device__ __forceinline__ float3 floor(float3 a)
{
    return make_float3(floor(a.x), floor(a.y), floor(a.z));
}

__host__ __device__ __forceinline__ float3 fmaxf(float3 a, float3 b)
{
    return make_float3(fmaxf(a.x, b.x), fmaxf(a.y, b.y), fmaxf(a.z, b.z));
}

__host__ __device__ __forceinline__ float3 fminf(float3 a, float3 b)
{
    return make_float3(fminf(a.x, b.x), fminf(a.y, b.y), fminf(a.z, b.z));
}

__host__ __forceinline__ float3 make_float3(const Sophus::SE3d &pose)
{
    auto t = pose.translation();
    return make_float3(t(0), t(1), t(2));
}

class DeviceMatrix3x4
{
public:
    DeviceMatrix3x4() = default;
    DeviceMatrix3x4(const Sophus::SE3d &pose)
    {
        Eigen::Matrix<float, 4, 4> mat = pose.cast<float>().matrix();
        row_0 = make_float4(mat(0, 0), mat(0, 1), mat(0, 2), mat(0, 3));
        row_1 = make_float4(mat(1, 0), mat(1, 1), mat(1, 2), mat(1, 3));
        row_2 = make_float4(mat(2, 0), mat(2, 1), mat(2, 2), mat(2, 3));
    }

    __host__ __device__ float3 rotate(const float3 &pt) const
    {
        float3 result;
        result.x = row_0.x * pt.x + row_0.y * pt.y + row_0.z * pt.z;
        result.y = row_1.x * pt.x + row_1.y * pt.y + row_1.z * pt.z;
        result.z = row_2.x * pt.x + row_2.y * pt.y + row_2.z * pt.z;
        return result;
    }

    __host__ __device__ float3 operator()(const float3 &pt) const
    {
        float3 result;
        result.x = row_0.x * pt.x + row_0.y * pt.y + row_0.z * pt.z + row_0.w;
        result.y = row_1.x * pt.x + row_1.y * pt.y + row_1.z * pt.z + row_1.w;
        result.z = row_2.x * pt.x + row_2.y * pt.y + row_2.z * pt.z + row_2.w;
        return result;
    }

    __host__ __device__ float4 operator()(const float4 &pt) const
    {
        float4 result;
        result.x = row_0.x * pt.x + row_0.y * pt.y + row_0.z * pt.z + row_0.w;
        result.y = row_1.x * pt.x + row_1.y * pt.y + row_1.z * pt.z + row_1.w;
        result.z = row_2.x * pt.x + row_2.y * pt.y + row_2.z * pt.z + row_2.w;
        result.w = 1.0f;
        return result;
    }

    float4 row_0, row_1, row_2;
};

#endif
