#pragma once

//#include <algorithm>
//#include <cmath>
//#include <initializer_list>

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#ifndef M_PI
#define M_PI 3.141592653f
#endif

/* Linear Algebra */
template <typename T>
struct Vec3
{
    T x, y, z;

    __host__ __device__ inline bool isZero() const
    {
        return x == T(0) && y == T(0) && z == T(0);
    }

    __host__ __device__ inline Vec3<T> &zero()
    {
        x = y = z = T(0);
        return *this;
    }
    __host__ __device__ inline Vec3<T> &norm()
    {
        if (!isZero())
        {
            T mod = rsqrt(x * x + y * y + z * z);
            //T mod = T(1) / sqrt(x * x + y * y + z * z);
            x *= mod;
            y *= mod;
            z *= mod;
        }
        return *this;
    }
    __host__ __device__ inline Vec3<T> &assign(T _x, T _y, T _z)
    {
        x = _x;
        y = _y;
        z = _z;
        return *this;
    }
    __host__ __device__ inline Vec3<T> &assign(const Vec3<T> &v)
    {
        x = v.x;
        y = v.y;
        z = v.z;
        return *this;
    }
    __host__ __device__ inline Vec3<T> &add(const Vec3<T> &v)
    {
        x += v.x;
        y += v.y;
        z += v.z;
        return *this;
    }
    __host__ __device__ inline Vec3<T> &sub(const Vec3<T> &v)
    {
        x -= v.x;
        y -= v.y;
        z -= v.z;
        return *this;
    }
    __host__ __device__ inline Vec3<T> &mul(const Vec3<T> &v)
    {
        x *= v.x;
        y *= v.y;
        z *= v.z;
        return *this;
    }
    __host__ __device__ inline Vec3<T> &scale(T factor)
    {
        x *= factor;
        y *= factor;
        z *= factor;
        return *this;
    }
    __host__ __device__ inline T dot(const Vec3<T> &v) const
    {
        return x * v.x + y * v.y + z * v.z;
    }

    __host__ __device__ inline static Vec3<T> Zero()
    {
        Vec3<T> result = {T(0), T(0), T(0)};
        return result;
    }
    __host__ __device__ inline static Vec3<T> Add(const Vec3<T> &v1, const Vec3<T> &v2)
    {
        Vec3<T> result = {v1.x + v2.x, v1.y + v2.y, v1.z + v2.z};
        return result;
    }
    __host__ __device__ inline static Vec3<T> Sub(const Vec3<T> &v1, const Vec3<T> &v2)
    {
        Vec3<T> result = {v1.x - v2.x, v1.y - v2.y, v1.z - v2.z};
        return result;
    }
    __host__ __device__ inline static Vec3<T> Mul(const Vec3<T> &v1, const Vec3<T> &v2)
    {
        Vec3<T> result = {v1.x * v2.x, v1.y * v2.y, v1.z * v2.z};
        return result;
    }
    __host__ __device__ inline static Vec3<T> Scale(const Vec3<T> &v, T factor)
    {
        Vec3<T> result = {v.x * factor, v.y * factor, v.z * factor};
        return result;
    }
    __host__ __device__ inline static Vec3<T> Norm(const Vec3<T> &v)
    {
        Vec3<T> result = v;
        result.norm();
        return result;
    }
    __host__ __device__ inline static T Dot(const Vec3<T> &v1, const Vec3<T> &v2)
    {
        return v1.x * v2.x + v1.y * v2.y + v1.z * v2.z;
    }
    __host__ __device__ inline static Vec3<T> Cross(const Vec3<T> &v1, const Vec3<T> &v2)
    {
        Vec3<T> result = {v1.y * v2.z - v1.z * v2.y, v1.z * v2.x - v1.x * v2.z,
                          v1.x * v2.y - v1.y * v2.x};
        return result;
    }

    __host__ __device__ inline Vec3<T> operator+(const Vec3<T> &v) const
    {
        return Add(*this, v);
    }
    __host__ __device__ inline Vec3<T> operator-(const Vec3<T> &v) const
    {
        return Sub(*this, v);
    }
    __host__ __device__ inline Vec3<T> operator-(void) const
    {
        return{-x, -y, -z};
    }
    __host__ __device__ inline Vec3<T> &operator+=(const Vec3<T> &v)
    {
        return add(v);
    }
    __host__ __device__ inline Vec3<T> operator-=(const Vec3<T> &v)
    {
        return sub(v);
    }

    // Debug
    __host__ __device__ Vec3<T>(T _x = 0, T _y = 0, T _z = 0) : x(_x), y(_y), z(_z)
    {}
};

/* Geometry */
template <typename T>
T DistanceSquared(const Vec3<T> &l, const Vec3<T> &r)
{
    return (l.x - r.x) * (l.x - r.x) + (l.y - r.y) * (l.y - r.y) +
        (l.z - r.z) * (l.z - r.z);
}

double clamp(double d)
{
    return d > 1.0 ? 1.0 : (d < 0.0 ? 0.0 : d);
}

__device__ __host__ float clamp(float d, float min, float max)
{
	return d > max ? max : (d < min ? min : d);
}


