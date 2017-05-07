#pragma once

#include <curand_kernel.h>

inline bool SameHemisphere(const Vector &w, const Vector &wp)
{
    return w.z * wp.z > 0.f;
}

__device__ void RejectionSampleDisk(float *x, float *y, curandState *state)
{
    float sx, sy;
    do
    {
        sx = 1.f - 2.f * curand_uniform(state);
        sy = 1.f - 2.f * curand_uniform(state);
    } while (sx*sx + sy*sy > 1.f);
    *x = sx;
    *y = sy;
}

__device__ void ConcentricSampleDisk(float u1, float u2, float *dx, float *dy)
{
    float r, theta;
    // Map uniform random numbers to $[-1,1]^2$
    float sx = 2 * u1 - 1;
    float sy = 2 * u2 - 1;

    // Map square to $(r,\theta)$

    // Handle degeneracy at the origin
    if (sx == 0.0 && sy == 0.0)
    {
        *dx = 0.0;
        *dy = 0.0;
        return;
    }
    if (sx >= -sy)
    {
        if (sx > sy)
        {
            // Handle first region of disk
            r = sx;
            if (sy > 0.0)
                theta = sy / r;
            else
                theta = 8.0f + sy / r;
        }
        else
        {
            // Handle second region of disk
            r = sy;
            theta = 2.0f - sx / r;
        }
    }
    else
    {
        if (sx <= sy)
        {
            // Handle third region of disk
            r = -sx;
            theta = 4.0f - sy / r;
        }
        else
        {
            // Handle fourth region of disk
            r = -sy;
            theta = 6.0f + sx / r;
        }
    }
    theta *= M_PI / 4.f;
    *dx = r * cosf(theta);
    *dy = r * sinf(theta);
}

__device__ inline Vector CosineSampleHemisphere(float u1, float u2)
{
    Vector ret;
    ConcentricSampleDisk(u1, u2, &ret.x, &ret.y);
    ret.z = sqrtf(fmax(0.f, 1.f - ret.x * ret.x - ret.y * ret.y));
    return ret;
}

__device__ inline Vector UniformSampleHemisphere(float u1, float u2)
{
    float z = u1;
    float r = sqrtf(fmax(0.f, 1.f - z*z));
    float phi = 2.0f * M_PI * u2;
    float x = r * cosf(phi);
    float y = r * sinf(phi);
    return Vector(x, y, z);
}

//__device__ void dConcentricSampleDisk(double u1, double u2, double *dx, double *dy)
//{
//    double r, theta;
//    // Map uniform random numbers to $[-1,1]^2$
//    double sx = 2 * u1 - 1;
//    double sy = 2 * u2 - 1;
//
//    // Map square to $(r,\theta)$
//
//    // Handle degeneracy at the origin
//    if (sx == 0.0 && sy == 0.0)
//    {
//        *dx = 0.0;
//        *dy = 0.0;
//        return;
//    }
//    if (sx >= -sy)
//    {
//        if (sx > sy)
//        {
//            // Handle first region of disk
//            r = sx;
//            if (sy > 0.0)
//                theta = sy / r;
//            else
//                theta = 8.0 + sy / r;
//        }
//        else
//        {
//            // Handle second region of disk
//            r = sy;
//            theta = 2.0 - sx / r;
//        }
//    }
//    else
//    {
//        if (sx <= sy)
//        {
//            // Handle third region of disk
//            r = -sx;
//            theta = 4.0 - sy / r;
//        }
//        else
//        {
//            // Handle fourth region of disk
//            r = -sy;
//            theta = 6.0 + sx / r;
//        }
//    }
//    theta *= 3.14159265358979323846 / 4.0;
//    *dx = r * cos(theta);
//    *dy = r * sin(theta);
//}
//
//__device__ inline Vector dCosineSampleHemisphere(double u1, double u2)
//{
//    double x, y, z;
//    dConcentricSampleDisk(u1, u2, &x, &y);
//    z = sqrt(fmax(0.0, 1.0 - x * x - y * y));
//    return{x, y, z};
//}

#include <random>
float frandom()
{
    static std::default_random_engine e;
    static std::uniform_real_distribution<> dist(0.0, 1.0); // rage 0 - 1
    return dist(e);
}
