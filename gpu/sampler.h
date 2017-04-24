#pragma once

inline bool SameHemisphere(const Vector &w, const Vector &wp)
{
    return w.z * wp.z > 0.f;
}

void ConcentricSampleDisk(float u1, float u2, float *dx, float *dy)
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

inline Vector CosineSampleHemisphere(float u1, float u2)
{
    Vector ret;
    ConcentricSampleDisk(u1, u2, &ret.x, &ret.y);
    ret.z = sqrtf(fmax(0.f, 1.f - ret.x * ret.x - ret.y * ret.y));
    return ret;
}

#include <random>
float frandom()
{
    static std::default_random_engine e;
    static std::uniform_real_distribution<> dist(0.0, 1.0); // rage 0 - 1
    return dist(e);
}
