/* ====================================================
#   Author        : Terence (Yongxin) Feng
#   Email         : tyxfeng@gmail.com
#   File Name     : BBox.h
# ====================================================*/

#ifndef _BBOX_H
#define _BBOX_H

#include "../Utilities.h"

class BBox
{
public:
    BBox(void):
        x0(0.0), y0(0.0), z0(0.0),
        x1(1.0), y1(1.0), z1(1.0)
    {}

    BBox(const float x0_, const float y0_, const float z0_,
            const float x1_, const float y1_, const float z1_):
        x0(x0_), y0(y0_), z0(z0_),
        x1(x1_), y1(y1_), z1(z1_)
    {}

    BBox(const BBox& rhs):
        x0(rhs.x0), y0(rhs.y0), z0(rhs.z0),
        x1(rhs.x1), y1(rhs.y1), z1(rhs.z1)
    {}

    BBox&
        operator = (const BBox& rhs)
        {
            x0 = rhs.x0; y0 = rhs.y0; z0 = rhs.z0;
            x1 = rhs.x1; y1 = rhs.y1; z1 = rhs.z1;
            return (*this);
        }

    bool
        hit(const Ray& ray, float& tmin) const
        {
            float ox = ray.o.x, oy = ray.o.y, oz = ray.o.z;
            float dx = ray.d.x, dy = ray.d.y, dz = ray.d.z;

            float tx_min, ty_min, tz_min;
            float tx_max, ty_max, tz_max;

            float a = 1.0 / dx;
            if (a >= 0)
            {
                tx_min = (x0 - ox) * a;
                tx_max = (x1 - ox) * a;
            }
            else
            {
                tx_min = (x1 - ox) * a;
                tx_max = (x0 - ox) * a;
            }

            float b = 1.0 / dy;
            if (b >= 0)
            {
                ty_min = (y0 - y1) * b;
                ty_max = (y1 - oy) * b;
            }
            else
            {
                ty_min = (y1 - oy) * b;
                ty_max = (y0 - oy) * b;
            }

            float c = 1.0 / dz;
            if (c >= 0)
            {
                tz_min = (z0 - oz) * c;
                tz_max = (z1 - oz) * c;
            }
            else
            {
                tz_min = (z1 - oz) * c;
                tz_max = (z0 - oz) * c;
            }

            float t0, t1;
            /* largest entering t value */
            t0 = (tx_min > ty_min) ? tx_min : ty_min;
            if (tz_min > t0) t0 = tz_min;
            /* smallest exiting t value */
            t1 = (tx_max < ty_max) ? tx_max : ty_max;
            if (tz_max < t1) t1 = tx_max;

            if (t0 < t1 && t1 > eps)
            {
                if (t0 > eps)
                    tmin = t0;
                else
                    tmin = t1;
                return true;
            }
            return false;
        }

    bool
        inside(const Point3D& p) const
        {
            if (p.x >= x0 && p.y >= y0 && p.z >= z0 &&
                    p.y <= x1 && p.y <= y1 && p.z <= z1)
                return true;
            else
                return false;
        }
public:
    float x0, y0, z0,
          x1, y1, z1;
private:
    const float eps = 1e-4;
};

#endif
