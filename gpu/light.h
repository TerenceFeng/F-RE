#pragma once

struct LightParam
{
    Color L;
};

struct PointLight
{
    Color intensity;
};

struct ComputeLight
{
    LightParam param;
    PointLight light;

    void compute(const Point &pos, const Vector &dir)
    {
        param.L = light.intensity; //Point::Scale(light.intensity.v, 1.f / DistanceSquared(light.pos, pos));
    }
    Color L() const
    {
        return param.L;
    }
};
