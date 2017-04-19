#pragma once

class Light
{
   public:
    virtual Color Sample_L(const Point &p, Vector *wi, float *pdf) const = 0;
    virtual Vector getDirection(const Point &hit) const = 0;
};

class PointLight : public Light
{
   private:
    Point lightPos;
    Color intensity;

   public:
    PointLight(const Point &pos, const Color &_intensity)
        : lightPos(pos), intensity(_intensity)
    {
    }

    // Power     is Flux
    // L         is E (Irradiance)
    // Intensity is Eo (E when r = 1)

    // Flux = E * 4*PI * R*R = Eo * 4*PI * 1*1

    // 对光源的采样，实际是对 Irradiance 的采样。

    // 此函数采样的是：在目标点 p 处，光源的 E，记作 Ep
    // Flux = Ep * 4*PI * R*R = Eo * 4*PI * 1*1
    //    => Ep = Eo / (R*R)
    // XXX: 入射方向直接计算
    // XXX: pdf 直接为 1
    // 用于计算物体表面亮度
    virtual Color Sample_L(const Point &p, Vector *wi, float *pdf) const
    {
        // *wi = Point::Norm(lightPos - p);
        // *pdf = 1.f;
        return Point::Scale(intensity.v, 1.f / DistanceSquared(lightPos, p));
    }

    virtual Vector getDirection(const Point &hit) const
    {
        return (lightPos - hit).norm();
    }
};
