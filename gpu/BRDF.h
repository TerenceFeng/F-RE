#pragma once

#include "sampler.h"

// DATA
struct BSDFParam
{
    Normal nr;
    Vector wo;

    Vector wi;
    Color f;
    float pdf;
};

struct Lambertian
{
    Color R;
};
struct SpecularReflection
{
    Color R;
};
struct SpecularTransmission
{
    Color R;
};

// Pool<Lambertian> Lpool;
// Pool<SpecularReflection> Rpool;
// Pool<SpecularTransmission> Tpool;

// FUNC
struct ComputeBSDF
{
    BSDFParam param;
    float ratio[3];
    // model_t model[3];
    Lambertian l;
    SpecularReflection r;
    SpecularTransmission t;

    void compute(float pick, Normal nr, Vector wo)
    {
        param.nr = nr;
        param.wo = wo;
        if (pick <= ratio[0])
            _f0(param, l);
        else if (pick <= ratio[1])
            _f1(param, r);
        else // if (pick <= ratio[2])
            _f2(param, t);
    }
    Color f() const
    {
        return param.f;
    }
    float pdf() const
    {
        return param.pdf;
    }
    Vector wi() const
    {
        return param.wi;
    }

   private:
    static void _f0(BSDFParam &param, const Lambertian &model)
    {
        param.wi = CosineSampleHemisphere(frandom(), frandom());
        if (param.wi.dot(param.nr) < 0.0f) param.wi = -param.wi;
        param.f.v = Vector::Scale(model.R.v, 1.0f / 3.14159f);
        param.pdf = 1.0f / 3.14159f;
    }
    static void _f1(BSDFParam &param, const SpecularReflection &model)
    {
        Vector nr = param.nr;
        Vector wo = param.wo;
        param.wi = Vector(nr).scale(wo.dot(nr)).sub(wo).scale(2.0f).add(wo).norm();
    }
    static void _f2(BSDFParam &param, const SpecularTransmission &model)
    {
    }
};

