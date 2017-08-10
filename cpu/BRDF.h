/* ====================================================
#   Copyright (C)2017 All rights reserved.
#   Author        : Terence (Yongxin) Feng
#   Email         : tyxfeng@gmail.com
#   File Name     : BRDF.h
#   Last Modified : 2017-03-24 16:47
# ====================================================*/

#ifndef  _BRDF_H
#define  _BRDF_H

#include "sampler.h"
#include "RGBColor.h"
#include "ShadeRec.h"
#include "Utilities.h"

#ifndef INV_PI
#define INV_PI 0.31831f
#endif

extern NRooks sampler;

class BRDF
{
public:
	BRDF(void):
        color(BLACK)
    {}
	void set_sampler(Sampler* s)
    {
        sampler_ptr = s;
    }
	virtual void set_color(const RGBColor& c)
    {
        color = c;
    }

protected:
	Sampler* sampler_ptr;
	RGBColor color;
};

class Lambertian: public BRDF
{
public:
	Lambertian():
        BRDF(),
        kd(0.0f)
    {}
	Lambertian(const float kd_, const RGBColor& c_):
        kd(kd_)
    {
        color = c_;
    }
    virtual RGBColor f(const ShadeRec& sr, const Vector3D& wo, const Vector3D& wi) const
    {
        return (color * (kd * INV_PI));
    }

    RGBColor sample_f(const ShadeRec& sr, const Vector3D& wo, Vector3D& wi, float& pdf) const
    {
        Vector3D w = sr.normal;
        Vector3D v = Vector3D(0.0034, 1.0, 0.0071) ^ w;
        v.normalize();
        Vector3D u = v ^ w;
        Point3D sp = sampler.sample_unit_hemisphere();
        wi = u * sp.x + v * sp.y + v * sp.z;
        wi.normalize();
        pdf = sr.normal * wi * INV_PI;
        return color * kd * INV_PI;
    }

	virtual RGBColor rho(const ShadeRec& sr, const Vector3D& wo) const
    {
        return color * kd;
    }

    void set_kd(const float kd_)
    {
        kd = kd_;
    }

private:
	/* diffuse reflection coefficient */
	float kd;
};

class GlossySpecular: public BRDF
{
public:
    GlossySpecular():
        BRDF(),
        ks(0.0f),
        e(1.0f)
    {}

    GlossySpecular(float ks_, float e_, RGBColor c_):
        ks(ks_),
        e(e_)
    {
        color = c_;
    }

    GlossySpecular(const Lambertian& g_);
    virtual RGBColor f(const ShadeRec& sr, const Vector3D& wo, const Vector3D& wi) const
    {
        RGBColor L;
        float ndotwi = sr.normal * wi;
        Vector3D r(-wi + sr.normal * ndotwi * 2.0f);
        r.normalize();
        float rdotwo = r * wo;
        if (rdotwo > 0.0f) {
            L = color * ks * pow(rdotwo, e);
        }
        return L;
    }

    RGBColor sample_f(const ShadeRec& sr, const Vector3D& wo, Vector3D& wi, float& pdf) const
    {
        float ndotwo = sr.normal * wo;
        Vector3D r = -wo + sr.normal * ndotwo * 2.0;
        r.normalize();

        Vector3D w = r;
        Vector3D u = Vector3D(0.00424, 1, 0.00764) ^ w;
        u.normalize();
        Vector3D v = u ^ w;

        Point3D sp = sampler_ptr->sample_unit_hemisphere();

        if (sr.normal * wi < 0.0)
            wi = -(u * sp.x) - (v * sp.y) + w * sp.z;
        else
            wi = u * sp.x + v * sp.y + w * sp.z;

        float phong_lobe = pow(wi * r, e);
        pdf = phong_lobe * (sr.normal * wi);

        return color * ks * phong_lobe;
    }

    virtual RGBColor rho(const ShadeRec& sr, const Vector3D& wo) const
    {
        return color * ks;
    }


    void set_samples(const int num_samples = 100, const float exp = 5.0)
    {
        sampler_ptr = new NRooks(num_samples);
        sampler_ptr->map_samples_to_hemisphere(exp);
    }

    // void set_samples()
    // {
    //     sampler_ptr->map_samples_to_hemisphere(e);
    // }

    RGBColor get_color(void);

    void set_ks(const float ks_)
    { ks = ks_; }

    void set_e(const float e_)
    { e = e_; }

private:
	float ks;
	float e;
};

class PerfectSpecular: public BRDF
{
public:
    PerfectSpecular():
        BRDF(),
        kr(0.0)
    {}

    PerfectSpecular(const float kr_, const RGBColor& c_):
        kr(kr_)
    {
        color = c_;
    }

    RGBColor sample_f(const ShadeRec& sr, const Vector3D& wo, Vector3D& wi, float& f) const
    {
        float ndotwo = sr.normal * wo;
        wi = -wo + sr.normal * ndotwo * 2.0;

        return (color * kr) / (sr.normal * wi);
    }

    void set_kr(const float kr_)
    { kr = kr_; }

private:
	float kr;
};

#endif // _BRDF_H


