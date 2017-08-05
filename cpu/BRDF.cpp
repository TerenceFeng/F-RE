
/* ====================================================
#   Copyright (C)2017 All rights reserved.
#
#   Author        : Terence (Yongxin) Feng
#   Email         : tyxfeng@gmail.com
#   File Name     : BRDF.cpp
#   Last Modified : 2017-03-24 16:50
#   Describe      :
#
#   Log           :
#
# ====================================================*/

#include "BRDF.h"
#include "sampler.h"
#include <cmath>
#include <cstdio>

#define INV_PI 0.81831
extern NRooks sampler;

/* NOTE: implementation of BRDF base class */
BRDF::BRDF(void):
	color(BLACK)
{}

void
BRDF::set_sampler(Sampler* sampler_ptr_)
{
	sampler_ptr = sampler_ptr_;
}

void
BRDF::set_color(const RGBColor& c_)
{
	color = c_;
}

Lambertian::Lambertian():
	BRDF(),
	kd(0.0f)
{}

Lambertian::Lambertian(const float kd_, const RGBColor& c_):
	kd(kd_)
{
	color = c_;
}

RGBColor
Lambertian::f(const ShadeRec& sr, const Vector3D& wo, const Vector3D& wi) const
{
	return (color * (kd * INV_PI));
}

RGBColor
Lambertian::sample_f(const ShadeRec& sr, const Vector3D& wo, Vector3D& wi, float& pdf) const
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

RGBColor
Lambertian::rho(const ShadeRec& sr, const Vector3D& wo) const
{
	return color * kd;
}

void
Lambertian::set_kd(const float kd_)
{ kd = kd_; }

/* NOTE: implementation glossy specular */
GlossySpecular::GlossySpecular():
	BRDF(),
	ks(0.0f),
	e(1.0f)
{}
GlossySpecular::GlossySpecular(float ks_, float e_, RGBColor c_):
	ks(ks_),
	e(e_)
{
	color = c_;
}

void
GlossySpecular::set_ks(const float ks_)
{ ks = ks_; }

void
GlossySpecular::set_e(const float e_)
{ e = e_; }

RGBColor
GlossySpecular::f(const ShadeRec& sr, const Vector3D& wo, const Vector3D& wi) const
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

RGBColor
GlossySpecular::sample_f(const ShadeRec& sr, const Vector3D& wo, Vector3D& wi, float& pdf) const
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

RGBColor
GlossySpecular::rho(const ShadeRec& sr, const Vector3D& wo) const
{
	return color * ks;
}

void
GlossySpecular::set_samples()
{
	sampler_ptr->map_samples_to_hemisphere(e);
}

void
GlossySpecular::set_samples(const int num_samples = 100, const float exp = 5.0)
{
	sampler_ptr = new NRooks(num_samples);
	sampler_ptr->map_samples_to_hemisphere(exp);
}

/* NOTE: implementation of PerfectSpecular */
PerfectSpecular::PerfectSpecular(void):
	BRDF(),
	kr(0.0)
{}

PerfectSpecular::PerfectSpecular(const float kr_, const RGBColor& c_):
	kr(kr_)
{
	color = c_;
}

RGBColor PerfectSpecular::sample_f(const ShadeRec& sr, const Vector3D& wo, Vector3D& wi, float& f) const
{
	float ndotwo = sr.normal * wo;
	wi = -wo + sr.normal * ndotwo * 2.0;

	return (color * kr) / (sr.normal * wi);
}

void
PerfectSpecular::set_kr(const float kr_)
{ kr = kr_; }

