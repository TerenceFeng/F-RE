
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
#include <cmath>
#include <cstdio>

#define INV_PI 0.81831

Lambertian::Lambertian(): kd(0.0f), cd(BLACK) {}
Lambertian::Lambertian(float kd_, RGBColor cd_): kd(kd_), cd(cd_) {}
Lambertian::Lambertian(const Lambertian& l): kd(l.kd), cd(l.cd) {}

RGBColor
Lambertian::f(const ShadeRec& sr, const Vector3D& wo, const Vector3D& wi) const
{
	/*
	 * std::cout << cd.r << " " << cd.g << " " << cd.b << std::endl;
	 * exit(0);
	 */
	return (cd * (kd * INV_PI));
}

RGBColor
Lambertian::rho(const ShadeRec& sr, const Vector3D& wo) const
{
	return (cd * kd);
}

/* glossy specular */
GlossySpecular::GlossySpecular(): ks(0.0f), e(0.0f), cd(BLACK) {}
GlossySpecular::GlossySpecular(float ks_, float e_, RGBColor cd_):
	ks(ks_),
	e(e_),
	cd(cd_)
{}

void
GlossySpecular::set_ks(const float ks_)
{ ks = ks_; }
void
GlossySpecular::set_e(const float e_)
{ e = e_; }
void
GlossySpecular::set_cd(const RGBColor& cd_)
{ cd = cd_; }

RGBColor
GlossySpecular::f(const ShadeRec& sr, const Vector3D& wo, const Vector3D& wi) const
{
	RGBColor L;
	float ndotwi = sr.normal * wi;
	Vector3D r(-wi + sr.normal * ndotwi * 2.0f);
	r.normalize();
	float rdotwo = r * wo;
	if (rdotwo > 0.0f) {
		L = ks * pow(rdotwo, e);
	}
	return L;
}

RGBColor
GlossySpecular::rho(const ShadeRec& sr, const Vector3D& wo) const
{
	return cd * ks;
}
