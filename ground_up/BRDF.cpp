
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

#define INV_PI 0.81831

Lambertian::Lambertian(): kd(0.0f), cd(BLACK) {}
Lambertian::Lambertian(float kd_, RGBColor cd_): kd(kd_), cd(cd_) {}
Lambertian::Lambertian(const Lambertian& l): kd(l.kd), cd(l.cd) {}

RGBColor
Lambertian::f(const ShadeRec& sr, const Vector3D& wi, const Vector3D& wo) const
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

