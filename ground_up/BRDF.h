/* ====================================================
#   Copyright (C)2017 All rights reserved.
#   Author        : Terence (Yongxin) Feng
#   Email         : tyxfeng@gmail.com
#   File Name     : BRDF.h
#   Last Modified : 2017-03-24 16:47
# ====================================================*/

#ifndef  _BRDF_H
#define  _BRDF_H

#include "RGBColor.h"
#include "ShadeRec.h"
#include "Utilities.h"

class BRDF
{
public:
	virtual RGBColor
		f(const ShadeRec& sr, const Vector3D& wi, const Vector3D& wo) const = 0;

	virtual RGBColor
		rho(const ShadeRec& sr, const Vector3D& wo) const = 0;

};

class Lambertian: public BRDF
{
public:
	Lambertian();
	Lambertian(float kd_, RGBColor cd_);
	Lambertian(const Lambertian& l);
	virtual RGBColor
		f(const ShadeRec& sr, const Vector3D& wi, const Vector3D& wo) const;
	virtual RGBColor
		rho(const ShadeRec& sr, const Vector3D& wo) const;
	inline RGBColor
		get_color() const { return cd; }
	/*
	 * inline Lambertian&
	 *     operator = (const Lambertian& rhs) {
	 *         kd = rhs.kd;
	 *         cd = rhs.kd;
	 *         return (*this);
	 *     }
	 */
	inline void set_kd(const float& kd_) { kd = kd_; }
	inline float get_kd() const { return kd; }
	inline void set_cd(const RGBColor& cd_) { cd = cd_; }
	inline RGBColor get_cd() const { return cd; }

private:
	/* diffuse reflection coefficient */
	float kd;
	RGBColor cd;
};


#endif // _BRDF_H


