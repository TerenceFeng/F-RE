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

class Sampler;

class BRDF
{
public:
	virtual RGBColor f(const ShadeRec& sr, const Vector3D& wi, const Vector3D& wo) const = 0;
	virtual RGBColor rho(const ShadeRec& sr, const Vector3D& wo) const = 0;
	void set_sampler(Sampler*);
protected:
	Sampler* sampler_ptr;
};

class Lambertian: public BRDF
{
public:
	Lambertian();
	Lambertian(float kd_, RGBColor cd_);
	Lambertian(const Lambertian& l);
	virtual RGBColor f(const ShadeRec&, const Vector3D& wo, const Vector3D& wi) const;
	RGBColor sample_f(const ShadeRec&, const Vector3D& wo, Vector3D& wi, float& pdf) const;
	virtual RGBColor rho(const ShadeRec& sr, const Vector3D& wo) const;
	inline RGBColor get_color() const { return cd; }
	inline void set_kd(const float& kd_) { kd = kd_; }
	inline float get_kd() const { return kd; }
	inline void set_cd(const RGBColor& cd_) { cd = cd_; }
	inline RGBColor get_cd() const { return cd; }

private:
	/* diffuse reflection coefficient */
	float kd;
	RGBColor cd;
};

class GlossySpecular: public BRDF
{
public:
	GlossySpecular(void);
	GlossySpecular(float ks_, float e_, RGBColor cd_);
	GlossySpecular(const Lambertian& g_);
	virtual RGBColor f(const ShadeRec&, const Vector3D&, const Vector3D&) const;
	virtual RGBColor rho(const ShadeRec&, const Vector3D&) const;
	inline RGBColor get_color() { return cd; }

	void set_ks(const float ks_);
	void set_e(const float e_);
	void set_cd(const RGBColor& cd_);

private:
	float ks;
	float e;
	RGBColor cd;
};

#endif // _BRDF_H


