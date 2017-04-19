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
	/*
	 * virtual RGBColor f(const ShadeRec& sr, const Vector3D& wi, const Vector3D& wo) const;
	 * virtual RGBColor rho(const ShadeRec& sr, const Vector3D& wo) const;
	 */
	BRDF(void);
	void set_sampler(Sampler*);
	virtual void set_color(const RGBColor&);
protected:
	Sampler* sampler_ptr;
	RGBColor color;
};

class Lambertian: public BRDF
{
public:
	Lambertian();
	Lambertian(const float, const RGBColor&);
	virtual RGBColor f(const ShadeRec&, const Vector3D& wo, const Vector3D& wi) const;
	RGBColor sample_f(const ShadeRec&, const Vector3D& wo, Vector3D& wi, float& pdf) const;
	virtual RGBColor rho(const ShadeRec& sr, const Vector3D& wo) const;
	void set_kd(const float);

private:
	/* diffuse reflection coefficient */
	float kd;
};

class GlossySpecular: public BRDF
{
public:
	GlossySpecular(void);
	GlossySpecular(float ks_, float e_, RGBColor cd_);
	GlossySpecular(const Lambertian& g_);
	virtual RGBColor f(const ShadeRec&, const Vector3D&, const Vector3D&) const;
	RGBColor sample_f(const ShadeRec&, const Vector3D&, Vector3D&, float& pdf) const;
	virtual RGBColor rho(const ShadeRec&, const Vector3D&) const;
	void set_samples(const int, const float);
	RGBColor get_color(void);

	void set_ks(const float ks_);
	void set_e(const float e_);

private:
	float ks;
	float e;
};

class PerfectSpecular: public BRDF
{
public:
	PerfectSpecular(void);
	PerfectSpecular(const float kr_, const RGBColor& cr_);
	RGBColor sample_f(const ShadeRec& sr, const Vector3D& wo, Vector3D& wi, float& pdf) const;

	void set_kr(const float kr);
private:
	float kr;
};

#endif // _BRDF_H


