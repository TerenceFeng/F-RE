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
	RGBColor get_color();
	void set_kd(const float);
	void set_cd(const RGBColor&);

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
	RGBColor sample_f(const ShadeRec&, const Vector3D&, Vector3D&, float& pdf) const;
	virtual RGBColor rho(const ShadeRec&, const Vector3D&) const;
	void set_samples(const int, const float);
	RGBColor get_color(void);

	void set_ks(const float ks_);
	void set_e(const float e_);
	void set_cd(const RGBColor& cd_);

private:
	float ks;
	float e; /* exponational coefficient */
	RGBColor cd;
};

class PerfectSpecular: public BRDF
{
public:
	PerfectSpecular(void);
	PerfectSpecular(const float kr_, const RGBColor& cr_);
	RGBColor sample_f(const ShadeRec& sr, const Vector3D& wo, Vector3D& wi, float& pdf) const;
	RGBColor get_color();

	void set_kr(const float kr);
	void set_cr(const RGBColor& cr_);
private:
	float kr;
	RGBColor cr;
};

#endif // _BRDF_H


