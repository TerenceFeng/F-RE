
/* ====================================================
#   Copyright (C)2017 All rights reserved.
#   Author        : Terence (Yongxin) Feng
#   Email         : tyxfeng@gmail.com
#   File Name     : Material.h
# ====================================================*/

#ifndef _MATERIAL_H
#define _MATERIAL_H

#include "BRDF.h"
#include "RGBColor.h"
#include "ShadeRec.h"
#include <vector>

class Object;

class Material
{
public:

	Material();
	virtual ~Material();

	virtual RGBColor area_light_shade(ShadeRec&) const;
	virtual RGBColor path_shade(ShadeRec&) const;
	virtual RGBColor global_shade(ShadeRec&) const;

	virtual RGBColor get_Le(ShadeRec& sr) const;

	virtual void set_color(const RGBColor&);

protected:
	RGBColor color;
};

class Matte: public Material
{
public:
	Matte(void);
	Matte(const float, const float, const RGBColor&);
	~Matte(void);

	void set_ka(const float);
	void set_kd(const float);
	void set_color(const RGBColor&);

	virtual RGBColor area_light_shade(ShadeRec&) const;
	virtual RGBColor path_shade(ShadeRec&) const;
	virtual RGBColor global_shade(ShadeRec& sr) const;
private:
	Lambertian *ambient_brdf;
	Lambertian *diffuse_brdf;
};

class Phong: public Material
{
public:
	Phong(void);
	Phong(const float ka_, const float kd_, const float ks_, const float es_, const RGBColor&);
	virtual RGBColor area_light_shade(ShadeRec&) const;
	virtual RGBColor path_shade(ShadeRec&) const;

	void set_ka(const float ka_);
	void set_kd(const float kd_);
	void set_ks(const float ks_);
	void set_es(const float es_);
	void set_color(const RGBColor&);
	void set_sampler(Sampler*);

protected:
	Lambertian *ambient_brdf;
	Lambertian *diffuse_brdf;
	GlossySpecular *specular_brdf;
};

class Emissive: public Material
{
public:
	Emissive(void);
	Emissive(int, const RGBColor&);

	virtual RGBColor area_light_shade(ShadeRec& sr) const;
	virtual RGBColor path_shade(ShadeRec& sr) const;
	virtual RGBColor get_Le(ShadeRec& sr) const;
private:
	float ls; /* radiance scaling factor */
};

class Reflective: public Phong
{
public:
	Reflective(void);
	Reflective(const float ka_, const float kd_, const float ks_, const float kr_, const float es_, const RGBColor& cd_, const RGBColor& cr_);
	void set_color(const RGBColor&);
	void set_kr(const float kr_);

	virtual RGBColor area_light_shade(ShadeRec&) const;
	virtual RGBColor path_shade(ShadeRec&) const;
	virtual RGBColor global_shade(ShadeRec& sr) const;
private:
	PerfectSpecular *reflective_brdf;
};

class GlossyReflective: public Phong
{
public:
	GlossyReflective(void);
	GlossyReflective(const float ka_, const float kd_, const float ks_, const float kr_, float es_, const RGBColor& color);
	void set_kr(const float);
	void set_exponent(const float);
	void set_color(const RGBColor&);
	void set_sampler(Sampler *s_);

	virtual RGBColor area_light_shade(ShadeRec& sr) const;
	virtual RGBColor path_shade(ShadeRec& sr) const;
	virtual RGBColor global_shade(ShadeRec& sr) const;
private:
	GlossySpecular *glossy_specular_brdf;
};

#endif
