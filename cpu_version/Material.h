
/* ====================================================
#   Copyright (C)2017 All rights reserved.
#   Author        : Terence (Yongxin) Feng
#   Email         : tyxfeng@gmail.com
#   File Name     : Material.h
# ====================================================*/

#ifndef _MATERIAL_H
#define _MATERIAL_H

#include "BRDF.h"
#include "Light.h"
#include "RGBColor.h"
#include "ShadeRec.h"
#include <vector>

class GeometricObject;
class Material
{
public:

	virtual ~Material();
	virtual Material& operator = (const Material& rhs);

	virtual RGBColor shade(ShadeRec&) const = 0;
	virtual RGBColor area_light_shade(ShadeRec&) const = 0;
	virtual RGBColor path_shade(ShadeRec&) const = 0;

	virtual RGBColor get_Le(ShadeRec& sr) const;

};

class Matte: public Material
{
public:
	Matte(void);
	Matte(const Matte&);
	Matte(const float, const float, const RGBColor&);
	~Matte(void);

	virtual Matte& operator = (const Matte& rhs);

	void set_ka(const float);
	inline float get_ka() const { return ambient_brdf->get_kd(); }
	void set_kd(const float);
	inline float get_kd() const { return diffuse_brdf->get_kd(); }
	void set_cd(const RGBColor&);

	virtual RGBColor shade(ShadeRec&) const;
	virtual RGBColor area_light_shade(ShadeRec&) const;
	virtual RGBColor path_shade(ShadeRec&) const;

private:
	Lambertian *ambient_brdf;
	Lambertian *diffuse_brdf;
	RGBColor cd;
};

class Phong: public Material
{
public:
	Phong(void);
	virtual RGBColor shade(ShadeRec&) const;
	virtual RGBColor area_light_shade(ShadeRec&) const;
	virtual RGBColor path_shade(ShadeRec&) const;

	void set_ka(const float ka_);
	void set_kd(const float kd_);
	void set_ks(const float ks_);
	void set_es(const float es_);
	void set_cd(const RGBColor cd_);
private:
	Lambertian *ambient_brdf;
	Lambertian *diffuse_brdf;
	GlossySpecular *specular_brdf;
	RGBColor cd;
};

class Emissive: public Material
{
public:
	Emissive(void);
	Emissive(int, const RGBColor&);

	virtual RGBColor shade(ShadeRec&) const;
	virtual RGBColor area_light_shade(ShadeRec& sr) const;
	virtual RGBColor path_shade(ShadeRec& sr) const;
	virtual RGBColor get_Le(ShadeRec& sr) const;
private:
	float ls; /* radiance scaling factor */
	RGBColor ce; /* color */
};

#endif
