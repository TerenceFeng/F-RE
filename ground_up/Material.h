
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
// #include "Lambertian.h"

class Material
{
public:

	// Material(const Material& rhs);
	virtual ~Material();
	virtual Material& operator = (const Material& rhs);
	virtual RGBColor
		shade(ShadeRec& sr, const Ambient* amb_ptr, const std::vector<Light*> light_ptrs) const = 0;

/*
 *     virtual RGBColor
 *         area_light_shade(ShadeRec& sr);
 *
 *     virtual RGBColor
 *         path_shade(ShadeRec& sr);
 */
};

class Matte: public Material
{
public:
	Matte(void);
	Matte(const Matte& rhs);
	Matte(const float ka_, const float kd_, const RGBColor& cd_);
	~Matte(void);

	virtual Matte& operator = (const Matte& rhs);

	void
		set_ka(const float ka_);
	inline float
		get_ka() const { return ambient_brdf->get_kd(); }
	void
		set_kd(const float kd_);
	inline float
		get_kd() const { return diffuse_brdf->get_kd(); }
	void
		set_cd(const RGBColor& cd_);
	/*
	 * virtual RGBColor
	 *     shade(ShadeRec& sr);
	 */

	virtual RGBColor
		shade(ShadeRec& sr, const Ambient* amb_ptr, const std::vector<Light*> light_ptrs) const;

private:
	Lambertian *ambient_brdf;
	Lambertian *diffuse_brdf;
	RGBColor cd;
};

#endif
