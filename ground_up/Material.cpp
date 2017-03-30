
/* ====================================================
#   Copyright (C)2017 All rights reserved.
#   Author        : Terence (Yongxin) Feng
#   Email         : tyxfeng@gmail.com
#   File Name     : Material.cpp
# ====================================================*/
#include "Material.h"

#define INV_PI 0.31831f

Material::~Material() {}
Material&
Material::operator = (const Material& rhs) {
	return (*this);
}

Matte::Matte(void):
	ambient_brdf(new Lambertian),
	diffuse_brdf(new Lambertian)
{}
Matte::Matte(const float ka_, const float kd_, const RGBColor& cd_):
	cd(cd_),
	ambient_brdf(new Lambertian),
	diffuse_brdf(new Lambertian)
{
	set_ka(ka_);
	set_kd(kd_);
}
Matte::Matte(const Matte& rhs):
	cd(rhs.cd),
	ambient_brdf(new Lambertian),
	diffuse_brdf(new Lambertian)
{
	set_ka(rhs.get_ka());
	set_kd(rhs.get_kd());
}

Matte::~Matte(void)
{
	delete ambient_brdf;
	delete diffuse_brdf;
}

Matte&
Matte::operator = (const Matte& rhs)
{
	cd = rhs.cd;
	set_ka(rhs.get_ka());
	set_kd(rhs.get_kd());
	return (*this);
}

void
Matte::set_ka(const float ka_)
{
	ambient_brdf->set_kd(ka_);
}

void
Matte::set_kd(const float kd_)
{
	diffuse_brdf->set_kd(kd_);
}

void
Matte::set_cd(const RGBColor& cd_)
{
	ambient_brdf->set_cd(cd_);
	diffuse_brdf->set_cd(cd_);
}

RGBColor
Matte::shade(ShadeRec& sr, const Ambient* amb_ptr, const std::vector<Light*> light_ptrs) const
{
	Vector3D wo = -sr.ray.d;
	RGBColor L = ambient_brdf->rho(sr, wo) * amb_ptr->L(sr);
	int num_of_lights = light_ptrs.size();

	for (int i = 0; i < num_of_lights; i++)
	{
		Vector3D wi = light_ptrs[i]->get_direction(sr);
		float ndotwi = sr.normal * wi;

		if (ndotwi > 0.0f)
			L += (diffuse_brdf->f(sr, wo, wi) * light_ptrs[i]->L(sr)) * ndotwi;
	}

	return L;
}
