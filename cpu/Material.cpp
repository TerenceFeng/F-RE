
/* ====================================================
#   Copyright (C)2017 All rights reserved.
#   Author        : Terence (Yongxin) Feng
#   Email         : tyxfeng@gmail.com
#   File Name     : Material.cpp
# ====================================================*/
#include "World.h"
#include "Material.h"
#include "Object.h"
#include <cmath>
#include <cstdio>

#define INV_PI 0.31831f

extern World world;

Material::Material():
	color(BLACK)
{}

Material::~Material() {}
RGBColor
Material::area_light_shade(ShadeRec&) const
{ return BLACK; }

RGBColor
Material::path_shade(ShadeRec&) const
{ return BLACK; }

RGBColor
Material::global_shade(ShadeRec& sr) const
{
	sr.depth++;
	return BLACK;
}

RGBColor
Material::get_Le(ShadeRec& sr) const
{ return BLACK; }

void
Material::set_color(const RGBColor& c_)
{
	color = c_;
}

/* implementation of Matte */
Matte::Matte(void):
	ambient_brdf(new Lambertian),
	diffuse_brdf(new Lambertian)
{}
Matte::Matte(const float ka_, const float kd_, const RGBColor& c_):
	ambient_brdf(new Lambertian),
	diffuse_brdf(new Lambertian)
{
	set_ka(ka_);
	set_kd(kd_);
	color = c_;
	set_color(c_);
}

Matte::~Matte(void)
{
	delete ambient_brdf;
	delete diffuse_brdf;
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
Matte::set_color(const RGBColor& c_)
{
	color = c_;
	ambient_brdf->set_color(c_);
	diffuse_brdf->set_color(c_);
}

RGBColor
Matte::area_light_shade(ShadeRec& sr) const
{
	Vector3D wo = -sr.ray.d;
	RGBColor L = ambient_brdf->rho(sr, wo) * world.ambient_ptr->L(sr);

	for (auto light_ptr: world.light_ptrs)
	{
		Vector3D wi = light_ptr->get_direction(sr);
		float ndotwi = sr.normal * wi;

		if (ndotwi > 0.0f)
		{
			Ray shadow_ray(sr.hit_point, wi);
			bool in_shadow = light_ptr->in_shadow(shadow_ray);

			if (!in_shadow)
			{
				L += diffuse_brdf->f(sr, wo, wi)
						* light_ptr->L(sr)
						* light_ptr->G(sr)
						* ndotwi
						/ light_ptr->pdf(sr);
			}
		}
	}

	return L;
}

RGBColor
Matte::path_shade(ShadeRec& sr) const
{
	sr.color = area_light_shade(sr);

	float pdf;
	Vector3D wi, wo = -sr.ray.d;
	RGBColor f = diffuse_brdf->sample_f(sr, wo, wi, pdf);
	float ndotwi = sr.normal * wi;
	float x = ndotwi / pdf;

	sr.reflected_dir = wi;
	if (isnan(x))
	{
		return f;
	}
	else
		return f * (ndotwi / pdf);
}

RGBColor
Matte::global_shade(ShadeRec& sr) const
{
	RGBColor L;
	if (sr.depth == 0)
		L = area_light_shade(sr);
	float pdf;
	Vector3D wi, wo = -sr.ray.d;
	RGBColor f = diffuse_brdf->sample_f(sr, wo, wi, pdf);
	float ndotwi = sr.normal * wi;
	float x = ndotwi / pdf;

	sr.depth++;
	sr.reflected_dir = wi;
	if (isnan(x))
		return L + f;
	else
		return L + f * (ndotwi / pdf);
}

/* NOTE: Phong */
Phong::Phong(void):
	ambient_brdf(new Lambertian),
	diffuse_brdf(new Lambertian),
	specular_brdf(new GlossySpecular)
{}

Phong::Phong(const float ka_, const float kd_, const float ks_, const float es_, const RGBColor& c_):
	ambient_brdf(new Lambertian),
	diffuse_brdf(new Lambertian),
	specular_brdf(new GlossySpecular)
{
	ambient_brdf->set_kd(ka_);
	ambient_brdf->set_color(c_);
	diffuse_brdf->set_kd(kd_);
	diffuse_brdf->set_color(c_);
	specular_brdf->set_ks(ks_);
	specular_brdf->set_color(color);
	specular_brdf->set_e(es_);
}

void
Phong::set_ka(const float ka_)
{ ambient_brdf->set_kd(ka_); }
void
Phong::set_kd(const float kd_)
{ diffuse_brdf->set_kd(kd_); }
void
Phong::set_ks(const float ks_)
{ specular_brdf->set_ks(ks_); }
void
Phong::set_es(const float es_)
{ specular_brdf->set_e(es_); }
void
Phong::set_color(const RGBColor& c_)
{
	ambient_brdf->set_color(c_);
	diffuse_brdf->set_color(c_);
	specular_brdf->set_color(c_);
}
void
Phong::set_sampler(Sampler* s_)
{
	specular_brdf->set_sampler(s_);
}

RGBColor
Phong::area_light_shade(ShadeRec& sr) const
{
	Vector3D wo = -sr.ray.d;
	wo.normalize();
	RGBColor L;

	for (auto light_ptr: world.light_ptrs)
	{
		Vector3D wi = light_ptr->get_direction(sr);
		wi.normalize();
		float ndotwi = sr.normal * wi;
		if (ndotwi > 0.0f) {
			Ray shadowRay(sr.hit_point, wi);
			bool in_shadow = light_ptr->in_shadow(shadowRay);

			if (!in_shadow)
				L += (diffuse_brdf->f(sr, wo, wi)
						+ specular_brdf->f(sr, wo, wi))
					* light_ptr->L(sr)
					* light_ptr->G(sr)
					* ndotwi
					/ light_ptr->pdf(sr);
		}
	}

	return L;
}

RGBColor
Phong::path_shade(ShadeRec& sr) const
{
	sr.color = area_light_shade(sr);

	float pdf;
	Vector3D wi, wo = -sr.ray.d;
	RGBColor f = specular_brdf->sample_f(sr, wo, wi, pdf);
	float ndotwi = sr.normal * wi;
	sr.reflected_dir = wi;
	float x = ndotwi / pdf;

	sr.reflected_dir = wi;
	if (isnan(x))
		return f;
	else
	{
		return f * ((sr.normal * wi) * (ndotwi / pdf));
	}
}


/* NOTE: implementation of Emissive */
Emissive::Emissive(void):
	ls(0)
{
	color = BLACK;
}

Emissive::Emissive(int ls_, const RGBColor& c_):
	ls(ls_)
{
	color = c_;
}

RGBColor
Emissive::path_shade(ShadeRec& sr) const
{
	if (-sr.normal * sr.ray.d > 0.0)
		return color * ls;
	else
		return BLACK;
}

RGBColor
Emissive::area_light_shade(ShadeRec& sr) const
{
	if (-sr.normal * sr.ray.d > 0.0)
		return color * ls;
	else
		return BLACK;
}

RGBColor
Emissive::get_Le(ShadeRec& sr) const
{
	return color * ls;
}

/* NOTE: implementation of Emissive */
Reflective::Reflective(void):
	Phong(),
	reflective_brdf(new PerfectSpecular)
{}

Reflective::Reflective(const float ka_, const float kd_, const float ks_, const float kr_, const float es_, const RGBColor& cd_, const RGBColor& cr_):
	Phong(),
	reflective_brdf(new PerfectSpecular)
{
	Phong::set_ka(ka_);
	Phong::set_kd(kd_);
	Phong::set_ks(ks_);
	Phong::set_es(es_);
	Phong::set_color(cd_);
	set_color(cr_);
	set_kr(kr_);
}

void
Reflective::set_color(const RGBColor& c_)
{
	reflective_brdf->set_color(c_);
}

void
Reflective::set_kr(const float kr_)
{
	reflective_brdf->set_kr(kr_);
}

RGBColor
Reflective::area_light_shade(ShadeRec& sr) const
{
	RGBColor L(Phong::area_light_shade(sr));

	Vector3D wo = -sr.ray.d;
	Vector3D wi;
	float dummy_pdf;
	RGBColor fr = reflective_brdf->sample_f(sr, wo, wi, dummy_pdf);
	sr.reflected_dir = wi;

	sr.color += L;
	return fr * (sr.normal * wi);
}

RGBColor
Reflective::path_shade(ShadeRec& sr) const
{
	return area_light_shade(sr);
/*
 *     sr.color = Phong::area_light_shade(sr);
 * 
 *     Vector3D wo = -sr.ray.d;
 *     Vector3D wi;
 *     float dummy_pdf; [> always 1 in PerfectSpecular BRDf <]
 *     RGBColor fr = reflective_brdf->sample_f(sr, wo, wi, dummy_pdf);
 *     sr.reflected_dir = wi;
 * 
 *     return fr * (sr.normal * wi);
 */
}

RGBColor
Reflective::global_shade(ShadeRec& sr) const
{
	Vector3D wo = -sr.ray.d;
	Vector3D wi;
	float dummy_pdf; /* always 1 in PerfectSpecular BRDf */
	RGBColor fr = reflective_brdf->sample_f(sr, wo, wi, dummy_pdf);

	sr.reflected_dir = wi;
	sr.depth++;
	return fr * (sr.normal * wi);
}

/* NOTE: implementation of GlossyReflective */
GlossyReflective::GlossyReflective(void):
	Phong(),
	glossy_specular_brdf(new GlossySpecular)
{}

GlossyReflective::GlossyReflective(const float ka_, const float kd_, const float ks_, const float kr_, float es_, const RGBColor& c_):
	Phong(),
	glossy_specular_brdf(new GlossySpecular)
{
	Phong::set_ka(ka_);
	Phong::set_kd(kd_);
	Phong::set_ks(ks_);
	Phong::set_es(es_);
	set_color(c_);
	set_kr(kr_);
	glossy_specular_brdf->set_samples(100, es_);
}

void
GlossyReflective::set_kr(const float kr_)
{
	glossy_specular_brdf->set_ks(kr_);
}

void
GlossyReflective::set_color(const RGBColor& c_)
{
	Phong::set_color(c_);
	glossy_specular_brdf->set_color(c_);
}

void
GlossyReflective::set_exponent(const float e_)
{
	glossy_specular_brdf->set_e(e_);
	Phong::set_es(e_);
	glossy_specular_brdf->set_samples(100, e_);
}

void
GlossyReflective::set_sampler(Sampler *s_)
{
	glossy_specular_brdf->set_sampler(s_);
	glossy_specular_brdf->set_samples();
}

RGBColor
GlossyReflective::area_light_shade(ShadeRec& sr) const
{
	sr.color = Phong::area_light_shade(sr);

	Vector3D wo(-sr.ray.d);
	wo.normalize();
	Vector3D wi;
	float pdf;
	RGBColor fr(glossy_specular_brdf->sample_f(sr, wo, wi, pdf));
	sr.reflected_dir = wi;

	float ndotwi = (sr.normal * wi);
	return fr * ndotwi / pdf;
}

RGBColor
GlossyReflective::path_shade(ShadeRec& sr) const
{
	sr.color = Phong::area_light_shade(sr);

	Vector3D wo = -sr.ray.d;
	Vector3D wi;
	float pdf;
	RGBColor fr = glossy_specular_brdf->sample_f(sr, wo, wi, pdf);

	sr.reflected_dir = wi;
	return fr * (sr.normal * wi) / pdf;
}

RGBColor
GlossyReflective::global_shade(ShadeRec& sr) const
{

	Vector3D wo = -sr.ray.d;
	Vector3D wi;
	float pdf;
	RGBColor fr = glossy_specular_brdf->sample_f(sr, wo, wi, pdf);

	sr.reflected_dir = wi;

	if (sr.depth == 0) sr.depth = 1;
	return fr * (sr.normal * wi) / pdf;
}
