
/* ====================================================
#   Copyright (C)2017 All rights reserved.
#   Author        : Terence (Yongxin) Feng
#   Email         : tyxfeng@gmail.com
#   File Name     : Light.cpp
# ====================================================*/

#include "Light.h"
#include "World.h"
#include "Sampler.h"
#include "Material.h"
#include "object/Object.h"
#include <cfloat>

extern World world;

Light::Light(void) {}
Light::~Light(void) {}

float
Light::G(const ShadeRec& sr) const
{	return 1; }
float
Light::pdf(ShadeRec& sr) const
{	return 1; }
void
Light::set_sampler(Sampler *s_)
{
	sampler_ptr = s_;
}

/* Ambient */
Ambient::Ambient(void):
	Light(),
	ls(1.0f),
	color(1.0f)
{}
Ambient::Ambient(float ls_, RGBColor color_):
	Light(),
	ls(ls_),
	color(color_)
{}

Vector3D
Ambient::get_direction(ShadeRec& sr) {
	return Vector3D(0.0);
}

RGBColor
Ambient::L(ShadeRec& sr)
{
	return color * ls;
}
bool
Ambient::in_shadow(const Ray& ray) const
{ return false; }

/* NOTE: PointLight */
PointLight::PointLight(void):
	Light(),
	ls(1.0f),
	color(1.0f),
	location(Vector3D(0.0f))
{
}
PointLight::PointLight(float ls_, const RGBColor& color_, const Vector3D& location_):
	Light(),
	ls(ls_),
	color(color_),
	location(location_)
{
}

Vector3D
PointLight::get_direction(ShadeRec& sr)
{
	return (location - sr.hit_point).hat();
}

RGBColor
PointLight::L(ShadeRec& sr)
{
	return color * ls;
}
bool
PointLight::in_shadow(const Ray& ray) const
{
	float t = FLT_MAX;
	float d = location.distance(ray.o);
	for (auto obj_ptr: world.obj_ptrs)
		if (obj_ptr->shadow_hit(ray, t) && t < d)
			return true;
	return false;
}

/* NOTE: implementation of AreaLight */
AreaLight::AreaLight(void):
	Light()
{
}

AreaLight::AreaLight(Object* object_ptr_, Material* material_ptr_)
{
	material_ptr = material_ptr_;
	object_ptr = object_ptr_;
}

AreaLight::~AreaLight(void)
{
	delete object_ptr;
}

Vector3D
AreaLight::get_direction(ShadeRec& sr)
{
	sample_point = object_ptr->sample();
	light_normal = object_ptr->get_normal(sample_point);
	wi = sample_point - sr.hit_point;
	wi.normalize();
	return wi;
}

bool
AreaLight::in_shadow(const Ray& ray) const
{
	float t = FLT_MAX;
	float ts = (sample_point - ray.o) * ray.d;
	for (auto obj_ptr: world.obj_ptrs)
		if (obj_ptr->shadow_hit(ray, t) && t < ts)
			return true;
	return false;
}

void
AreaLight::set_object(Object* object_ptr_)
{
	object_ptr = object_ptr_;
}
void
AreaLight::set_material(Material* material_ptr_)
{
	material_ptr = material_ptr_;
}

RGBColor
AreaLight::L(ShadeRec& sr)
{
	float ndotd = -light_normal * wi;
	if (ndotd > 0.0f)
		return material_ptr->get_Le(sr);
	else
		return BLACK;
}

float
AreaLight::G(const ShadeRec& sr) const
{
	float ndotd = -light_normal * wi;
	float dsqr = sample_point.distance_sqr(sr.hit_point);
	return (ndotd / dsqr);
}

float
AreaLight::pdf(ShadeRec& sr) const
{
	return object_ptr->pdf(sr);
}

/* NOTE: implementation of AmbientOccluder */
AmbientOccluder::AmbientOccluder(void):
	color(1, 1, 1),
	ls(1),
	min_amount(1, 1, 1)
{}

AmbientOccluder::AmbientOccluder(float ls_, const RGBColor& color_):
	ls(ls_),
	color(color_),
	min_amount(1, 1, 1)
{}

AmbientOccluder::AmbientOccluder(float ls_, const RGBColor& color_, const RGBColor& min_amount_):
	ls(ls_),
	color(color_),
	min_amount(min_amount_)
{}

Vector3D
AmbientOccluder::get_direction(ShadeRec& sr)
{
	Point3D sp = sampler_ptr->sample_unit_hemisphere();
	return (u * sp.x + v * sp.y + w * sp.z);
}

bool
AmbientOccluder::in_shadow(const Ray& ray) const
{
	float t = FLT_MAX;
	for (Object* obj_ptr: world.obj_ptrs)
		if (obj_ptr->shadow_hit(ray, t))
			return true;
	return false;
}

RGBColor
AmbientOccluder::L(ShadeRec& sr)
{
	w = sr.normal;
	v = w ^ Vector3D(0.0072, 1.0, 0.0034);
	v.normalize();
	u = v ^ w;
	Ray shadow_ray(sr.hit_point, get_direction(sr));
	if (in_shadow(shadow_ray))
		return min_amount * ls * color;
	else
		return color * ls;
}

/* NOTE: implementation of EnviormentLight */
EnviormentLight::EnviormentLight(void):
	Light()
{
}

EnviormentLight::EnviormentLight(Sampler *s_, Material *m_):
	Light(),
	material_ptr(m_)
{
	sampler_ptr = s_;
}

EnviormentLight::~EnviormentLight()
{}

void
EnviormentLight::set_material(Material* material_ptr_)
{
	material_ptr = material_ptr_;
}

bool
EnviormentLight::in_shadow(const Ray& ray) const
{
	float t = FLT_MAX;
	for (auto& obj_ptr: world.obj_ptrs)
		if (obj_ptr->shadow_hit(ray, t))
			return true;
	return false;
}

RGBColor
EnviormentLight::L(ShadeRec& sr)
{
	return material_ptr->get_Le(sr);
}

Vector3D
EnviormentLight::get_direction(ShadeRec& sr)
{
	w = sr.normal;
	v = Vector3D(0.0034, 1, 0.0071) ^ w;
	v.normalize();
	u = v ^ w;
	Point3D sp = sampler_ptr->sample_unit_hemisphere();
	return (u * sp.x + v * sp.y + w * sp.z);
}
