
/* ====================================================
#   Copyright (C)2017 All rights reserved.
#   Author        : Terence (Yongxin) Feng
#   Email         : tyxfeng@gmail.com
#   File Name     : Light.cpp
# ====================================================*/

#include "Light.h"
#include "World.h"
#include "Material.h"
#include "GeometricObject.h"
#include <cfloat>

extern World world;

Light::Light(void):
	shadows(false)
{}
Light::~Light(void) {}

float
Light::G(const ShadeRec& sr) const
{	return 1; }
float
Light::pdf(ShadeRec& sr) const
{	return 1; }


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
Ambient::Ambient(float ls_, RGBColor color_, bool shadows_):
	Light(),
	ls(ls_),
	color(color_)
{
	shadows = shadows_;
}

Vector3D
Ambient::get_direction(ShadeRec& sr) {
	return Vector3D(0.0);
}

RGBColor
Ambient::L(ShadeRec& sr) const
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
	shadows = true;
}
PointLight::PointLight(float ls_, RGBColor color_, Vector3D location_):
	Light(),
	ls(ls_),
	color(color_),
	location(location_)
{
	shadows = true;
}
PointLight::PointLight(float ls_, RGBColor color_, Vector3D location_, bool shadows_):
	Light(),
	ls(ls_),
	color(color_),
	location(location_)
{
	shadows = shadows_;
}
Vector3D
PointLight::get_direction(ShadeRec& sr)
{
	return (location - sr.hit_point).hat();
}

RGBColor
PointLight::L(ShadeRec& sr) const
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
	shadows = true;
}
AreaLight::AreaLight(GeometricObject* object_ptr_, Material* material_ptr_)
{
	material_ptr = material_ptr_;
	object_ptr = object_ptr_;
	shadows = true;
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
AreaLight::set_object(GeometricObject* object_ptr_)
{
	object_ptr = object_ptr_;
}
void
AreaLight::set_material(Material* material_ptr_)
{
	material_ptr = material_ptr_;
}

RGBColor
AreaLight::L(ShadeRec& sr) const
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
