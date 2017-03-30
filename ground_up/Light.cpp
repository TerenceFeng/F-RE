
/* ====================================================
#   Copyright (C)2017 All rights reserved.
#   Author        : Terence (Yongxin) Feng
#   Email         : tyxfeng@gmail.com
#   File Name     : Light.cpp
# ====================================================*/

#include "Light.h"

Light::Light(void):
	shadows(false)
{}
Light::~Light(void) {}

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

/* PointLight */
PointLight::PointLight(void):
	Light(),
	ls(1.0f),
	color(1.0f),
	location(Vector3D(0.0f))
{}
PointLight::PointLight(float ls_, RGBColor color_, Vector3D location_):
	Light(),
	ls(ls_),
	color(color_),
	location(location_)
{}
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
