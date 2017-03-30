
/* ====================================================
#   Copyright (C)2017 All rights reserved.
#   Author        : Terence (Yongxin) Feng
#   Email         : tyxfeng@gmail.com
#   File Name     : Light.h
# ====================================================*/

#ifndef _LIGHT_H
#define _LIGHT_H

#include "RGBColor.h"
#include "ShadeRec.h"
#include "Utilities.h"

class Light
{
public:
	Light(void);
	virtual ~Light(void);

	inline void
		set_shadows(bool s) {
			shadows = s;
		}
	inline bool
		get_shadows() const {
			return shadows;
		}

	virtual Vector3D
		get_direction(ShadeRec& sr) = 0;
	virtual RGBColor
		L(ShadeRec& sr) const = 0;

protected:
	bool shadows;
};


class Ambient: public Light
{
public:
	Ambient(void);
	Ambient(float ls_, RGBColor color_);
	Ambient(float ls_, RGBColor color_, bool shadows_);
	virtual Vector3D
		get_direction(ShadeRec& sr);
	virtual RGBColor
		L(ShadeRec& sr) const;

	inline void scale_radiance(const float b) {ls = b;}
	inline void set_color(const RGBColor& color_) {color = color_;}

private:
	float ls;
	RGBColor color;
};

class PointLight: public Light
{
public:
	PointLight(void);
	PointLight(float ls_, RGBColor color_, Vector3D location_);
	PointLight(float ls_, RGBColor color_, Vector3D location_, bool shadows_);
	virtual Vector3D
		get_direction(ShadeRec& sr);
	virtual RGBColor
		L(ShadeRec& sr) const;

	inline void set_radiance(const float b) {ls = b;}
	inline void set_location(const Point3D& location_) {location = location_;}

private:
	float ls;
	RGBColor color;
	Point3D location;
};

#endif
