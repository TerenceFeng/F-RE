
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
#include <vector>

class Material;
class GeometricObject;

class Light
{
public:
	Light(void);
	virtual ~Light(void);

	inline void set_shadows(bool s) {
		shadows = s;
	}
	inline bool cast_shadows() const {
		return shadows;
	}

	virtual Vector3D get_direction(ShadeRec& sr) = 0;
	virtual RGBColor L(ShadeRec& sr) const = 0;
	virtual bool in_shadow(const Ray& ray) const = 0;
	virtual float G(const ShadeRec&) const;
	virtual float pdf(ShadeRec&) const;

protected:
	bool shadows;
};


class Ambient: public Light
{
public:
	Ambient(void);
	Ambient(float ls_, RGBColor color_);
	Ambient(float ls_, RGBColor color_, bool shadows_);
	virtual Vector3D get_direction(ShadeRec& sr);
	virtual RGBColor L(ShadeRec& sr) const;
	virtual bool in_shadow(const Ray& ray) const;

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
	virtual Vector3D get_direction(ShadeRec& sr);
	virtual RGBColor L(ShadeRec& sr) const;
	virtual bool in_shadow(const Ray& ray) const;

	inline void set_radiance(const float b) {ls = b;}
	inline void set_location(const Point3D& location_) {location = location_;}

private:
	float ls;
	RGBColor color;
	Point3D location;
};

class AreaLight: public Light
{
public:
	AreaLight(void);
	AreaLight(GeometricObject*, Material*);
	~AreaLight(void);

	virtual bool in_shadow(const Ray&) const;

	virtual RGBColor L(ShadeRec&) const;
	virtual Vector3D get_direction(ShadeRec&);
	virtual float G(const ShadeRec&) const;
	virtual float pdf(ShadeRec&) const;
	void set_object(GeometricObject* object_ptr_);
	void set_material(Material* material_ptr_);

private:
	bool V(const Ray&) const;
	GeometricObject* object_ptr;
	Material* material_ptr;
	Point3D sample_point;
	Normal light_normal;
	Vector3D wi;
};

#endif
