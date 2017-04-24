
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

class Sampler;
class Material;
class Object;

class Light
{
public:
	Light(void);
	virtual ~Light(void);

	virtual Vector3D get_direction(ShadeRec& sr) = 0;
	virtual RGBColor L(ShadeRec& sr) = 0;
	virtual bool in_shadow(const Ray& ray) const = 0;
	virtual float G(const ShadeRec&) const;
	virtual float pdf(ShadeRec&) const;
	virtual void set_sampler(Sampler *);

protected:
	Sampler *sampler_ptr;
};

class Ambient: public Light
{
public:
	Ambient(void);
	Ambient(float ls_, RGBColor color_);
	virtual Vector3D get_direction(ShadeRec& sr);
	virtual RGBColor L(ShadeRec& sr);
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
	PointLight(float ls_, const RGBColor& color_, const Vector3D& location_);
	virtual Vector3D get_direction(ShadeRec& sr);
	virtual RGBColor L(ShadeRec& sr);
	virtual bool in_shadow(const Ray& ray) const;

private:
	float ls;
	RGBColor color;
	Point3D location;
};

class AreaLight: public Light
{
public:
	AreaLight(void);
	AreaLight(Object*, Material*);
	~AreaLight(void);

	virtual bool in_shadow(const Ray&) const;

	virtual RGBColor L(ShadeRec&);
	virtual Vector3D get_direction(ShadeRec&);
	virtual float G(const ShadeRec&) const;
	virtual float pdf(ShadeRec&) const;
	void set_object(Object* object_ptr_);
	void set_material(Material* material_ptr_);

private:
	bool V(const Ray&) const;
	Object* object_ptr;
	Material* material_ptr;
	Point3D sample_point;
	Normal light_normal;
	Vector3D wi;
};

class AmbientOccluder: public Light
{
public:
	AmbientOccluder(void);
	AmbientOccluder(float, const RGBColor&);
	AmbientOccluder(float, const RGBColor&, const RGBColor&);
	virtual Vector3D get_direction(ShadeRec& sr);
	bool in_shadow(const Ray& ray) const;
	virtual RGBColor L(ShadeRec& sr);

private:
	Vector3D u, v, w;
	RGBColor color;
	float ls;
	RGBColor min_amount;
};

class EnviormentLight: public Light
{
public:
	EnviormentLight(void);
	EnviormentLight(Sampler*, Material*);
	~EnviormentLight();
	void set_material(Material*);
	virtual Vector3D get_direction(ShadeRec&);
	virtual RGBColor L(ShadeRec& sr);
	virtual bool in_shadow(const Ray&) const;

private:
	Material* material_ptr;
	Vector3D u, v, w;
	Vector3D wi;
};

#endif
