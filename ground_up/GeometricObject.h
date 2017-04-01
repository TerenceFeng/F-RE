
/* ====================================================
#   Copyright (C)2017 All rights reserved.
#
#   Author        : Terence (Yongxin) Feng
#   Email         : tyxfeng@gmail.com
#   File Name     : Object.h
#   Last Modified : 2017-03-21 20:18
#   Describe      :
#
#   Log           :
#
# ====================================================*/

#ifndef  _GEOMETRICOBJECT_H
#define  _GEOMETRICOBJECT_H

#include <vector>
#include "RGBColor.h"
#include "Utilities.h"
#include "ShadeRec.h"
#include "Material.h"

class GeometricObject
{
protected:
	Material *m_ptr;
	const float eps = 1e-4;
	/* reflection coefficient */
	GeometricObject& operator= (const GeometricObject& rhs);

public:
	GeometricObject(void);
	GeometricObject(Material *m_ptr_);
	GeometricObject(const GeometricObject& go);
	virtual ~GeometricObject(void);

	virtual bool hit(const Ray& r, float& tmin, ShadeRec& sr) const = 0;

	inline void set_material(Material *m_ptr_) {m_ptr = m_ptr_;}
	inline Material * get_material() {return m_ptr;}

	RGBColor get_reflected_color(ShadeRec&, const Ambient*, const std::vector<Light*>) const;
	RGBColor get_reflected_color(ShadeRec&, const Ambient*, const std::vector<Light*>, const std::vector<GeometricObject*>) const;
	virtual bool shadow_hit(const Ray& ray, float& tmin) const = 0;

};

class Sphere: public GeometricObject
{
public:

	Sphere();
	Sphere(const Point3D& ct, float r);
	Sphere(const Point3D& ct, float r, const RGBColor& c);

	bool hit(const Ray& ray, float& tmin, ShadeRec& sr) const;
	bool shadow_hit(const Ray& ray, float& tmin) const;
	void set_center(float f);
	void set_center(float x, float y, float z);
	void set_radius(float r);

private:
	Point3D center;
	float radius;
	const float eps = 1e-2;
};

class Plane: public GeometricObject
{
public:
	Plane();
	Plane(const Point3D p, const Normal& n);
	Plane(const Point3D p, const Normal& n, const RGBColor c, float kd);
	bool hit(const Ray& ray, float& tmin, ShadeRec& sr) const;
	bool shadow_hit(const Ray& ray, float& tmin) const;

private:
	Point3D point;
	Normal normal;
	const float eps = 1e-2;
};



#endif


