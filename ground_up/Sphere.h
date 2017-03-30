
/* ====================================================
#   Copyright (C)2017 All rights reserved.
#
#   Author        : Terence (Yongxin) Feng
#   Email         : tyxfeng@gmail.com
#   File Name     : Sphere.h
#   Last Modified : 2017-03-21 20:28
#   Describe      :
#
#   Log           :
#
# ====================================================*/

#ifndef  _SPHERE_H
#define  _SPHERE_H

#include "Utilities.h"
#include "GeometricObject.h"

class Sphere: public GeometricObject
{
public:

	Sphere();
	Sphere(const Point3D& ct, float r);
	Sphere(const Point3D& ct, float r, const RGBColor& c);

	bool
	hit(const Ray& ray, float& tmin, ShadeRec& sr) const;
	void set_center(float f);
	void set_center(float x, float y, float z);
	void set_radius(float r);

private:
	Point3D center;
	float radius;
	const float eps = 1e-4;
};



#endif // _SPHERE_H


