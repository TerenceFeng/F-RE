
/* ====================================================
#   Copyright (C)2017 All rights reserved.
#
#   Author        : Terence (Yongxin) Feng
#   Email         : tyxfeng@gmail.com
#   File Name     : Sphere.cpp
#   Last Modified : 2017-03-21 20:29
#   Describe      :
#
#   Log           :
#
# ====================================================*/
#include "Sphere.h"

#include <cmath>

/* Implementation of Sphere */
Sphere::Sphere(): center(), radius(0.0f) {}
Sphere::Sphere(const Point3D& ct, float r): center(ct), radius(r) {}
Sphere::Sphere(const Point3D& ct, float r, const RGBColor& c):
	center(ct),
	radius(r)
{
}

bool
Sphere:: hit(const Ray& ray, float& tmin, ShadeRec& sr) const
{
	float t;
	Vector3D temp = ray.o - center;
	float a = ray.d * ray.d;
	float b = ray.d * 2.0f * temp;
	float c = temp * temp - radius * radius;
	float disc = b * b - 4.0f * a * c;

	if (disc < 0)
		return false;
	else {
		float e = sqrtf(disc);
		float denom = 2.0 * a;
		t = (-b - e) / denom;
		if (t > eps) {
			tmin = t;
			sr.normal = temp + ray.d * t;
			sr.local_hit_point = ray.o + ray.d * t;
			return true;
		}

		t = (-b + e) / denom;
		if (t > eps) {
			tmin = t;
			sr.normal = temp + ray.d * t;
			sr.local_hit_point = ray.o + ray.d * t;
			return true;
		}
	}
	return false;
}

void
Sphere:: set_center(float f) {
	center = Point3D(f);
}
void
Sphere:: set_center(float x, float y, float z) {
	center = Point3D(x, y, z);
}
void
Sphere:: set_radius(float r) {
	radius = r;
}

