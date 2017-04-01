/* ====================================================
#   Author        : Terence (Yongxin) Feng
#   Email         : tyxfeng@gmail.com
#   File Name     : Object.cpp
#   Last Modified : 2017-03-21 20:18
# ====================================================*/

#include "GeometricObject.h"
#include "ShadeRec.h"
#include <cmath>

#define INV_PI 0.31831f

/* NOTE: Implementation of GeometricObject */
GeometricObject&
GeometricObject::operator = (const GeometricObject& rhs)
{
	return (*this);
}

GeometricObject::GeometricObject(void) {}
GeometricObject::GeometricObject(Material *m_ptr_):
	m_ptr(m_ptr_) {}

GeometricObject::GeometricObject(const GeometricObject& rhs)
{ *m_ptr = (*rhs.m_ptr); }

GeometricObject::~GeometricObject(void)
{ delete m_ptr; }

RGBColor
GeometricObject::get_reflected_color(ShadeRec& sr, const Ambient* amb_ptr, const std::vector<Light*> light_ptrs) const
{ return m_ptr->shade(sr, amb_ptr, light_ptrs); }

RGBColor
GeometricObject::get_reflected_color(ShadeRec& sr, const Ambient* amb_ptr, const std::vector<Light*> light_ptrs, const std::vector<GeometricObject*> obj_ptrs) const
{
	return m_ptr->shade(sr, amb_ptr, light_ptrs, obj_ptrs);
}


/* NOTE: Implementation of Sphere */
Sphere::Sphere(): center(), radius(0.0f) {}
Sphere::Sphere(const Point3D& ct, float r): center(ct), radius(r) {}
Sphere::Sphere(const Point3D& ct, float r, const RGBColor& c):
	center(ct),
	radius(r)
{}

bool
Sphere::hit(const Ray& ray, float& tmin, ShadeRec& sr) const
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
bool
Sphere::shadow_hit(const Ray& ray, float& tmin) const
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
			return true;
		}

		t = (-b + e) / denom;
		if (t > eps) {
			tmin = t;
			return true;
		}
	}
	return false;
}

void
Sphere::set_center(float f) { center = Point3D(f); }
void
Sphere::set_center(float x, float y, float z) { center = Point3D(x, y, z); }
void
Sphere::set_radius(float r) { radius = r; }

/* NOTE: Implementation of GeometricObject */
Plane::Plane(): point(Point3D()), normal(Normal()) {}
Plane::Plane(const Point3D p, const Normal& n): point(p), normal(n) {}
Plane::Plane(const Point3D p, const Normal& n, const RGBColor c, float kd): point(p), normal(n) {
}

bool
Plane::hit(const Ray& ray, float& tmin, ShadeRec& sr) const
{
	float t = (point - ray.o) * normal / (ray.d * normal);
	if (t > eps) {
		tmin = t;
		sr.normal = normal;
		sr.local_hit_point = ray.o + ray.d * t;
		return true;
	}
	return false;
}
bool
Plane::shadow_hit(const Ray& ray, float& tmin) const
{
	float t = (point - ray.o) * normal / (ray.d * normal);
	if (t > eps) {
		tmin = t;
		return true;
	}
	return false;
}

