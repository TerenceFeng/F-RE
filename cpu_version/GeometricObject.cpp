/* ====================================================
#   Author        : Terence (Yongxin) Feng
#   Email         : tyxfeng@gmail.com
#   File Name     : Object.cpp
#   Last Modified : 2017-03-21 20:18
# ====================================================*/

#include "GeometricObject.h"
#include "ShadeRec.h"
#include "sampler.h"
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
	material_ptr(m_ptr_) {}

GeometricObject::GeometricObject(const GeometricObject& rhs)
{ *material_ptr = (*rhs.material_ptr); }

GeometricObject::~GeometricObject(void)
{	delete material_ptr; }

RGBColor
GeometricObject::get_reflected_color(ShadeRec& sr) const
{
	return material_ptr->area_light_shade(sr) /* + material_ptr->shade(sr) */;
}

Point3D
GeometricObject::sample(void)
{	return Point3D();}
float
GeometricObject::pdf(ShadeRec&)
{	return 1; }
Normal
GeometricObject::get_normal(const Point3D&)
{	return Normal(); }

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

/* NOTE: Implementation of Plane */
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

/* NOTE: implementation of Rectangle */
Rectangle::Rectangle():
	GeometricObject()
{}
Rectangle::Rectangle(const Point3D& p0_, const Vector3D& a_, const Vector3D& b_):
	GeometricObject(),
	p0(p0_),
	a(a_),
	b(b_)
{
	inv_area = 1 / (a.length() * b.length());
	normal = a ^ b;
	normal.normalize();
	a_len = a.length();
	b_len = b.length();
	a_len_2 = a_len * a_len;
	b_len_2 = b_len * b_len;
}

bool
Rectangle::hit(const Ray& ray, float& tmin, ShadeRec& sr) const
{
	float t = (p0 - ray.o) * normal / (ray.d * normal);
	if (t <= eps)
		return false;

	Point3D p = ray.o + ray.d * t;
	Vector3D d = p - p0;

	float ddota = d * a;
	if (ddota < 0.0 || ddota > a_len_2)
		return false;

	float ddotb = d * b;
	if (ddotb < 0.0 || ddotb > b_len_2)
		return false;

	tmin = t;
	sr.normal = normal;
	sr.local_hit_point = p;

	return true;
}

bool
Rectangle::shadow_hit(const Ray& ray, float& tmin) const
{
	float t = (p0 - ray.o) * normal / (ray.d * normal);
	if (t <= eps)
		return false;

	Point3D p = ray.o + ray.d * t;
	Vector3D d = p - p0;

	float ddota = d * a;
	if (ddota < 0.0 || ddota > a_len_2)
		return false;

	float ddotb = d * b;
	if (ddotb < 0.0 || ddotb > b_len_2)
		return false;

	tmin = t;

	return true;
}

void
Rectangle::set_sampler(Sampler *sampler_ptr_)
{
	sampler_ptr = sampler_ptr_;
}

Point3D
Rectangle::sample(void)
{
	Point2D sample_point = sampler_ptr->sample_unit_square();
	return (p0 + a * sample_point.x + b * sample_point.y);
}

Normal
Rectangle::get_normal(const Point3D& p)
{
	return normal;
}

float
Rectangle::pdf(ShadeRec& sr)
{
	return inv_area;
}
