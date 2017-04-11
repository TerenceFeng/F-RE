/* ====================================================
#   Author        : Terence (Yongxin) Feng
#   Email         : tyxfeng@gmail.com
#   File Name     : Object.cpp
#   Last Modified : 2017-03-21 20:18
# ====================================================*/

#include "BBox.h"
#include "GeometricObject.h"
#include "ShadeRec.h"
#include "sampler.h"
#include <cmath>
#include <cfloat>
#include <algorithm>

#define INV_PI 0.31831f

inline float
clamp(float x, float min, float max)
{
	return (x < min ? min : (x > max ? max : x));
}

/* NOTE: Implementation of GeometricObject */
GeometricObject::GeometricObject(void) {}

GeometricObject::~GeometricObject(void)
{	delete material_ptr; }

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
Sphere::hit(const Ray& ray, float& tmin, ShadeRec& sr)
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
Sphere::shadow_hit(const Ray& ray, float& tmin)
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

BBox
Sphere::get_bounding_box(void)
{
	float dist = sqrtf(3 * radius + radius);
	return BBox(center.x - dist, center.y - dist, center.z - dist,
		 center.x + dist, center.y + dist, center.z + dist);
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
Plane::hit(const Ray& ray, float& tmin, ShadeRec& sr)
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
Plane::shadow_hit(const Ray& ray, float& tmin)
{
	float t = (point - ray.o) * normal / (ray.d * normal);
	if (t > eps) {
		tmin = t;
		return true;
	}
	return false;
}

BBox
Plane::get_bounding_box(void)
{
	/* not support BBox */
	return BBox();
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
Rectangle::hit(const Ray& ray, float& tmin, ShadeRec& sr)
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
Rectangle::shadow_hit(const Ray& ray, float& tmin)
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

BBox
Rectangle::get_bounding_box(void)
{
	Point3D p1 = p0 + a;
	Point3D p2 = p0 + b;
	Point3D p3 = p1 + b;
	float x0, y0, z0;
	float x1, y1, z1;

	using namespace std;
	x0 = min(min(p0.x, p1.x), min(p2.x, p3.x));
	y0 = min(min(p0.y, p1.y), min(p2.y, p3.y));
	z0 = min(min(p0.z, p1.z), min(p2.z, p3.z));
	x1 = max(max(p0.x, p1.x), max(p2.x, p3.x));
	y1 = max(max(p0.y, p1.y), max(p2.x, p3.x));
	z1 = max(max(p0.z, p1.z), max(p2.z, p3.z));
	return BBox(x0, y0, z0, x1, y1, z1);
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

/* NOTE: implementation of Triangle */
Triangle::Triangle():
	v0(0, 0, 0),
	v1(0, 0, 1),
	v2(1, 0, 0),
	normal(0, 1, 0)
{}

Triangle::Triangle(const Point3D& a, const Point3D& b, const Point3D& c):
	v0(a),
	v1(b),
	v2(c)
{
	normal = (v1 - v0) ^ (v2 - v0);
	normal.normalize();
}

bool
Triangle::hit(const Ray& ray, float& tmin, ShadeRec& sr)
{
	float a = v0.x - v1.x, b = v0.x - v2.x, c = ray.d.x, d = v0.x - ray.o.x;
	float e = v0.y - v1.y, f = v0.y - v2.y, g = ray.d.y, h = v0.y - ray.o.y;
	float i = v0.z - v1.z, j = v0.z - v2.z, k = ray.d.z, l = v0.z - ray.o.z;

	float m = f * k - g * j, n = h * k - g * l, p = f * l - h * j;
	float q = g * i - e * k, s = e * j - f * i;

	float inv_denom = 1.0 / (a * m + b * q + c * s);

	float e1 = d * m - b * n - c * p;
	float beta = e1 * inv_denom;

	if (beta < 0)
		return false;

	float r = e * l - h * i;
	float e2 = a * n + d * q + c * r;
	float gamma = e2 * inv_denom;

	if (gamma < 0)
		return false;

	if (beta + gamma > 1)
		return false;

	float e3 = a * p - b * r + d * s;
	float t = e3 * inv_denom;

	if (t < eps)
		return false;

	tmin = t;
	sr.normal = normal;
	sr.local_hit_point = ray.o + ray.d * t;
	return true;
}

bool
Triangle::shadow_hit(const Ray& ray, float& tmin)
{
	float a = v0.x - v1.x, b = v0.x - v2.x, c = ray.d.x, d = v0.x - ray.o.x;
	float e = v0.y - v1.y, f = v0.y - v2.y, g = ray.d.y, h = v0.y - ray.o.y;
	float i = v0.z - v1.z, j = v0.z - v2.z, k = ray.d.z, l = v0.z - ray.o.z;

	float m = f * k - g * j, n = h * k - g * l, p = f * l - h * j;
	float q = g * i - e * k, s = e * j - f * i;

	float inv_denom = 1.0 / (a * m + b * q + c * s);

	float e1 = d * m - b * n - c * p;
	float beta = e1 * inv_denom;

	if (beta < 0)
		return false;

	float r = e * l - h * i;
	float e2 = a * n + d * q + c * r;
	float gamma = e2 * inv_denom;

	if (gamma < 0)
		return false;

	if (beta + gamma > 1)
		return false;

	float e3 = a * p - b * r + d * s;
	float t = e3 * inv_denom;

	if (t < eps)
		return false;

	tmin = t;
	return true;
}

BBox
Triangle::get_bounding_box(void)
{
	float x0, y0, z0;
	float x1, y1, z1;
	using namespace std;
	x0 = min(v0.x, min(v1.x, v2.x));
	y0 = min(v0.y, min(v1.y, v2.y));
	z0 = min(v0.z, min(v1.z, v2.z));
	x1 = max(v0.x, max(v1.x, v2.x));
	y1 = max(v0.y, max(v1.y, v2.y));
	z1 = max(v0.z, max(v1.z, v2.z));
	return BBox(x0, y0, z0, x1, y1, z1);
}

/* NOTE: implementation of Box */
Box::Box(void):
	x0(0.0), y0(0.0), z0(0.0),
	x1(1.0), y1(1.0), z1(1.0)
{}

Box::Box(const float x0_, const float y0_, const float z0_,
		   const float x1_, const float y1_, const float z1_):
	x0(x0_), y0(y0_), z0(z0_),
	x1(x1_), y1(y1_), z1(z1_)
{}

BBox
Box::get_bounding_box(void)
{
	return BBox(x0, y0, z0, z1, y1, z1);
}

bool
Box::hit(const Ray& ray, float& tmin, ShadeRec& sr)
{
	float ox = ray.o.x, oy = ray.o.y, oz = ray.o.z;
	float dx = ray.d.x, dy = ray.d.y, dz = ray.d.z;

	float tx_min, ty_min, tz_min;
	float tx_max, ty_max, tz_max;

	float a = 1.0 / dx;
	if (a >= 0)
	{
		tx_min = (x0 - ox) * a;
		tx_max = (x1 - ox) * a;
	}
	else
	{
		tx_min = (x1 - ox) * a;
		tx_max = (x0 - ox) * a;
	}

	float b = 1.0 / dy;
	if (b >= 0)
	{
		ty_min = (y0 - y1) * b;
		ty_max = (y1 - oy) * b;
	}
	else
	{
		ty_min = (y1 - oy) * b;
		ty_max = (y0 - oy) * b;
	}

	float c = 1.0 / dz;
	if (c >= 0)
	{
		tz_min = (z0 - oz) * c;
		tz_max = (z1 - oz) * c;
	}
	else
	{
		tz_min = (z1 - oz) * c;
		tz_max = (z0 - oz) * c;
	}

	int face_in, face_out;
	float t0, t1;
	/* largest entering t value */
	if (tx_min > ty_min)
	{
		t0 = tx_min;
		face_in = (a >= 0) ? 0 : 3;
	}
	else
	{
		t0 = ty_min;
		face_in = (b >= 0) ? 1 : 4;
	}
	if (tz_min > t0)
	{
		t0 = tz_min;
		face_in = (c >= 0) ? 2 : 5;
	}
	/* smallest exiting t value */
	if (tx_max < ty_max)
	{
		t1 = tx_max;
		face_out = (a >= 0) ? 3 : 0;
	}
	else
	{
		t1 = ty_max;
		face_out = (b >= 0) ? 4 : 1;
	}
	if (tx_max < t1)
	{
		t1 = tz_max;
		face_out = (c >= 0) ? 5 : 2;
	}

	if (t0 < t1 && t1 > eps)
	{
		if (t0 > eps)
		{
			tmin = t0;
			sr.normal = get_normal(face_in);
		}
		else
		{
			tmin = t1;
			sr.normal = get_normal(face_out);
		}
		sr.local_hit_point = ray.o + ray.d * tmin;
		return true;
	}
	return false;
}

Normal
Box::get_normal(const int face) const
{
	switch (face)
	{
		case 0: return Normal(-1, 0, 0);
		case 1: return Normal(0, -1, 0);
		case 2: return Normal(0, 0, -1);
		case 3: return Normal(1, 0, 0);
		case 4: return Normal(0, 1, 0);
		case 5: return Normal(0, 0, 1);
	}
	return Normal(0, 0, 0);
}

/* NOTE: implementation of Compound */
Compound::Compound():
	object_ptrs()
{}

void
Compound::set_material(Material* m_ptr_)
{
	material_ptr = m_ptr_;
	for (GeometricObject* obj_ptr: object_ptrs)
		obj_ptr->set_material(m_ptr_);
}

void
Compound::add_object(GeometricObject* obj_ptr_)
{
	object_ptrs.push_back(obj_ptr_);
}

bool
Compound::hit(const Ray& ray, float& tmin, ShadeRec& sr)
{
	float t;
	bool hit = false;
	Normal normal;
	Point3D local_hit_point;
	tmin = FLT_MAX;

	for (GeometricObject* obj_ptr: object_ptrs)
	{
		if (obj_ptr->hit(ray, t, sr) && (t < tmin))
		{
			hit = true;
			tmin = t;
			normal = sr.normal;
			material_ptr = obj_ptr->get_material();
			local_hit_point = sr.local_hit_point;
		}
	}
	if (hit)
	{
		sr.t = tmin;
		sr.normal = normal;
		sr.local_hit_point = local_hit_point;
		return true;
	}
	return false;
}

bool
Compound::shadow_hit(const Ray& ray, float& tmin)
{
	float t;
	tmin = FLT_MAX;
	bool hit = false;
	Normal normal;
	Point3D local_hit_point;

	for (GeometricObject* obj_ptr: object_ptrs)
	{
		if (obj_ptr->shadow_hit(ray, t) && (t < tmin))
		{
			hit = true;
			tmin = t;
		}
	}
	if (hit)
	{
		return true;
	}
	return false;
}

BBox
Compound::get_bounding_box(void)
{
	BBox bbox;
	using namespace std;
	float x0 = FLT_MAX, y0 = FLT_MAX, z0 = FLT_MAX;
	float x1 = FLT_MIN, y1 = FLT_MIN, z1 = FLT_MIN;
	for (GeometricObject* obj_ptr: object_ptrs)
	{
		bbox = obj_ptr->get_bounding_box();
		if (bbox.x0 < x0) x0 = bbox.x0;
		if (bbox.y0 < y0) y0 = bbox.y0;
		if (bbox.z0 < z0) z0 = bbox.z0;
		if (bbox.x1 > x1) x1 = bbox.x1;
		if (bbox.y1 > y1) y1 = bbox.y1;
		if (bbox.z1 > z1) z1 = bbox.z1;
	}
	return BBox(x0, y0, z0, x1, y1, z1);
}

/* NOTE: implementation of Grid */
Grid::Grid(void):
	cells(),
	bbox(),
	nx(0), ny(0), nz(0)
{}

void
Grid::setup_cells(void)
{
	Point3D p0 = min_coordinate();
	Point3D p1 = max_coordinate();
	bbox.x0 = p0.x; bbox.y0 = p0.y; bbox.z0 = p0.z;
	bbox.x1 = p1.x; bbox.y1 = p1.y; bbox.z1 = p0.z;

	int num_objects = object_ptrs.size();
	float wx = p1.x - p0.x;
	float wy = p1.y - p0.y;
	float wz = p1.z - p0.z;
	const float multiplier = 2.0;
	float s = powf(wx * wy * wz / num_objects, 0.33333);
	nx = multiplier * wx / s + 1;
	ny = multiplier * wy / s + 1;
	nz = multiplier * wz / s + 1;

	int num_cells = nx * ny * nz;
	cells.reserve(num_cells);
	for (int i = 0; i < num_cells; i++)
	{
		cells.push_back(nullptr);
	}

	std::vector<int> count(num_cells, 0);

	BBox obj_bbox;
	int index;

	for (GeometricObject *obj_ptr: object_ptrs)
	{
		obj_bbox = obj_ptr->get_bounding_box();

		/* compute the cell indices for the corners of the bouding box of the object */
		int ixmin = clamp((obj_bbox.x0 - p0.x) * nx / (p1.x - p0.x), 0, nx - 1);
		int iymin = clamp((obj_bbox.y0 - p0.y) * ny / (p1.y - p0.y), 0, ny - 1);
		int izmin = clamp((obj_bbox.z0 - p0.z) * nz / (p1.z - p0.z), 0, nz - 1);
		int ixmax = clamp((obj_bbox.x1 - p0.x) * nx / (p1.x - p0.x), 0, nx - 1);
		int iymax = clamp((obj_bbox.y1 - p0.y) * ny / (p1.y - p0.y), 0, ny - 1);
		int izmax = clamp((obj_bbox.z1 - p0.z) * nz / (p1.z - p0.z), 0, nz - 1);

		/* add objects to cells */
		for (int iz = izmin; iz <= izmax; iz++)
			for (int iy = iymin; iy <= iymax; iy++)
				for (int ix = ixmin; ix <= ixmax; ix++)
				{
					index = ix + nx * iy + nx * ny * iz;

					if (count[index] == 0)
					{
						cells[index] = obj_ptr;
						count[index] += 1;
					}
					else
					{
						if (count[index] == 1)
						{
							Compound *compound_ptr = new Compound;
							compound_ptr->add_object(cells[index]);
							compound_ptr->add_object(obj_ptr);
							cells[index] = compound_ptr;
							count[index] += 1;
						}
						else
						{
							((Compound *)cells[index])->add_object(obj_ptr);
							count[index] += 1;
						}
					}
				}
	}
	object_ptrs.erase(object_ptrs.begin(), object_ptrs.end());
	count.erase(count.begin(), count.end());
}

Point3D
Grid::min_coordinate(void)
{
	BBox bbox;
	Point3D p0(FLT_MAX);

	for (GeometricObject *obj_ptr: object_ptrs)
	{
		bbox = obj_ptr->get_bounding_box();
		if (bbox.x0 < p0.x) p0.x = bbox.x0;
		if (bbox.y0 < p0.y) p0.y = bbox.y0;
		if (bbox.z0 < p0.z) p0.z = bbox.z0;
	}
	p0.x -= eps; p0.y -= eps; p0.z -= eps;
	return p0;
}

Point3D
Grid::max_coordinate(void)
{
	BBox bbox;
	Point3D p1(FLT_MAX);
	for (GeometricObject *obj_ptr: object_ptrs)
	{
		bbox = obj_ptr->get_bounding_box();
		if (bbox.x1 > p1.x) p1.x = bbox.x1;
		if (bbox.y1 > p1.y) p1.y = bbox.y1;
		if (bbox.z1 > p1.z) p1.z = bbox.z1;
	}
	p1.x += eps; p1.y += eps; p1.z += eps;
	return p1;
}

/* NOTE: implementation of TrianglarPyramid */
TrianglarPyramid::TrianglarPyramid()
{
	object_ptrs.push_back(new Triangle(Point3D(0, 0, 1), Point3D(0, 0, 0), Point3D(0, 1, 0)));
	object_ptrs.push_back(new Triangle(Point3D(0, 0, 1), Point3D(0, 1, 0), Point3D(1, 0, 0)));
	object_ptrs.push_back(new Triangle(Point3D(0, 0, 1), Point3D(0, 1, 0), Point3D(0, 0, 0)));
	object_ptrs.push_back(new Triangle(Point3D(0, 0, 0), Point3D(0, 1, 0), Point3D(1, 0, 0)));
}

TrianglarPyramid::TrianglarPyramid(const Point3D& a, const Point3D& b, const Point3D& c, const Point3D& d)
{
	object_ptrs.push_back(new Triangle(a, c, b));
	object_ptrs.push_back(new Triangle(a, b, d));
	object_ptrs.push_back(new Triangle(a, d, c));
	object_ptrs.push_back(new Triangle(d, b, c));
}

