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
	float dist = sqrtf(3 * radius * radius);
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

Grid::~Grid(void)
{}

BBox
Grid::get_bounding_box(void)
{
	return bbox;
}
bool
Grid::hit(const Ray& ray, float& t, ShadeRec& sr)
{
	float ox = ray.o.x;
	float oy = ray.o.y;
	float oz = ray.o.z;
	float dx = ray.d.x;
	float dy = ray.d.y;
	float dz = ray.d.z;

	float x0 = bbox.x0;
	float y0 = bbox.y0;
	float z0 = bbox.z0;
	float x1 = bbox.x1;
	float y1 = bbox.y1;
	float z1 = bbox.z1;

	float tx_min, ty_min, tz_min;
	float tx_max, ty_max, tz_max;

	/* the following code includes modifications from Shirley and Morley (2003) */

	float a = 1.0 / dx;
	if (a >= 0) {
		tx_min = (x0 - ox) * a;
		tx_max = (x1 - ox) * a;
	}
	else {
		tx_min = (x1 - ox) * a;
		tx_max = (x0 - ox) * a;
	}

	float b = 1.0 / dy;
	if (b >= 0) {
		ty_min = (y0 - oy) * b;
		ty_max = (y1 - oy) * b;
	}
	else {
		ty_min = (y1 - oy) * b;
		ty_max = (y0 - oy) * b;
	}

	float c = 1.0 / dz;
	if (c >= 0) {
		tz_min = (z0 - oz) * c;
		tz_max = (z1 - oz) * c;
	}
	else {
		tz_min = (z1 - oz) * c;
		tz_max = (z0 - oz) * c;
	}

	float t0, t1;

	if (tx_min > ty_min)
		t0 = tx_min;
	else
		t0 = ty_min;

	if (tz_min > t0)
		t0 = tz_min;

	if (tx_max < ty_max)
		t1 = tx_max;
	else
		t1 = ty_max;

	if (tz_max < t1)
		t1 = tz_max;

	if (t0 > t1)
		return(false);

	/* initial cell coordinates */
	int ix, iy, iz;

	if (bbox.inside(ray.o)) {
		ix = clamp((ox - x0) * nx / (x1 - x0), 0, nx - 1);
		iy = clamp((oy - y0) * ny / (y1 - y0), 0, ny - 1);
		iz = clamp((oz - z0) * nz / (z1 - z0), 0, nz - 1);
	}
	else {
		Point3D p = ray.o + ray.d * t0;
		ix = clamp((p.x - x0) * nx / (x1 - x0), 0, nx - 1);
		iy = clamp((p.y - y0) * ny / (y1 - y0), 0, ny - 1);
		iz = clamp((p.z - z0) * nz / (z1 - z0), 0, nz - 1);
	}

	/* ray parameter increments per cell in the x, y, and z directions */
	float dtx = (tx_max - tx_min) / nx;
	float dty = (ty_max - ty_min) / ny;
	float dtz = (tz_max - tz_min) / nz;

	float 	tx_next, ty_next, tz_next;
	int 	ix_step, iy_step, iz_step;
	int 	ix_stop, iy_stop, iz_stop;

	if (dx > 0) {
		tx_next = tx_min + (ix + 1) * dtx;
		ix_step = +1;
		ix_stop = nx;
	}
	else {
		tx_next = tx_min + (nx - ix) * dtx;
		ix_step = -1;
		ix_stop = -1;
	}

	if (dx == 0.0) {
		tx_next = FLT_MAX;
		ix_step = -1;
		ix_stop = -1;
	}


	if (dy > 0) {
		ty_next = ty_min + (iy + 1) * dty;
		iy_step = +1;
		iy_stop = ny;
	}
	else {
		ty_next = ty_min + (ny - iy) * dty;
		iy_step = -1;
		iy_stop = -1;
	}

	if (dy == 0.0) {
		ty_next = FLT_MAX;
		iy_step = -1;
		iy_stop = -1;
	}

	if (dz > 0) {
		tz_next = tz_min + (iz + 1) * dtz;
		iz_step = +1;
		iz_stop = nz;
	}
	else {
		tz_next = tz_min + (nz - iz) * dtz;
		iz_step = -1;
		iz_stop = -1;
	}

	if (dz == 0.0) {
		tz_next = FLT_MAX;
		iz_step = -1;
		iz_stop = -1;
	}

	/* traverse the grid */
	while (true) {
		GeometricObject* object_ptr = cells[ix + nx * iy + nx * ny * iz];

		if (tx_next < ty_next && tx_next < tz_next) {
			if (object_ptr && object_ptr->hit(ray, t, sr) && t < tx_next) {
				material_ptr = object_ptr->get_material();
				return (true);
			}
			tx_next += dtx;
			ix += ix_step;
			if (ix == ix_stop)
				return (false);
		}
		else {
			if (ty_next < tz_next) {
				if (object_ptr && object_ptr->hit(ray, t, sr) && t < ty_next) {
					material_ptr = object_ptr->get_material();
					return (true);
				}
				ty_next += dty;
				iy += iy_step;
				if (iy == iy_stop)
					return (false);
		 	}
		 	else {
				if (object_ptr && object_ptr->hit(ray, t, sr) && t < tz_next) {
					material_ptr = object_ptr->get_material();
					return (true);
				}
				tz_next += dtz;
				iz += iz_step;
				if (iz == iz_stop)
					return (false);
		 	}
		}
	}
}


bool
Grid::shadow_hit(const Ray& ray, float& t)
{
	ShadeRec sr;
	return hit(ray, t, sr);
}

void
Grid::setup_cells(void)
{
	Point3D p0 = min_coordinate();
	Point3D p1 = max_coordinate();
	bbox.x0 = p0.x; bbox.y0 = p0.y; bbox.z0 = p0.z;
	bbox.x1 = p1.x; bbox.y1 = p1.y; bbox.z1 = p1.z;

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
	count.erase(count.begin(), count.end());
}

Point3D
Grid::min_coordinate(void)
{
	BBox obj_bbox;
	Point3D p0(FLT_MAX);

	for (GeometricObject *obj_ptr: object_ptrs)
	{
		obj_bbox = obj_ptr->get_bounding_box();
		if (obj_bbox.x0 < p0.x) p0.x = obj_bbox.x0;
		if (obj_bbox.y0 < p0.y) p0.y = obj_bbox.y0;
		if (obj_bbox.z0 < p0.z) p0.z = obj_bbox.z0;
	}
	p0.x -= eps; p0.y -= eps; p0.z -= eps;
	bbox.x0 = p0.x; bbox.y0 = p0.y; bbox.z0 = p0.z;
	return p0;
}

Point3D
Grid::max_coordinate(void)
{
	BBox obj_bbox;
	Point3D p1(FLT_MIN);
	for (GeometricObject *obj_ptr: object_ptrs)
	{
		obj_bbox = obj_ptr->get_bounding_box();
		if (obj_bbox.x1 > p1.x) p1.x = obj_bbox.x1;
		if (obj_bbox.y1 > p1.y) p1.y = obj_bbox.y1;
		if (obj_bbox.z1 > p1.z) p1.z = obj_bbox.z1;
	}
	p1.x += eps; p1.y += eps; p1.z += eps;
	bbox.x1 = p1.x; bbox.y1 = p1.y; bbox.z1 = p1.z;
	return p1;
}

