/* ====================================================
#   Author        : Terence (Yongxin) Feng
#   Email         : tyxfeng@gmail.com
#   File Name     : Object.h
# ====================================================*/

#ifndef _OBJECT_H
#define _OBJECT_H

#include <vector>
#include "BBox.h"
#include "RGBColor.h"
#include "Utilities.h"
#include "ShadeRec.h"
#include "Material.h"

class Sampler;

class Object
{
protected:
	const float eps = 1e-4;

public:
	Material *material_ptr;
	Sampler *sampler_ptr;

	Object(void);
	virtual ~Object(void);

	inline void set_material(Material *m_ptr_) { material_ptr = m_ptr_; }
	inline void set_sampler(Sampler *s_ptr_) {sampler_ptr = s_ptr_; }

	virtual bool hit(const Ray& r, float& tmin, ShadeRec& sr) = 0;
	virtual bool shadow_hit(const Ray& ray, float& tmin) = 0;

	virtual BBox get_bounding_box(void) = 0;

	virtual Point3D sample(void);
	virtual float pdf(ShadeRec&);
	virtual Normal get_normal(const Point3D&);

};

class Sphere: public Object
{
public:

	Sphere();
	/* center radius color */
	Sphere(const Point3D& ct, const float r, Material* m);

	bool hit(const Ray& ray, float& tmin, ShadeRec& sr);
	bool shadow_hit(const Ray& ray, float& tmin);
	void set_center(float x, float y, float z);
	void set_radius(float r);
	virtual BBox get_bounding_box(void);

private:
	Point3D center;
	float radius;
};

class Plane: public Object
{
public:
	Plane();
	Plane(const Point3D p, const Normal& n);
	Plane(const Point3D p, const Normal& n, const RGBColor c, float kd);
	bool hit(const Ray& ray, float& tmin, ShadeRec& sr);
	bool shadow_hit(const Ray& ray, float& tmin);
	BBox get_bounding_box(void);

private:
	Point3D point;
	Normal normal;
};

class Rectangle: public Object
{
public:
	Rectangle();
	Rectangle(const Point3D&, const Vector3D&, const Vector3D&);
	Point3D sample(void);
	float pdf(ShadeRec& sr);
	Normal get_normal(const Point3D&);
	bool hit(const Ray&, float&, ShadeRec&);
	bool shadow_hit(const Ray&, float&);
	virtual BBox get_bounding_box(void);

private:
	Point3D p0;
	Vector3D a, b;
	float a_len, b_len;
	float a_len_2, b_len_2;
	Normal normal;
	float inv_area;
};

class Triangle: public Object
{
public:
	Point3D v0, v1, v2;
	Normal normal;
public:
	Triangle(void);
	Triangle(const Point3D&, const Point3D&, const Point3D&);
	virtual bool hit(const Ray&, float&, ShadeRec&);
	virtual bool shadow_hit(const Ray&, float&);
	virtual BBox get_bounding_box(void);
};

class Compound: public Object
{
public:
	Compound(void);
	virtual void set_material(Material* material_ptr_);
	void add_object(Object* obj_ptr_);
	bool hit(const Ray&, float&, ShadeRec&);
	bool shadow_hit(const Ray&, float&);

	BBox get_bounding_box(void);
protected:
	std::vector<Object*> object_ptrs;
};
#endif
