/* ====================================================
#   Author        : Terence (Yongxin) Feng
#   Email         : tyxfeng@gmail.com
#   File Name     : Object.h
# ====================================================*/

#ifndef  _GEOMETRICOBJECT_H
#define  _GEOMETRICOBJECT_H

#include <vector>
#include "RGBColor.h"
#include "Utilities.h"
#include "ShadeRec.h"
#include "Material.h"

class Sampler;

class GeometricObject
{
protected:
	const float eps = 1e-4;
	GeometricObject& operator = (const GeometricObject& rhs);

public:
	Material *material_ptr;

	GeometricObject(void);
	GeometricObject(Material *m_ptr_);
	GeometricObject(const GeometricObject& go);
	virtual ~GeometricObject(void);

	virtual bool hit(const Ray& r, float& tmin, ShadeRec& sr) const = 0;

	inline void set_material(Material *m_ptr_) {material_ptr= m_ptr_;}

	RGBColor get_reflected_color(ShadeRec&) const;
	virtual bool shadow_hit(const Ray& ray, float& tmin) const = 0;

	virtual Point3D sample(void);
	virtual float pdf(ShadeRec&);
	virtual Normal get_normal(const Point3D&);

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

class Rectangle: public GeometricObject
{
public:
	Rectangle();
	Rectangle(const Point3D&, const Vector3D&, const Vector3D&);
	void set_sampler(Sampler *);
	virtual Point3D sample(void);
	virtual float pdf(ShadeRec& sr);
	virtual Normal get_normal(const Point3D&);
	bool hit(const Ray&, float&, ShadeRec&) const;
	bool shadow_hit(const Ray&, float&) const;
private:
	Point3D p0;
	Vector3D a, b;
	float a_len, b_len;
	float a_len_2, b_len_2;
	Normal normal;
	Sampler *sampler_ptr;
	float inv_area;
};

#endif
