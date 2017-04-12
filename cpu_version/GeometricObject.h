/* ====================================================
#   Author        : Terence (Yongxin) Feng
#   Email         : tyxfeng@gmail.com
#   File Name     : Object.h
# ====================================================*/

#ifndef  _GEOMETRICOBJECT_H
#define  _GEOMETRICOBJECT_H

#include <vector>
#include "BBox.h"
#include "RGBColor.h"
#include "Utilities.h"
#include "ShadeRec.h"
#include "Material.h"

class Sampler;

class GeometricObject
{
protected:
	const float eps = 1e-4;

public:
	Material *material_ptr;

	GeometricObject(void);
	virtual ~GeometricObject(void);

	inline void set_material(Material *m_ptr_) {material_ptr= m_ptr_;}
	inline Material* get_material() { return material_ptr; }

	virtual bool hit(const Ray& r, float& tmin, ShadeRec& sr) = 0;
	virtual bool shadow_hit(const Ray& ray, float& tmin) = 0;

	virtual BBox get_bounding_box(void) = 0;

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

	bool hit(const Ray& ray, float& tmin, ShadeRec& sr);
	bool shadow_hit(const Ray& ray, float& tmin);
	void set_center(float f);
	void set_center(float x, float y, float z);
	void set_radius(float r);
	virtual BBox get_bounding_box(void);

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
	bool hit(const Ray& ray, float& tmin, ShadeRec& sr);
	bool shadow_hit(const Ray& ray, float& tmin);
	virtual BBox get_bounding_box(void);

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
	bool hit(const Ray&, float&, ShadeRec&);
	bool shadow_hit(const Ray&, float&);
	virtual BBox get_bounding_box(void);

private:
	Point3D p0;
	Vector3D a, b;
	float a_len, b_len;
	float a_len_2, b_len_2;
	Normal normal;
	Sampler *sampler_ptr;
	float inv_area;
};

class Triangle: public GeometricObject
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

class Compound: public GeometricObject
{
public:
	Compound(void);
	virtual void set_material(Material* material_ptr_);
	void add_object(GeometricObject* obj_ptr_);
	virtual bool hit(const Ray&, float&, ShadeRec&);
	virtual bool shadow_hit(const Ray&, float&);

	virtual BBox get_bounding_box(void);
protected:
	std::vector<GeometricObject*> object_ptrs;
};

class Grid: public Compound
{
public:
	Grid(void);
	~Grid(void);

	virtual BBox get_bounding_box(void);
	void setup_cells(void);
	virtual bool hit(const Ray&, float&, ShadeRec&);
	virtual bool shadow_hit(const Ray&, float&);

private:
	std::vector<GeometricObject*> cells;
	BBox bbox;
	int nx, ny, nz;

	Point3D min_coordinate(void);
	Point3D max_coordinate(void);
};

#endif
