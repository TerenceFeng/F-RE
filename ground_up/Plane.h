
#ifndef _PLANE_H
#define _PLANE_H

#include "GeometricObject.h"

class Plane: public GeometricObject
{
public:
	Plane();
	Plane(const Point3D p, const Normal& n);
	Plane(const Point3D p, const Normal& n, const RGBColor c, float kd);
	bool
	hit(const Ray& ray, float& tmin, ShadeRec& sr) const;

private:
	Point3D point;
	Normal normal;
	const float eps = 1e-4;
};

#endif
