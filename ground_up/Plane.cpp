
#include "Plane.h"


/* Implementation of GeometricObject */
Plane::Plane(): point(Point3D()), normal(Normal()) {}
Plane::Plane(const Point3D p, const Normal& n): point(p), normal(n) {}
Plane::Plane(const Point3D p, const Normal& n, const RGBColor c, float kd): point(p), normal(n) {
	// reflec_coef = kd;
}

bool
Plane:: hit(const Ray& ray, float& tmin, ShadeRec& sr) const
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

