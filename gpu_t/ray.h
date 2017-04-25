#pragma once

class Ray
{
public:
	Vector3D d;
	Point3D o;

	Ray(): d(), o() {}
	Ray(const Ray& r): d(r.d), o(r.o) {}
	Ray(const Point3D& o_, const Point3D& d_): d(d_), o(o_) {}

	Ray& operator = (const Ray& rhs) {
		o = rhs.o; d = rhs.d;
		return (*this);
	}

};

