/* ====================================================
#   Author        : Terence (Yongxin) Feng
#   Email         : tyxfeng@gmail.com
#   File Name     : Utilities.cpp
#   Last Modified : 2017-03-21 17:23
# ====================================================*/

#include "Utilities.h"
#include <iostream>
using namespace std;

/* Implementation of Vector3D */
Vector3D::Vector3D(): x(0.0), y(0.0), z(0.0) {}
Vector3D::Vector3D(float a): x(a), y(a), z(a) {}
Vector3D::Vector3D(float _x, float _y, float _z): x(_x), y(_y), z(_z) {}
Vector3D::Vector3D(const Vector3D& v): x(v.x), y(v.y), z(v.z) {}

Vector3D&
Vector3D:: operator= (const Vector3D& rhs) {
	x = rhs.x;
	y = rhs.y;
	z = rhs.z;
	return (*this);
}

void
Vector3D::normalize() {
	float len = this->length();
	x /= len; y /= len; z /= len;
}
Vector3D&
Vector3D::hat() {
	this->normalize();
	return (*this);
}


/* Implementation of Ray */
Ray:: Ray(): d(Vector3D()), o(Point3D()) {}
Ray:: Ray(const Ray& r): d(r.d), o(r.d) {}
Ray:: Ray(const Point3D& _o, const Point3D& _d): o(_o), d(_d) {}

