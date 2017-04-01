
/* ====================================================
#   Copyright (C)2017 All rights reserved.
#
#   Author        : Terence (Yongxin) Feng
#   Email         : tyxfeng@gmail.com
#   File Name     : Utilities.h
#   Last Modified : 2017-03-21 17:14
#   Describe      :
#
#   Log           :
#
# ====================================================*/

#ifndef  _UTILITIES_H
#define  _UTILITIES_H

#include <cmath>

class Vector3D
{
public:
	float x, y, z;

	Vector3D();
	Vector3D(float a);
	Vector3D(float _x, float _y, float _z);
	Vector3D(const Vector3D& v);

	Vector3D& operator= (const Vector3D& rhs);

	inline Vector3D operator* (const float a) const {
		return Vector3D(x * a, y * a, z * a);
	}
	inline Vector3D operator/ (const float a) const {
		return Vector3D(x / a, y / a, z / a);
	}
	inline Vector3D operator+ (const Vector3D& v) const {
		return Vector3D(x + v.x, y + v.y, z + v.z);
	}
	inline Vector3D& operator+= (const Vector3D& v) {
		this->x += v.x;
		this->y += v.y;
		this->z += v.z;
		return (*this);
	}
	inline Vector3D operator- (const Vector3D& v) const {
		return Vector3D(x - v.x, y - v.y, z - v.z);
	}
	inline float operator* (const Vector3D& b) const {
		return x * b.x + y * b.y + z * b.z;
	}
	inline Vector3D operator^ (const Vector3D& v) const {
		return Vector3D(y * v.z - z * v.y, z * v.x - x * v.z, x * v.y - y * v.x);
	}
	inline Vector3D operator-() const {
		return Vector3D(-x, -y, -z);
	}
	inline float length() {
		return sqrtf(this->len_squared());
	}
	inline float len_squared() {
		return x * x + y * y + z * z;
	}
	inline float distance(const Vector3D& v) const {
		return (x - v.x) * (x - v.x) + (y - v.y) * (y - v.y) + (z - v.z) * (z - v.z);
	}
	void normalize();
	Vector3D& hat();
};

typedef Vector3D Point3D;
typedef Vector3D Normal;

class Ray
{
public:
	Vector3D d;
	Point3D o;

	Ray();
	Ray(const Ray& r);
	Ray(const Point3D& _o, const Point3D& _d);

	inline Ray& operator= (const Ray& rhs) {
		o = rhs.o; d = rhs.d;
		return (*this);
	}

};

class ViewPlane
{
public:
	int hres; /* horizontal resolution */
	int vres; /* vertical resolution */
	float s; /* size of pixel */
	float gamma;
	float inv_gamma;
};

#endif // _UTILITIES_H


