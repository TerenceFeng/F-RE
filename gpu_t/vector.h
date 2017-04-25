#pragma once
#include <cmath>

class Vector2D
{
public:
	float x, y;

	Vector2D(): x(0), y(0) {}
	Vector2D(float x_, float y_): x(x_), y(y_) {}
	Vector2D(const Vector2D& v)
	{
		x = v.x;
		y = v.y;
	}

	Vector2D& operator = (const Vector2D& rhs)
	{
		x = rhs.x;
		y = rhs.y;
		return (*this);
	}

	Vector2D operator * (const float a) const
	{
		return Vector2D(x * a, y * a);
	}
};

class Vector3D
{
public:
	float x, y, z;

	Vector3D(): x(0), y(0), z(0) {}
	Vector3D(float x_, float y_, float z_): x(x_), y(y_), z(z_) {}
	Vector3D(const Vector3D& v)
	{
		x = v.x;
		y = v.y;
		z = v.z;
	}

	Vector3D& operator = (const Vector3D& rhs)
	{
		x = rhs.x;
		y = rhs.y;
		z = rhs.z;
		return (*this);
	}

	Vector3D operator * (const float a) const
	{
		return Vector3D(x * a, y * a, z * a);
	}

	Vector3D operator / (const float a) const
	{
		return Vector3D(x / a, y / a, z / a);
	}

	Vector3D operator + (const Vector3D& v) const
	{
		return Vector3D(x + v.x, y + v.y, z + v.z);
	}

	Vector3D& operator += (const Vector3D& v)
	{
		this->x += v.x;
		this->y += v.y;
		this->z += v.z;
		return (*this);
	}

	Vector3D operator - (const Vector3D& v) const
	{
		return Vector3D(x - v.x, y - v.y, z - v.z);
	}

	float operator * (const Vector3D& b) const
	{
		return x * b.x + y * b.y + z * b.z;
	}

	Vector3D operator ^ (const Vector3D& v) const
	{
		return Vector3D(y * v.z - z * v.y, z * v.x - x * v.z, x * v.y - y * v.x);
	}

	Vector3D operator-() const
	{
		return Vector3D(-x, -y, -z);
	}

	bool operator == (const Vector3D& v) const
	{
		return x == v.x && y == v.y && z == v.z;
	}

	float length()
	{
		return sqrtf(this->len_squared());
	}

	float len_squared() {
		return x * x + y * y + z * z;
	}

	float distance(const Vector3D& v) const
	{
		return sqrtf(distance_sqr(v));
	}

	float distance_sqr(const Vector3D& v) const
	{
		return (x - v.x) * (x - v.x) + (y - v.y) * (y - v.y) + (z - v.z) * (z - v.z);
	}

	void normalize()
	{
		float len = length();
		x /= len; y /= len; z /= len;
	}
};

typedef Vector2D Point2D;
typedef Vector3D Point3D;
typedef Vector3D Normal;
