/* ====================================================
#   Copyright (C)2017 All rights reserved.
#   Author        : Terence (Yongxin) Feng
#   Email         : tyxfeng@gmail.com
#   File Name     : BBox.h
# ====================================================*/

#ifndef _BBOX_H
#define _BBOX_H

#include "Utilities.h"
class Ray;

class BBox
{
public:
	BBox(void);
	BBox(const float, const float, const float, const float, const float, const float);
	BBox(const BBox&);
	BBox& operator = (const BBox&);
	bool hit(const Ray&, float&) const;
	bool inside(const Point3D&) const;
public:
	float x0, y0, z0;
	float x1, y1, z1;
private:
	const float eps = 1e-4;
};

#endif
