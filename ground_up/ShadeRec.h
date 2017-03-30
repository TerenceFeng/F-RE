
/* ====================================================
#   Copyright (C)2017 All rights reserved.
#
#   Author        : Terence (Yongxin) Feng
#   Email         : tyxfeng@gmail.com
#   File Name     : ShadeRec.h
#   Last Modified : 2017-03-21 20:14
#   Describe      :
#
#   Log           :
#
# ====================================================*/

#ifndef  _SHADEREC_H
#define  _SHADEREC_H

#include "RGBColor.h"
#include "Utilities.h"
// class World;

class ShadeRec
{
public:
	bool		hit_an_object;
	Point3D		hit_point;
	Point3D		local_hit_point;
	Normal		normal;
	RGBColor	color;
	Ray			ray;
	int			depth;
	Vector3D	dir;
	float t;
	// World& w;

	ShadeRec();
	// ShadeRec(World& wr);
	ShadeRec(const ShadeRec& sr);

	ShadeRec&
	operator= (const ShadeRec& rhs);
};

#endif // _SHADEREC_H


