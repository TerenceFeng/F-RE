
/* ====================================================
#   Copyright (C)2017 All rights reserved.
#
#   Author        : Terence (Yongxin) Feng
#   Email         : tyxfeng@gmail.com
#   File Name     : ShadeRec.cpp
#   Last Modified : 2017-03-21 20:15
#   Describe      :
#
#   Log           :
#
# ====================================================*/

#include "ShadeRec.h"
#include <iostream>
using namespace std;

ShadeRec::ShadeRec():
	hit_an_object(false),
	hit_point(),
	local_hit_point(),
	normal(),
	color(),
	ray(),
	depth(0),
	dir()
{}


/*
 * [> Implementation of ShadeRec <]
 * ShadeRec:: ShadeRec(World& wr):
 *     hit_an_object(false),
 *     local_hit_point(),
 *     normal(),
 *     color()
 *     // w(wr)
 * {}
 */

ShadeRec:: ShadeRec(const ShadeRec& sr):
	hit_an_object(sr.hit_an_object),
	hit_point(sr.hit_point),
	local_hit_point(sr.local_hit_point),
	normal(sr.normal),
	color(sr.color),
	ray(sr.ray),
	depth(sr.depth),
	dir(sr.dir)
	// w(sr.w)
{}

ShadeRec&
ShadeRec:: operator= (const ShadeRec& rhs) {
	hit_an_object = rhs.hit_an_object;
	hit_point = rhs.hit_point;
	local_hit_point = rhs.local_hit_point;
	normal = rhs.normal;
	color = rhs.color;
	ray = rhs.ray;
	depth = rhs.depth;
	dir = rhs.dir;
	return (*this);
}


