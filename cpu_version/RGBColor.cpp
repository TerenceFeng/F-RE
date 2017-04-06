/* ====================================================
#   Copyright (C)2017 All rights reserved.
#
#   Author        : Terence (Yongxin) Feng
#   Email         : tyxfeng@gmail.com
#   File Name     : RGBColor.cpp
#   Last Modified : 2017-03-24 16:59
# ====================================================*/

#include "RGBColor.h"

/* Implementation of RGBColor */
RGBColor:: RGBColor(): r(0.0), g(0.0), b(0.0) {}
RGBColor:: RGBColor(float c): r(c), g(c), b(c) {}
RGBColor:: RGBColor(float _r, float _g, float _b): r(_r), g(_g), b(_b) {}
RGBColor:: RGBColor(const RGBColor& c): r(c.r), g(c.g), b(c.b) {}

RGBColor&
RGBColor::operator = (const RGBColor& rhs)
{
	r = rhs.r; g = rhs.g; b = rhs.b;
	return (*this);
}
