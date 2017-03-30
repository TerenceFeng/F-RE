/* ====================================================
#   Copyright (C)2017 All rights reserved.
#
#   Author        : Terence (Yongxin) Feng
#   Email         : tyxfeng@gmail.com
#   File Name     : Object.cpp
#   Last Modified : 2017-03-21 20:18
#   Describe      :
#   Log           :
# ====================================================*/

#include "GeometricObject.h"
#include "ShadeRec.h"

#define INV_PI 0.31831f

/* Implementation of GeometricObject */
GeometricObject&
GeometricObject::operator = (const GeometricObject& rhs)
{
	return (*this);
}

GeometricObject::GeometricObject(void) {}
GeometricObject::GeometricObject(Material *m_ptr_):
	m_ptr(m_ptr_)
{}

GeometricObject::GeometricObject(const GeometricObject& rhs)
{
	*m_ptr = (*rhs.m_ptr);
}

GeometricObject::~GeometricObject(void)
{
	delete m_ptr;
}

RGBColor
GeometricObject::get_reflected_color(ShadeRec& sr, const Ambient* amb_ptr, const std::vector<Light*> light_ptrs) const
{
	return m_ptr->shade(sr, amb_ptr, light_ptrs);
}

