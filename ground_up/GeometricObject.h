
/* ====================================================
#   Copyright (C)2017 All rights reserved.
#
#   Author        : Terence (Yongxin) Feng
#   Email         : tyxfeng@gmail.com
#   File Name     : Object.h
#   Last Modified : 2017-03-21 20:18
#   Describe      :
#
#   Log           :
#
# ====================================================*/

#ifndef  _GEOMETRICOBJECT_H
#define  _GEOMETRICOBJECT_H

#include <vector>
#include "RGBColor.h"
#include "Utilities.h"
#include "ShadeRec.h"
#include "Material.h"

class GeometricObject
{
protected:
	Material *m_ptr;
	const float eps = 1e-4;
	/* reflection coefficient */
	GeometricObject& operator= (const GeometricObject& rhs);

public:
	GeometricObject(void);
	GeometricObject(Material *m_ptr_);
	GeometricObject(const GeometricObject& go);
	virtual ~GeometricObject(void);

	virtual bool
	hit(const Ray& r, float& tmin, ShadeRec& sr) const = 0;

	/*
	 * inline void
	 *     set_color(const RGBColor& c) {
	 *         color = c;
	 *     }
	 * inline RGBColor
	 *     get_color() const { return ma; }
	 */

	inline void
		set_material(Material *m_ptr_) {m_ptr = m_ptr_;}
	inline Material *
		get_material() {return m_ptr;}

	RGBColor
		get_reflected_color(ShadeRec& sr, const Ambient* amb_ptr, const std::vector<Light*> light_ptrs) const;

};
#endif


