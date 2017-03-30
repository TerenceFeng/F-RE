
/* ====================================================
#   Copyright (C)2017 All rights reserved.
#
#   Author        : Terence (Yongxin) Feng
#   Email         : tyxfeng@gmail.com
#   File Name     : World.h
#   Last Modified : 2017-03-21 17:08
#   Describe      :
#
#   Log           :
#
# ====================================================*/

#ifndef  _WORLD_H
#define  _WORLD_H

#include "Light.h"
#include "RGBColor.h"
#include "Utilities.h"
#include "GeometricObject.h"

#include <vector>

class World
{
public:
	ViewPlane vp;
	RGBColor background_color;
	Ambient* ambient_ptr;
	std::vector<GeometricObject*> objects;
	std::vector<Light *> light_ptrs;
	// Tracer tracer

	World(void);
	~World(void);
	// void render_scene(ShadeRec&) const;

	void render_scene() const;

	inline void set_hres(int hres_) {vp.hres = hres_;}
	inline void set_vres(int vres_) {vp.vres = vres_;}
	inline void set_ambient_light(Ambient *ambient_ptr_) {
		ambient_ptr = ambient_ptr_;
	}

	void
		add_object(GeometricObject* obj);
	void
		add_light(Light* light_ptr);
	ShadeRec
		hit_bare_bones_object(const Ray& ray) const;

	void display_pixel(const int row, const int column, const RGBColor& pixel_color) const;
};


#endif // _WORLD_H


