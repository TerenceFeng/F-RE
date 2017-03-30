
/* ====================================================
#   Copyright (C)2017 All rights reserved.
#
#   Author        : Terence (Yongxin) Feng
#   Email         : tyxfeng@gmail.com
#   File Name     : World.cpp
#   Last Modified : 2017-03-21 17:09
#   Describe      :
#
#   Log           :
#
# ====================================================*/

#include "World.h"
#include "Display.h"
#include "Sphere.h"

#include <fcntl.h>
#include <errno.h>
#include <cstdio>
#include <cfloat>
#include <cstring>
#include <iostream>
using namespace std;

World::World(void):
	vp(ViewPlane()),
	background_color(black)
	// ambient_ptr(new Ambient)
{

	/* default view plane */
	vp.hres = 200;
	vp.vres = 200;
	vp.s = 1.0f;

}

World::~World(void)
{
	for (int i = 0; i < objects.size(); i++) {
		delete objects[i];
	}
	delete ambient_ptr;
	for (int i = 0; i < light_ptrs.size(); i++) {
		delete light_ptrs[i];
	}
}

void
World::
add_object(GeometricObject* obj_ptr)
{
	objects.push_back(obj_ptr);
}

void
World::add_light(Light* light_)
{
	light_ptrs.push_back(light_);
}

ShadeRec
World::hit_bare_bones_object(const Ray& ray) const
{
	ShadeRec sr;
	float t;
	Normal normal;
	Point3D local_hit_point;
	float tmin = FLT_MAX;
	int num_objects = objects.size();
	int nearest_object_index;

	for (int i = 0; i < num_objects; i++) {
		if (objects[i]->hit(ray, t, sr) && (t < tmin)) {
			sr.hit_an_object = true;
			tmin = t;
			sr.hit_point = ray.o + ray.d * t;
			nearest_object_index = i;
			local_hit_point = sr.local_hit_point;
			normal = sr.normal;
		}
	}

	if (sr.hit_an_object) {
		sr.t = tmin;
		sr.normal = normal;
		sr.local_hit_point = local_hit_point;
		sr.color = objects[nearest_object_index]->get_reflected_color(sr, ambient_ptr, light_ptrs) * 0.01;
	}

	return sr;
}

void
World::render_scene() const
{
	RGBColor pixel_color;
	Ray ray;
	float zw = 100.f;
	float x, y;

	Display disp(vp.hres, vp.vres);

	ray.d = Vector3D(0, 0, -1);
	float t;
	ShadeRec sr;

	for (int r = 0; r < vp.vres; r++) {
		for (int c = 0; c < vp.hres; c++) {
			x = vp.s * (c - 0.5 * (vp.hres - 1.0));
			y = vp.s * (r - 0.5 * (vp.vres - 1.0));
			ray.o = Point3D(x, y, zw);

			sr = hit_bare_bones_object(ray);
			if (sr.hit_an_object) {
				disp.add_pixel(r, c, sr.color);
			} else {
				disp.add_pixel(r, c, background_color);
			}
		}
	}
	disp.display();
}


