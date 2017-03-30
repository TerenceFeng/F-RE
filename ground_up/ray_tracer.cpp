/* ====================================================
#   Copyright (C)2017 All rights reserved.
#
#   Author        : Terence (Yongxin) Feng
#   Email         : tyxfeng@gmail.com
#   File Name     : ray_tracer.cpp
#   Last Modified : 2017-03-17 18:25
# ====================================================*/

#include "World.h"
#include "Plane.h"
#include "Sphere.h"
#include "RGBColor.h"
#include "ShadeRec.h"
#include "Material.h"
#include "Utilities.h"

void
buile_two_balls_matte(World& w)
{
	w.set_hres(400);
	w.set_vres(400);

	Ambient *ambient_ptr = new Ambient;
	ambient_ptr->scale_radiance(1.0);
	w.set_ambient_light(ambient_ptr);

	PointLight *light_ptr = new PointLight;
	light_ptr->set_location(Point3D(100, 50, 150));
	light_ptr->set_radiance(3.0);
	w.add_light(light_ptr);

	Matte *matte_ptr = new Matte;
	matte_ptr->set_ka(0.15f);
	matte_ptr->set_kd(0.45f);
	matte_ptr->set_cd(RGBColor(1, 1, 0));

	Sphere *sphere_ptr = new Sphere(Point3D(0, -25, 0), 80, RGBColor(1, 0, 0));
	sphere_ptr->set_material(matte_ptr);
	w.add_object(sphere_ptr);

	Sphere *sphere_ptr2 = new Sphere(Point3D(0,60,0), 100, RGBColor(1, 0, 0));
	Matte *matte_ptr2 = new Matte;
	matte_ptr2->set_ka(0.2f);
	matte_ptr2->set_kd(0.3f);
	matte_ptr2->set_cd(RGBColor(0, 1, 1));
	sphere_ptr2->set_material(matte_ptr2);
	w.add_object(sphere_ptr2);
}


int
main()
{

	World w;
	buile_two_balls_matte(w);
	w.render_scene();
	return 0;
}


