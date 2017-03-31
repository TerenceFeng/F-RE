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
#include "camera.h"
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

	/*
	 * Matte *matte_ptr = new Matte;
	 * matte_ptr->set_ka(0.15f);
	 * matte_ptr->set_kd(0.45f);
	 * matte_ptr->set_cd(RGBColor(0, 1, 1));
	 */

	Phong *phong_ptr = new Phong;
	phong_ptr->set_ka(0.10f);
	phong_ptr->set_kd(0.40f);
	phong_ptr->set_ks(0.25f);
	phong_ptr->set_es(20);
	phong_ptr->set_cd(RGBColor(0, 1, 1));

	Sphere *sphere_ptr = new Sphere(Point3D(0, -25, 0), 70, RGBColor(1, 0, 0));
	sphere_ptr->set_material(phong_ptr);
	w.add_object(sphere_ptr);
/*
 *
 *     Sphere *sphere_ptr2 = new Sphere(Point3D(0,60,0), 100, RGBColor(1, 0, 0));
 */
     Matte *matte_ptr2 = new Matte;
     matte_ptr2->set_ka(15.0f);
     matte_ptr2->set_kd(16.0f);
     matte_ptr2->set_cd(RGBColor(0.5f, 0.5f, 0.5f));
 /*     sphere_ptr2->set_material(matte_ptr2);
 *     w.add_object(sphere_ptr2);
 */
	Plane *plane_ptr = new Plane(Point3D(-60, -60, 0), Normal(0, 0.5f, 1));
	plane_ptr->set_material(matte_ptr2);
	w.add_object(plane_ptr);

}


int
main()
{

	World w;
	buile_two_balls_matte(w);

	PinHole camera(Point3D(200, 200, 500), Point3D(-20, -30, -60), 0.01, 400, 1.8);
	camera.set_viewplane(400, 400, 1.0f);
	camera.render_scene(w);
	return 0;
}


