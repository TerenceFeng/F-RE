/* ====================================================
#   Copyright (C)2017 All rights reserved.
#
#   Author        : Terence (Yongxin) Feng
#   Email         : tyxfeng@gmail.com
#   File Name     : ray_tracer.cpp
#   Last Modified : 2017-03-17 18:25
# ====================================================*/
#include "World.h"
#include "camera.h"
#include "Sampler.h"
#include "RGBColor.h"
#include "ShadeRec.h"
#include "Material.h"
#include "Utilities.h"
#include "GeometricObject.h"

/* global variables */
World world;
PinHole camera;

void
buile_two_balls_matte()
{
	Ambient *ambient_ptr = new Ambient;
	ambient_ptr->scale_radiance(1.0);
	world.ambient_ptr = ambient_ptr;
/*
 *     PointLight *light_ptr = new PointLight(3.0f, RGBColor(1, 1, 1), Point3D(100, 50, 150), true);
 *     world.add_light(light_ptr);
 */

	Phong *phong_ptr = new Phong;
	phong_ptr->set_ka(0.10f);
	phong_ptr->set_kd(0.40f);
	phong_ptr->set_ks(0.25f);
	phong_ptr->set_es(20);
	phong_ptr->set_cd(RGBColor(1, 0.3f, 0.58f));
	// phong_ptr->set_cd(RGBColor(1, 0.843f, 0));

	/*
	 * Matte *matte_ptr = new Matte;
	 * matte_ptr->set_ka(0.1f);
	 * matte_ptr->set_kd(0.4f);
	 * matte_ptr->set_cd(RGBColor(1, 0.3f, 0.58f));
	 */
	Sphere *sphere_ptr = new Sphere(Point3D(0, -20, 0), 50, RGBColor(0, 1, 1));
	sphere_ptr->set_material(phong_ptr);
	// sphere_ptr->set_material(matte_ptr);
	world.add_object(sphere_ptr);


	Matte *matte_ptr2 = new Matte;
	matte_ptr2->set_ka(20.0f);
	matte_ptr2->set_kd(20.0f);
	matte_ptr2->set_cd(RGBColor(2.0f, 2.0f, 2.0f));
	Plane *plane_ptr = new Plane(Point3D(-150, -150, 0), Normal(0, 0.3f, 1));
	plane_ptr->set_material(matte_ptr2);
	world.add_object(plane_ptr);


	AreaLight *light_ptr2 = new AreaLight;
	Rectangle *rect_ptr = new Rectangle(Point3D(100, 50, 150), Vector3D(30, 0, -9), Vector3D(0, -30, 1));
	NRooks *sampler_ptr = new NRooks(200);
	rect_ptr->set_sampler(sampler_ptr);
	Emissive *ems_ptr = new Emissive(80.0, RGBColor(1, 1, 1));
	rect_ptr->set_material(ems_ptr);
	light_ptr2->set_object(rect_ptr);
	light_ptr2->set_material(ems_ptr);

	world.add_light(light_ptr2);

}

int
main()
{

	World w;
	buile_two_balls_matte();

	camera = PinHole(Point3D(300, 300, 200), Point3D(-20, -30, -10), 0.01, 400, 1.8);
	camera.set_viewplane(400, 400, 1.0f);
	camera.render_scene();
	return 0;
}


