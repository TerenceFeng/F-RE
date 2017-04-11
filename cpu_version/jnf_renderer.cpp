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
NRooks sampler;

void
add_ambient_occ()
{
	AmbientOccluder* occluder_ptr = new AmbientOccluder(13.0, RGBColor(1, 1, 1), RGBColor(0, 0, 0));
	occluder_ptr->set_sampler(&sampler);
	world.ambient_ptr = occluder_ptr;
}

void
add_area_light()
{
	AreaLight *light_ptr2 = new AreaLight;
	Rectangle *rect_ptr = new Rectangle(Point3D(100, 100, 100), Vector3D(30, 0, -9) * 0.7, Vector3D(0, -30, 1) * 0.7);
	Emissive *ems_ptr = new Emissive(200.0, RGBColor(1, 1, 1));
	rect_ptr->set_material(ems_ptr);
	rect_ptr->set_sampler(&sampler);
	light_ptr2->set_object(rect_ptr);
	light_ptr2->set_material(ems_ptr);

	world.add_light(light_ptr2);
}

void
add_pyramid()
{
	Matte *matte_ptr = new Matte;
	matte_ptr->set_ka(0.1f);
	matte_ptr->set_kd(0.9f);
	matte_ptr->set_cd(RGBColor(0.4, 1, 0.58f));
	TrianglarPyramid *t_ptr = new TrianglarPyramid(Point3D(0, 0, 50), Point3D(60, 60, 30), Point3D(50, 0, 10), Point3D(0, 55, 10));
	t_ptr->set_material(matte_ptr);
	world.add_object(t_ptr);
}

void
add_triangles()
{
	Matte *matte_ptr = new Matte;
	matte_ptr->set_ka(0.4f);
	matte_ptr->set_kd(0.6f);
	matte_ptr->set_cd(RGBColor(1, 0.4f, 0.58f));
	Triangle *triandle_ptr = new Triangle(Point3D(0, 0, 50), Point3D(50, 0, 10), Point3D(0, 55, 10));
	triandle_ptr->set_material(matte_ptr);
	world.add_object(triandle_ptr);

	Matte *matte_ptr2 = new Matte;
	matte_ptr2->set_ka(0.1f);
	matte_ptr2->set_kd(0.9f);
	matte_ptr2->set_cd(RGBColor(0.4, 1, 0.58f));
	Triangle *triandle_ptr2 = new Triangle(Point3D(0, 0, 50), Point3D(60, 60, 5), Point3D(0, 55, 10));
	triandle_ptr2->set_material(matte_ptr2);
	world.add_object(triandle_ptr2);

	Phong *phong_ptr = new Phong;
	phong_ptr->set_ka(0.20f);
	phong_ptr->set_kd(0.50f);
	phong_ptr->set_ks(0.10f);
	phong_ptr->set_es(50);
	phong_ptr->set_cd(RGBColor(1, 0.3f, 0.58f));

	Matte *matte_ptr3 = new Matte;
	matte_ptr3->set_ka(0.1f);
	matte_ptr3->set_kd(0.9f);
	matte_ptr3->set_cd(RGBColor(0.4, 1, 0.58f));
	Triangle *triandle_ptr3 = new Triangle(Point3D(0, 0, 50), Point3D(50, 0, 10), Point3D(60, 60, 5));
	triandle_ptr3->set_material(matte_ptr3);
	world.add_object(triandle_ptr3);
}

void add_balls()
{
	Phong *phong_ptr = new Phong;
	phong_ptr->set_ka(0.20f);
	phong_ptr->set_kd(0.50f);
	phong_ptr->set_ks(0.10f);
	phong_ptr->set_es(50);
	phong_ptr->set_cd(RGBColor(1, 0.3f, 0.58f));

	/*
	 * Matte *matte_ptr3 = new Matte;
	 * matte_ptr3->set_ka(0.1f);
	 * matte_ptr3->set_kd(0.6f);
	 * matte_ptr3->set_cd(RGBColor(1, 0.3f, 0.58f));
	 */
	Sphere *sphere_ptr = new Sphere(Point3D(0, -20, 0), 50, RGBColor(0, 1, 1));
	sphere_ptr->set_material(phong_ptr);
	// sphere_ptr->set_material(matte_ptr3);
	world.add_object(sphere_ptr);

	Matte *matte_ptr = new Matte;
	matte_ptr->set_ka(0.1f);
	matte_ptr->set_kd(0.6f);
	matte_ptr->set_cd(RGBColor(0, 1, 1));
	Sphere *sphere_ptr2 = new Sphere(Point3D(-5, 50, -35), 30, RGBColor(0));
	sphere_ptr2->set_material(matte_ptr);
	world.add_object(sphere_ptr2);

}

void add_plane()
{
	Matte *matte_ptr2 = new Matte;
	matte_ptr2->set_ka(0.5);
	matte_ptr2->set_kd(0.5);
	matte_ptr2->set_cd(RGBColor(1.0f, 1.0f, 1.0f));
	// Plane *plane_ptr = new Plane(Point3D(-150, -150, 0), Normal(0, 0.3f, 1));
	Plane *plane_ptr = new Plane(Point3D(0, 0, 0), Normal(0, 0, 1));
	plane_ptr->set_material(matte_ptr2);
	world.add_object(plane_ptr);
}

void
build_world()
{
	add_ambient_occ();
	add_area_light();
	// add_balls();
	// add_triangles();
	add_pyramid();
	add_plane();
}

int
main()
{

	World w;
	sampler = NRooks(200);
	sampler.map_samples_to_hemisphere(1);
	build_world();

	// camera = PinHole(Point3D(300, 300, 200), Point3D(-20, -30, -10), 0.028, 400, 1.8);
	camera = PinHole(Point3D(200, 200, 200), Point3D(25, 25, 25), 0.10, 400, 1);
	camera.set_viewplane(400, 400, 1.0f);
	camera.set_up(-1, -1, 1);
	camera.render_scene();
	return 0;
}


