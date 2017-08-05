/* ====================================================
#   Copyright (C)2017 All rights reserved.
#
#   Author        : Terence (Yongxin) Feng
#   Email         : tyxfeng@gmail.com
#   File Name     : ray_tracer.cpp
#   Last Modified : 2017-03-17 18:25
# ====================================================*/
#include "object/Grid.h"
#include "World.h"
#include "camera.h"
#include "sampler.h"
#include "RGBColor.h"
#include "ShadeRec.h"
#include "Material.h"
#include "Utilities.h"
#include "object/Object.h"

/* global variables */
World world;
Camera camera;
NRooks sampler;

void
add_ambient_occ()
{
	AmbientOccluder* occluder_ptr = new AmbientOccluder(10, RGBColor(1, 1, 1), RGBColor(0.1, 0.1, 0.1));
	occluder_ptr->set_sampler(&sampler);
	world.ambient_ptr = occluder_ptr;
}

void
add_env_light()
{
	Emissive *e = new Emissive(1, WHITE);
	world.add_light(new EnviormentLight(&sampler, e));
}

void
add_area_light()
{
	AreaLight *light_ptr2 = new AreaLight;
	Rectangle *rect_ptr = new Rectangle(Point3D(250, 250, 300), Vector3D(30, 0, -9), Vector3D(0, -30, 1));
	Emissive *ems_ptr = new Emissive(300.0, RGBColor(1, 1, 1));
	rect_ptr->set_material(ems_ptr);
	rect_ptr->set_sampler(&sampler);
	light_ptr2->set_object(rect_ptr);
	light_ptr2->set_material(ems_ptr);
	world.add_light(light_ptr2);
}

void
add_pyramid_grid()
{
	Grid *grid = new Grid;

	Matte *matte_ptr = new Matte;
	matte_ptr->set_ka(0.1f);
	matte_ptr->set_kd(0.9f);
	matte_ptr->set_color(RGBColor(0.4, 1, 0.58f));
	Triangle *triandle_ptr = new Triangle(Point3D(0, 0, 50), Point3D(60, 60, 5), Point3D(0, 55, 10));
	triandle_ptr->set_material(matte_ptr);
	grid->add_object(triandle_ptr);

	Matte *matte_ptr3 = new Matte;
	matte_ptr3->set_ka(0.1f);
	matte_ptr3->set_kd(0.9f);
	matte_ptr3->set_color(RGBColor(0.4, 1, 0.58f));
	Triangle *triandle_ptr3 = new Triangle(Point3D(0, 0, 50), Point3D(50, 0, 10), Point3D(60, 60, 5));
	triandle_ptr3->set_material(matte_ptr3);
	grid->add_object(triandle_ptr3);

	grid->setup_cells();
	world.add_object(grid);
}

void add_random_balls()
{
	int num_spheres = 4;
	float volume = 4;
	float radius = 10;

	Grid *grid_ptr = new Grid;

	for (int i = 0; i < num_spheres; i++)
	{
		Matte *reflect_ptr = new Matte(0.6, 0.6, RGBColor(rand_float(), rand_float(), rand_float()));

	// GlossyReflective *reflect_ptr = new GlossyReflective(0, 0, 0, 1, 100, WHITE);

		Sphere *sphere_ptr = new Sphere(Point3D(50.0 * rand_float(), 50.0 * rand_float(), 10.0 * rand_float() + 5), radius, reflect_ptr);

		grid_ptr->add_object(sphere_ptr);
	}

	grid_ptr->setup_cells();
	world.add_object(grid_ptr);
}

void add_plane()
{
	GlossyReflective *reflect_ptr = new GlossyReflective(0, 0, 0, 1, 100, WHITE);
	// Reflective *reflect_ptr = new Reflective(0, 0, 0.2, 0.8, 20, WHITE, WHITE);

	Plane *plane_ptr = new Plane(Point3D(0, 0, 0), Normal(0, 0, 1));
	plane_ptr->set_material(reflect_ptr);
	world.add_object(plane_ptr);
}

void read_ply_file(char *filename)
{
	Phong* phont_ptr = new Phong;
	phont_ptr->set_ka(0.1);
	phont_ptr->set_kd(0.9);
	phont_ptr->set_ks(0.45);
	phont_ptr->set_es(5);
	phont_ptr ->set_color(RGBColor(0.4, 1.0, 0.58));

	Grid *grid_ptr = new Grid;
	grid_ptr->read_ply_file(filename);
	grid_ptr->set_material(phont_ptr);
	grid_ptr->setup_cells();
	world.add_object(grid_ptr);
}

void
test_path_tracing()
{
	camera = Camera(Point3D(200, 200, 200), Point3D(20, 20, 20), Vector3D(-1, -1, 1), 3, 400, 1.8);
	camera.set_viewplane(300, 300, 1.0f);

	add_ambient_occ();
	add_area_light();
	add_random_balls();
	add_plane();
}

void
test_cornell_box()
{

	camera = Camera(Point3D(400, 0, 0), Point3D(300, 0, 0), Vector3D(0, 0, 1), 1.5, 400, 1);
	camera.set_viewplane(400, 300, 1.0);

	AreaLight *light_ptr = new AreaLight;
	Rectangle *rect_ptr = new Rectangle(Point3D(300, -10, 140), Vector3D(14, 20, 1), Vector3D(-5, 1, -20));
	Emissive *ems_ptr = new Emissive(200.0, RGBColor(1, 1, 1));
	rect_ptr->set_material(ems_ptr);
	rect_ptr->set_sampler(&sampler);
	light_ptr->set_object(rect_ptr);
	light_ptr->set_material(ems_ptr);
	world.add_light(light_ptr);

	AreaLight *light_ptr2 = new AreaLight;
	Rectangle *rect_ptr2 = new Rectangle(Point3D(300, -120, -140), Vector3D(14, -20, -1), Vector3D(5, 1, 20));
	Emissive *ems_ptr2 = new Emissive(200.0, RGBColor(1, 1, 1));
	rect_ptr2->set_material(ems_ptr2);
	rect_ptr2->set_sampler(&sampler);
	light_ptr2->set_object(rect_ptr2);
	light_ptr2->set_material(ems_ptr2);
	world.add_light(light_ptr2);

	add_ambient_occ();

	Plane *plane_left = new Plane(Point3D(0, -230, 0), Normal(0, 1, 0));
	Matte *mat_left = new Matte(0.2, 0.6, RGBColor(0.75, 0.75, 0.65));
	plane_left->set_material(mat_left);

	Plane *plane_right = new Plane(Point3D(0, 230, 0), Normal(0, -1, 0));
	Matte *mat_right = new Matte(0.2, 0.6, RGBColor(0.75, 0.75, 0.65));
	plane_right->set_material(mat_right);

	Plane *plane_up = new Plane(Point3D(0, 0, 150), Normal(0, 0, -1));
	Matte *mat_up = new Matte(0.2, 0.6, RGBColor(0.75, 0.25, 0.25));
	plane_up->set_material(mat_up);

	Plane *plane_down = new Plane(Point3D(0, 0, -150), Normal(0, 0, 1));
	Matte *mat_down = new Matte(0.2, 0.6, RGBColor(0.25, 0.25, 0.75));
	plane_down->set_material(mat_down);

	Plane *plane_back = new Plane(Point3D(-300, 0, 0), Normal(1, 0, 0));
	Matte *mat_back = new Matte(0.2, 0.25, RGBColor(0.75, 0.75, 0.55));
	plane_back->set_material(mat_back);

	Plane *plane_front= new Plane(Point3D(400, 0, 0), Normal(-1, 0, 0));
	Matte *mat_front= new Matte(0, 0, WHITE * 0.1);
	plane_front->set_material(mat_front);

	world.add_object(plane_up);
	world.add_object(plane_down);
	world.add_object(plane_left);
	world.add_object(plane_right);
	world.add_object(plane_back);
	world.add_object(plane_front);

	GlossyReflective *reflect_ptr = new GlossyReflective;
	reflect_ptr->set_ka(0);
	reflect_ptr->set_kd(0.2);
	reflect_ptr->set_ks(0.2);
	reflect_ptr->set_exponent(1000000);
	reflect_ptr->set_kr(0.4);
	reflect_ptr->set_color(WHITE * 0.6);
	Sphere *sphere_ptr = new Sphere(Point3D(-150, -20, -90), 60, reflect_ptr);

	world.add_object(sphere_ptr);
}

int
main(int argc, char ** argv)
{

    sampler = NRooks(100);
	sampler.map_samples_to_hemisphere(1);

	if (argc == 2)
	{
        read_ply_file(argv[1]);
	}
	// test_path_tracing();
	test_cornell_box();
	camera.render_scene();
	return 0;
}

