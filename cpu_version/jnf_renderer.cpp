/* ====================================================
#   Copyright (C)2017 All rights reserved.
#
#   Author        : Terence (Yongxin) Feng
#   Email         : tyxfeng@gmail.com
#   File Name     : ray_tracer.cpp
#   Last Modified : 2017-03-17 18:25
# ====================================================*/
#include "Grid.h"
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
add_point_light()
{
	PointLight *light_ptr = new PointLight(5.0, RGBColor(1, 1, 1), Point3D(70, 70, 70));
	world.add_light(light_ptr);
}

void
add_pyramid_grid()
{
	Grid *grid = new Grid;

	Matte *matte_ptr2 = new Matte;
	matte_ptr2->set_ka(0.1f);
	matte_ptr2->set_kd(0.9f);
	matte_ptr2->set_cd(RGBColor(0.4, 1, 0.58f));
	Triangle *triandle_ptr2 = new Triangle(Point3D(0, 0, 50), Point3D(60, 60, 5), Point3D(0, 55, 10));
	triandle_ptr2->set_material(matte_ptr2);
	grid->add_object(triandle_ptr2);

	Matte *matte_ptr3 = new Matte;
	matte_ptr3->set_ka(0.1f);
	matte_ptr3->set_kd(0.9f);
	matte_ptr3->set_cd(RGBColor(0.4, 1, 0.58f));
	Triangle *triandle_ptr3 = new Triangle(Point3D(0, 0, 50), Point3D(50, 0, 10), Point3D(60, 60, 5));
	triandle_ptr3->set_material(matte_ptr3);
	grid->add_object(triandle_ptr3);

	grid->setup_cells();
	world.add_object(grid);
}

void add_balls()
{
	Grid *grid_ptr = new Grid;

	Phong *phong_ptr = new Phong;
	phong_ptr->set_ka(0.05f);
	phong_ptr->set_kd(0.2f);
	phong_ptr->set_ks(0.05f);
	phong_ptr->set_es(5);
	phong_ptr->set_cd(RGBColor(1, 0.3f, 0.58f));
	Sphere *sphere_ptr = new Sphere(Point3D(25, 25, 25), 50, RGBColor(0, 1, 1));
	sphere_ptr->set_material(phong_ptr);
	grid_ptr->add_object(sphere_ptr);

	Matte *matte_ptr = new Matte;
	matte_ptr->set_ka(0.1f);
	matte_ptr->set_kd(0.6f);
	matte_ptr->set_cd(RGBColor(0, 1, 1));
	Sphere *sphere_ptr2 = new Sphere(Point3D(78, 0, 30), 30, RGBColor(0));
	sphere_ptr2->set_material(matte_ptr);
	grid_ptr->add_object(sphere_ptr2);

	grid_ptr->setup_cells();
	world.add_object(grid_ptr);
}

void add_random_balls_world()
{
	int num_spheres = 50;
	float volume = 4;
	float radius = 3.5;

	for (int i = 0; i < num_spheres; i++)
	{
		Phong* phont_ptr = new Phong;
		phont_ptr->set_ka(0.3);
		phont_ptr->set_kd(0.7);
		phont_ptr->set_ks(0.25);
		phont_ptr->set_es(5);
		phont_ptr ->set_cd(RGBColor(rand_float(), rand_float(), rand_float()));

		Sphere *sphere_ptr = new Sphere;
		sphere_ptr->set_radius(radius);
		sphere_ptr->set_center(50.0 * rand_float(),
							   50.0 * rand_float(),
							   50.0 * rand_float());
		sphere_ptr->set_material(phont_ptr);

		world.add_object(sphere_ptr);
	}
}


void add_random_balls()
{
	int num_spheres = 50;
	float volume = 4;
	float radius = 3.5;

	Grid *grid_ptr = new Grid;

	for (int i = 0; i < num_spheres; i++)
	{
		Phong* phont_ptr = new Phong;
		phont_ptr->set_ka(0.3);
		phont_ptr->set_kd(0.7);
		phont_ptr->set_ks(0.25);
		phont_ptr->set_es(5);
		phont_ptr ->set_cd(RGBColor(rand_float(), rand_float(), rand_float()));

		Sphere *sphere_ptr = new Sphere;
		sphere_ptr->set_radius(radius);
		sphere_ptr->set_center(50.0 * rand_float(),
							   50.0 * rand_float(),
							   50.0 * rand_float());
		sphere_ptr->set_material(phont_ptr);

		grid_ptr->add_object(sphere_ptr);
	}
	grid_ptr->setup_cells();

	world.add_object(grid_ptr);
}

void add_plane()
{
	Matte *matte_ptr2 = new Matte;
	matte_ptr2->set_ka(0.5);
	matte_ptr2->set_kd(0.5);
	matte_ptr2->set_cd(RGBColor(1.0f, 1.0f, 1.0f));
	Plane *plane_ptr = new Plane(Point3D(0, 0, 0), Normal(0, 0, 1));
	plane_ptr->set_material(matte_ptr2);
	world.add_object(plane_ptr);
}

void read_ply_file(char *filename)
{
	Phong* phont_ptr = new Phong;
	phont_ptr->set_ka(0.1);
	phont_ptr->set_kd(0.9);
	phont_ptr->set_ks(0.45);
	phont_ptr->set_es(5);
	phont_ptr ->set_cd(RGBColor(0.4, 1.0, 0.58));

	Grid *grid_ptr = new Grid;
	grid_ptr->read_ply_file(filename);
	grid_ptr->set_material(phont_ptr);
	grid_ptr->setup_cells();
	world.add_object(grid_ptr);
}

void
build_world()
{
	add_ambient_occ();
	add_area_light();
	// add_random_balls_world();
	// add_random_balls();
	add_plane();
	// add_point_light();
	// add_pyramid_grid();
}

int
main(int argc, char ** argv)
{

	sampler = NRooks(100);
	sampler.map_samples_to_hemisphere(1);
	build_world();

	if (argc == 2)
	{
		read_ply_file(argv[1]);
	}

	/* plane: exposure_time: 0.1 */
	camera = PinHole(Point3D(200, 200, 200), Point3D(30, 30, 30), 0.1, 400, 1);
	camera.set_viewplane(400, 400, 1.0f);
	camera.set_up(-1, -1, 1);
	camera.render_scene();
	return 0;
}

