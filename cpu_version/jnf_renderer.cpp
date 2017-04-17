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
	Rectangle *rect_ptr = new Rectangle(Point3D(250, 250, 300), Vector3D(30, 0, -9), Vector3D(0, -30, 1));
	Emissive *ems_ptr = new Emissive(400.0, RGBColor(1, 1, 1));
	rect_ptr->set_material(ems_ptr);
	rect_ptr->set_sampler(&sampler);
	light_ptr2->set_object(rect_ptr);
	light_ptr2->set_material(ems_ptr);

	world.add_light(light_ptr2);
	// world.add_object(rect_ptr);
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

void add_random_balls()
{
	int num_spheres = 3;
	float volume = 4;
	float radius = 5;

	Grid *grid_ptr = new Grid;

	for (int i = 0; i < num_spheres; i++)
	{
		Matte * reflect_ptr = new Matte;
		reflect_ptr->set_ka(0.4);
		reflect_ptr->set_kd(0.35);
		reflect_ptr->set_cd(RGBColor(rand_float(), rand_float(), rand_float()));
		Sphere *sphere_ptr = new Sphere;
		sphere_ptr->set_radius(radius);
		sphere_ptr->set_center(50.0 * rand_float(),
							   50.0 * rand_float(),
							   30.0 * rand_float() + 5);
		sphere_ptr->set_material(reflect_ptr);

		grid_ptr->add_object(sphere_ptr);
	}
	grid_ptr->setup_cells();

	world.add_object(grid_ptr);
}

void add_plane()
{
	/*
	 * Matte *matte_ptr2 = new Matte;
	 * matte_ptr2->set_ka(0.5);
	 * matte_ptr2->set_kd(0.5);
	 * matte_ptr2->set_cd(RGBColor(1.0f, 1.0f, 1.0f));
	 */
	GlossyReflective *reflect_ptr = new GlossyReflective;
	reflect_ptr->set_ka(0);
	reflect_ptr->set_kd(0);
	reflect_ptr->set_ks(0);
	reflect_ptr->set_exponent(100);
	reflect_ptr->set_kr(0.9);
	reflect_ptr->set_cr(WHITE);

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
	camera = PinHole(Point3D(200, 200, 200), Point3D(20, 20, 20), 0.25, 400, 2.2);
	camera.set_viewplane(400, 400, 1.0f);
	camera.set_up(-1, -1, 1);

	add_ambient_occ();
	add_area_light();
	add_random_balls();
	add_plane();
	// add_point_light();
	// add_pyramid_grid();
}

void
build_cornell_box()
{

	camera = PinHole(Point3D(400, 0, 0), Point3D(200, 0, 0), 0.1, 400, 1);
	camera.set_viewplane(400, 400, 1.0);
	camera.set_up(0, 0, 1);

	/*
	 * AreaLight *light_ptr = new AreaLight;
	 * Rectangle *rect_ptr = new Rectangle(Point3D(300, -10, 10), Vector3D(0, 20, 0), Vector3D(0, 0, -20));
	 * Emissive *ems_ptr = new Emissive(200.0, RGBColor(1, 1, 1));
	 * rect_ptr->set_material(ems_ptr);
	 * rect_ptr->set_sampler(&sampler);
	 * light_ptr->set_object(rect_ptr);
	 * light_ptr->set_material(ems_ptr);
	 */

	// world.add_light(light_ptr);

	PointLight *point_light_ptr = new PointLight(80.0, RGBColor(1, 1, 1), Point3D(0, 0, 0));
	world.add_light(point_light_ptr);

	Plane *plane_left = new Plane(Point3D(0, -230, 0), Normal(0, 1, 0));
	Matte *mat_left = new Matte(0.2, 0.3, RGBColor(rand_float(), rand_float(), rand_float()));
	plane_left->set_material(mat_left);

	Plane *plane_right = new Plane(Point3D(0, 230, 0), Normal(0, -1, 0));
	Matte *mat_right = new Matte(0.2, 0.3, RGBColor(rand_float(), rand_float(), rand_float()));
	plane_right->set_material(mat_right);

	Plane *plane_up = new Plane(Point3D(0, 0, 150), Normal(0, 0, -1));
	Matte *mat_up = new Matte(0.2, 0.3, RGBColor(rand_float(), rand_float(), rand_float()));
	plane_up->set_material(mat_up);

	Plane *plane_down = new Plane(Point3D(0, 0, -150), Normal(0, 0, 1));
	Matte *mat_down = new Matte(0.2, 0.3, RGBColor(rand_float(), rand_float(), rand_float()));
	plane_down->set_material(mat_down);

	Plane *plane_back = new Plane(Point3D(-300, 0, 0), Normal(1, 0, 0));
	Matte *mat_back = new Matte(0.2, 0.3, RGBColor(rand_float(), rand_float(), rand_float()));
	plane_back->set_material(mat_back);

	world.add_object(plane_up);
	world.add_object(plane_down);
	world.add_object(plane_left);
	world.add_object(plane_right);
	world.add_object(plane_back);

	Sphere *sphere_ptr = new Sphere;
	GlossyReflective *reflect_ptr = new GlossyReflective;
	reflect_ptr->set_ka(0);
	reflect_ptr->set_kd(0);
	reflect_ptr->set_ks(0);
	reflect_ptr->set_exponent(100000);
	reflect_ptr->set_kr(0.9);
	reflect_ptr->set_cr(WHITE);
	sphere_ptr->set_material(reflect_ptr);

	world.add_object(sphere_ptr);

	add_ambient_occ();
}

int
main(int argc, char ** argv)
{

	sampler = NRooks(300);
	sampler.map_samples_to_hemisphere(1);

	if (argc == 2)
	{
		read_ply_file(argv[1]);
	}
	/* plane: exposure_time: 0.1 */
	build_world();
	// build_cornell_box();
	camera.render_scene();
	return 0;
}

