
/* ====================================================
#   Copyright (C)2017 All rights reserved.
#   Author        : Terence (Yongxin) Feng
#   Email         : tyxfeng@gmail.com
#   File Name     : camera.cpp
# ====================================================*/

#include "Display.h"
#include "camera.h"
#include "sampler.h"
#include "World.h"
#include <cfloat>
#include <cmath>
#include <iostream>

#define MAX_DEPTH 5
Vector3D UP(-1, -1, 0);
extern NRooks sampler;

Camera::Camera():
	position(),
	lookat(),
	up(UP),
	exposure_time(0.01)
{
	compute_uvw();
}

Camera::Camera(Point3D position_, Point3D lookat_):
	position(position_),
	lookat(lookat_),
	up(UP),
	exposure_time(0.01)
{
	compute_uvw();
}

Camera::Camera(Point3D position_, Point3D lookat_, float exp_time_):
	position(position_),
	lookat(lookat_),
	up(UP),
	exposure_time(exp_time_)
{
	compute_uvw();
}

void
Camera::compute_uvw(void)
{
	w = position - lookat;
	w.normalize();
	u = up ^ w;
	u.normalize();
	v = w ^ u;
}

void
Camera::set_viewplane(int h_ = 200, int w_ = 200, float s_ = 1.0f)
{
	height = h_;
	width = w_;
	s = s_;
}

void
Camera::set_up(int a, int b, int c)
{
	up = Vector3D(a, b, 0);
	up.normalize();
}

/* NOTE: PinHole */
PinHole::PinHole():
	Camera(),
	d(100),
	zoom(1)
{}

PinHole::PinHole(Point3D position_, Point3D lookat_, float exp_time_, float d_, float zoom_ = 1):
	Camera(position_, lookat_, exp_time_),
	d(d_),
	zoom(zoom_)
{}

Vector3D
PinHole::ray_direction(const float& xv, const float& yv) const
{
	Vector3D dir = u * xv + v * yv - w * d;
	dir.normalize();
	return dir;
}

void
PinHole::render_scene()
{
	RGBColor L;
	Ray ray;
	int depth = 0;
	s /= zoom;

	ray.o = position;
	float x, y;
	Display printer(height, width);
	// Jittered sampler(100);
	// Hammersley sampler(100);
	Point2D sp;

	for (int r = 0; r < height; r++)
		for (int c = 0; c < width; c++)
		{
			L = BLACK;
			for (int j = 0; j < sampler.num_samples; j++)
			{
				sp = sampler.sample_unit_square();
				x = s * (c - 0.5f * width + sp.x);
				y = s * (r - 0.5f * height + sp.y);
				ray.d = ray_direction(x, y);
				L += trace_ray(ray);
				// L += trace_path(ray, 0);
			}
			L /= sampler.num_samples;
			L = L * exposure_time;
			printer.add_pixel(r, c, L);
		}

	printer.display();
}

RGBColor
PinHole::cast_ray(const Ray& ray)
{
	ShadeRec sr;
	sr.color = BLACK;
	float t;
	Normal normal;
	Point3D local_hit_point;
	float tmin = FLT_MAX;
	int num_objects = world.obj_ptrs.size();
	GeometricObject* nearest_object;
	for (int i = 0; i < num_objects; i++)
	{
		if (world.obj_ptrs[i]->hit(ray, t, sr) && t < tmin)
		{
			sr.hit_an_object = true;
			tmin = t;
			sr.hit_point = ray.o + ray.d * t;
			nearest_object = world.obj_ptrs[i];
			local_hit_point = sr.local_hit_point;
			normal = sr.normal;
		}
	}
	if (sr.hit_an_object)
	{
		sr.t = tmin;
		sr.normal = normal;
		sr.local_hit_point = local_hit_point;
		sr.ray = ray;
		sr.color = nearest_object->material_ptr->area_light_shade(sr);
	}
	return sr.color;
}

RGBColor
PinHole::trace_ray(const Ray& ray)
{
	ShadeRec sr;
	sr.color = BLACK;
	float t;
	Normal normal;
	Point3D local_hit_point;
	float tmin = FLT_MAX;
	int num_objects = world.obj_ptrs.size();
	GeometricObject* nearest_object;
	for (int i = 0; i < num_objects; i++)
	{
		if (world.obj_ptrs[i]->hit(ray, t, sr) && t < tmin)
		{
			sr.hit_an_object = true;
			tmin = t;
			sr.hit_point = ray.o + ray.d * t;
			nearest_object = world.obj_ptrs[i];
			local_hit_point = sr.local_hit_point;
			normal = sr.normal;
		}
	}
	if (sr.hit_an_object)
	{
		sr.t = tmin;
		sr.normal = normal;
		sr.local_hit_point = local_hit_point;
		sr.ray = ray;
		sr.color = nearest_object->material_ptr->area_light_shade(sr);
	}
	return sr.color;
}

RGBColor
PinHole::trace_path(const Ray& ray, int depth)
{
	if (depth >= MAX_DEPTH)
		return BLACK;
	ShadeRec sr;
	Normal normal;
	Point3D local_hit_point;
	float tmin = FLT_MAX, t;
	GeometricObject *nearest_object;
	size_t num_objects = world.obj_ptrs.size();
	for (int i = 0; i < num_objects; i++)
	{
		if (world.obj_ptrs[i]->hit(ray, t, sr) && t < tmin)
		{
			sr.hit_an_object = true;
			tmin = t;
			sr.hit_point = ray.o + ray.d * t;
			nearest_object = world.obj_ptrs[i];
			local_hit_point = sr.local_hit_point;
			normal = sr.normal;
		}
	}
	if (sr.hit_an_object)
	{
		sr.t = tmin;
		sr.normal = normal;
		sr.local_hit_point = local_hit_point;
		sr.ray = ray;
		RGBColor traced_color = nearest_object->material_ptr->path_shade(sr);
		sr.hit_an_object = false;
		Ray reflected_ray(sr.hit_point, sr.reflected_dir);
		return traced_color * trace_path(reflected_ray, depth + 1);
	}
	else
		return RGBColor(1, 1, 1);

}
