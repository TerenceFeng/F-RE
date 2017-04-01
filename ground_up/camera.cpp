
/* ====================================================
#   Copyright (C)2017 All rights reserved.
#   Author        : Terence (Yongxin) Feng
#   Email         : tyxfeng@gmail.com
#   File Name     : camera.cpp
# ====================================================*/

#include "Display.h"
#include "camera.h"
#include <cfloat>

Camera::Camera():
	position(),
	lookat(),
	up(0, 1, 0),
	exposure_time(0.01)
{
	compute_uvw();
}

Camera::Camera(Point3D position_, Point3D lookat_):
	position(position_),
	lookat(lookat_),
	up(0, 1, 0),
	exposure_time(0.01)
{
	compute_uvw();
}

Camera::Camera(Point3D position_, Point3D lookat_, float exp_time_):
	position(position_),
	lookat(lookat_),
	up(0, 1, 0),
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
PinHole::render_scene(const World& w)
{
	RGBColor L;
	Ray ray;
	int depth = 0;
	s /= zoom;
	/* TODO: sample point in [0, 1] x [0, 1] */
	/* sample point on a pixel */

	ray.o = position;
	float x, y;
	Display printer(height, width);

	for (int r = 0; r < height; r++)
	{
		for (int c = 0; c < width; c++)
		{
			L = BLACK;
			x = s * (c - 0.5f * (width - 1.0f));
			y = s * (r - 0.5f * (height - 1.0f));
			ray.d = ray_direction(x, y);
			L += trace_ray(ray, w);
			L = L * exposure_time;
			printer.add_pixel(r, c, L);
		}
	}

	printer.display();
}

RGBColor
PinHole::trace_ray(const Ray& ray, const World& world)
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
		sr.color = nearest_object->get_reflected_color(sr, world.ambient_ptr, world.light_ptrs, world.obj_ptrs);
	}

	return sr.color;
}
