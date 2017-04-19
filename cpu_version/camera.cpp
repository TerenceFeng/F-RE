
/* ====================================================
#   Copyright (C)2017 All rights reserved.
#   Author        : Terence (Yongxin) Feng
#   Email         : tyxfeng@gmail.com
#   File Name     : camera.cpp
# ====================================================*/

#include "camera.h"
#include "sampler.h"
#include "World.h"
#include <cfloat>
#include <cmath>
#include <iostream>

#define MAX_DEPTH 5
const Vector3D UP(0, 0, 1);
extern NRooks sampler;

Camera::Camera():
	position(Point3D(200, 200, 200)),
	lookat(),
	up(UP),
	width(200),
	height(200),
	s(1),
	exposure_time(0.01),
	d(100),
	zoom(1)
{
	compute_uvw();
}

Camera::Camera(const Point3D& position_, const Point3D& lookat_):
	position(position_),
	lookat(lookat_),
	up(UP),
	width(200),
	height(200),
	s(1),
	exposure_time(0.01)
{
	compute_uvw();
}

Camera::Camera(const Point3D& position_, const Point3D& lookat_, const Vector3D& up_, const float exp_time_, const float d_, const float zoom_):
	position(position_),
	lookat(lookat_),
	exposure_time(exp_time_),
	d(d_),
	s(1),
	up(up_),
	width(200),
	height(200),
	zoom(zoom_)
{
	compute_uvw();
}

Vector3D
Camera::ray_direction(const float& xv, const float& yv) const
{
	Vector3D dir = u * xv + v * yv - w * d;
	dir.normalize();
	return dir;
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
Camera::set_viewplane(int w_, int h_, float s_)
{
	width = w_;
	height = h_;
	s = s_;

	pixels = std::vector<std::vector<float>>(height);
	maxval = FLT_MIN;
}

void
Camera::set_up(const Vector3D& up_)
{
	up = up_;
	up.normalize();
}

void
Camera::add_pixel(int r, int c, RGBColor& color)
{
	// max_to_one(color);
	maxval = std::max(maxval, color.r);
	pixels[r].push_back(color.r);
	maxval = std::max(maxval, color.g);
	pixels[r].push_back(color.g);
	maxval = std::max(maxval, color.b);
	pixels[r].push_back(color.b);
}

void
Camera::print() {
	FILE *fp;
	fp = fopen("result.ppm", "wb");
	if (!fp) {
		fprintf(stderr, "ERROR: cannot open output file: %s\n", strerror(errno));
		return;
	}

	fprintf(fp, "P6\n");
	fprintf(fp, "%d %d\n%d\n", width, height, 255);
	printf("%f\n", maxval);
	for(int r = pixels.size() - 1; r >= 0; r--) {
		for(int c = 0; c < pixels[r].size(); c++) {
			fprintf(fp, "%c", (unsigned char)(int)(pixels[r][c] / maxval * 255));
		}
	}
	fprintf(fp, "\n");
	fclose(fp);

}

void
Camera::render_scene()
{
	RGBColor L;
	Ray ray;
	int depth = 0;
	s /= zoom;

	ray.o = position;
	float x, y;
	Point2D sp;

	printf("%d\n", sampler.num_samples);
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
				// L += trace_ray(ray);
				L += trace_path(ray, 0);
			}

			L /= sampler.num_samples;
			add_pixel(r, c, L);
		}

	print();
}

RGBColor
Camera::trace_ray(const Ray& ray)
{
	ShadeRec sr;
	sr.color = BLACK;
	float t;
	Normal normal;
	Point3D local_hit_point;
	float tmin = FLT_MAX;
	int num_objects = world.obj_ptrs.size();
	Object* nearest_object;
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
Camera::trace_path(const Ray& ray, const int depth)
{
	if (depth >= MAX_DEPTH)
		return BLACK;

	ShadeRec sr;
	Normal normal;
	Point3D local_hit_point;
	float tmin = FLT_MAX, t;
	Object *nearest_object;
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
		normal.normalize();
		sr.normal = normal;
		sr.local_hit_point = local_hit_point;
		sr.ray = ray;
		RGBColor traced_color = nearest_object->material_ptr->path_shade(sr);
		Ray reflected_ray(sr.hit_point, sr.reflected_dir);
		return traced_color * trace_path(reflected_ray, depth + 1) + sr.color;
	}
	return world.background_color;
}

RGBColor
Camera::trace_path_global(const Ray& ray, const int depth)
{
	if (depth >= MAX_DEPTH)
		return BLACK;
	ShadeRec sr;
	Normal normal;
	Point3D local_hit_point;
	float tmin = FLT_MAX, t;
	Object *nearest_object;
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
		normal.normalize();
		sr.normal = normal;
		sr.local_hit_point = local_hit_point;
		sr.ray = ray;
		/* TODO: change path_shade to global_shade */
		RGBColor traced_color = nearest_object->material_ptr->path_shade(sr);
		Ray reflected_ray(sr.hit_point, sr.reflected_dir);
		return traced_color * trace_path(reflected_ray, depth + 1);
	}
	return world.background_color;
}
