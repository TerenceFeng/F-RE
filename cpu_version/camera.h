
/* ====================================================
#   Copyright (C)2017 All rights reserved.
#   Author        : Terence (Yongxin) Feng
#   Email         : tyxfeng@gmail.com
#   File Name     : camera.h
# ====================================================*/

#ifndef _CAMERA_H
#define _CAMERA_H

#include "RGBColor.h"
#include "Utilities.h"

#include <vector>

class World;
class ShadeRec;
extern World world;

class Camera
{
public:
	Camera();
	Camera(const Point3D& position_, const Point3D& lookat );
	Camera(const Point3D& position_, const Point3D& lookat_, const Vector3D& up, const float exp_time_,  const float d_, const float zoom);
	void compute_uvw(void);
	void set_viewplane(int h_, int w_, float s_);
	void set_up(const Vector3D&);

	void render_scene();
	Vector3D ray_direction(const float& xv, const float& yv) const;
public:
	RGBColor cast_ray(const Ray&);
	RGBColor trace_ray(const Ray&);
	RGBColor trace_path(const Ray&, const int);
	RGBColor trace_path_global(const Ray&, const int);

protected:
	void add_pixel(int r, int c, RGBColor& color);
	void print();

protected:
	Point3D position;
	Point3D lookat;
	Vector3D up;
	Vector3D u, v, w;
	float exposure_time;
	/* view plane */
	int width, height;
	float s; /* size of pixel */
	float d; /* view plane distance */
	float zoom;

	/* printer */
	std::vector<std::vector<float>> pixels;
	float maxval;
};

#endif
