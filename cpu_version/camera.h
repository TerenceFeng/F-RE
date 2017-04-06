
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

class World;
extern World world;

class Camera
{
public:
	Camera();
	Camera(Point3D, Point3D);
	Camera(Point3D, Point3D, float);
	void compute_uvw(void);
	void set_viewplane(int h_, int w_, float s_);

	virtual void render_scene() = 0;
	virtual RGBColor trace_ray(const Ray&) = 0;

protected:
	Point3D position;
	Point3D lookat;
	Vector3D up;
	Vector3D u, v, w;
	float exposure_time;
	/* view plane */
	float width = 200, height = 200;
	float s = 1; /* size of pixel */

};

class PinHole: public Camera
{
public:
	PinHole();
	PinHole(Point3D position_, Point3D lookat_, float exp_time_, float d_, float zoom);
	Vector3D ray_direction(const float& xv, const float& yv) const;

	virtual void render_scene();
	virtual RGBColor trace_ray(const Ray&);
private:
	float d; /* view plane distance */
	float zoom;
};

#endif
