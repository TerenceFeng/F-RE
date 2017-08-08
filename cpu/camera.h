
/* ====================================================
#   Copyright (C)2017 All rights reserved.
#   Author        : Terence (Yongxin) Feng
#   Email         : tyxfeng@gmail.com
#   File Name     : camera.h
# ====================================================*/

#ifndef _CAMERA_H
#define _CAMERA_H

#include "World.h"
#include "sampler.h"
#include "RGBColor.h"
#include "Utilities.h"
#include "ShadeRec.h"

#include <vector>
#include <cfloat>
#include <iostream>

extern World world;

#define MAX_DEPTH 5
const Vector3D UP(0, 0, 1);
extern NRooks sampler;

class Camera
{
public:
    Camera():
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

    Camera(const Point3D& position_, const Point3D& lookat_):
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

    Camera(const Point3D& position_, const Point3D& lookat_, const Vector3D& up_, const float exp_time_, const float d_, const float zoom_):
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

    void compute_uvw(void)
    {
        w = position - lookat;
        w.normalize();
        u = up ^ w;
        u.normalize();
        v = w ^ u;
    }

    void set_viewplane(int w_, int h_, float s_)
    {
        width = w_;
        height = h_;
        s = s_;

        pixels = std::vector<std::vector<float>>(height);
        maxval = FLT_MIN;
    }

    void set_up(const Vector3D& up_)
    {
        up = up_;
        up.normalize();
    }

    void render_scene(int algo = 0)
    {
        RGBColor L;
        Ray ray;
        int depth = 0;
        s /= zoom;

        ray.o = position;
        float x, y;
        Point2D sp;

        printf("Number of samples:      %d\n", sampler.num_samples);
        for (int r = 0; r < height; r++)
        {
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
                    // L += trace_path_global(ray, 0);
                }

                L /= sampler.num_samples;
                add_pixel(r, c, L);
            }
            fprintf(stderr, "\rProcess:                %3.2f", ((float)(r + 1) / height * 100));
        }

        print();
    }

    Vector3D ray_direction(const float xv, const float yv) const
    {
        Vector3D dir = u * xv + v * yv - w * d;
        return dir.hat();
        // dir.normalize();
        // return dir;
    }
public:
    RGBColor cast_ray(const Ray&);
    RGBColor trace_ray(const Ray& ray)
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

    RGBColor trace_path(const Ray& ray, const int depth)
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

    RGBColor trace_path_global(const Ray& ray, const int depth)
    {
        if (depth >= MAX_DEPTH)
            return trace_ray(ray);
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
            sr.depth = depth;
            sr.ray = ray;
            /* TODO: change path_shade to global_shade */
            RGBColor traced_color = nearest_object->material_ptr->global_shade(sr);
            Ray reflected_ray(sr.hit_point, sr.reflected_dir);
            return traced_color * trace_path_global(reflected_ray, sr.depth + 1);
        }
        return world.background_color;
    }

protected:
    void add_pixel(int r, int c, RGBColor& color)
    {
        // max_to_one(color);
        maxval = std::max(maxval, color.r);
        pixels[r].push_back(color.r);
        maxval = std::max(maxval, color.g);
        pixels[r].push_back(color.g);
        maxval = std::max(maxval, color.b);
        pixels[r].push_back(color.b);
    }

    void print() {
        FILE *fp;
        fp = fopen("result.ppm", "wb");
        if (!fp) {
            fprintf(stderr, "ERROR: cannot open output file: %s\n", strerror(errno));
            return;
        }

        fprintf(fp, "P6\n");
        fprintf(fp, "%d %d\n%d\n", width, height, 255);
        printf("\nBrightest value:        %f\n", maxval);
        for(int r = pixels.size() - 1; r >= 0; r--) {
            for(int c = 0; c < pixels[r].size(); c++) {
                fprintf(fp, "%c", (unsigned char)(int)(pixels[r][c] / maxval * 255));
            }
        }
        fprintf(fp, "\n");
        fclose(fp);

    }

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
