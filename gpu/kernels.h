#pragma once

//#include <cassert>
//#include <cmath>
//#include <cstdlib>

#include <ctime>
#include <curand.h>
#include <curand_kernel.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include "struct.h"

__global__ void init_rand(curandState *state)
{
    int w = gridDim.x * blockDim.x, h = gridDim.y * blockDim.y;
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int i = y * w + x;
    curand_init(0, 0, 0, state + i);
}
__global__ void init_ray(Ray *ray, const Camera *camera, curandState *_state, float px, float py)
{
    int w = gridDim.x * blockDim.x, h = gridDim.y * blockDim.y;
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int i = y * w + x;
    Ray r = {camera->pos, camera->dir,{ 1.0f, 1.0f, 1.0f }};
    float half_clip_x = tanf(0.5f * camera->fov_v);
    float half_clip_y = tanf(0.5f * camera->fov_h);
    float dx = 1.0f - 2.0f * curand_uniform(_state + i);
    float dy = 1.0f - 2.0f * curand_uniform(_state + i);
    //RejectionSampleDisk(&dx, &dy, _state + i);
    r.dir.x += ((float(x) + dx + px) / float(w) - 0.5f) * half_clip_x;
    r.dir.y += ((float(y) + dy + py) / float(h) - 0.5f) * half_clip_y;
    r.dir.norm();
    ray[i] = r;
}
__global__ void ray2color(Color *color, const Ray *ray)
{
    int w = gridDim.x * blockDim.x, h = gridDim.y * blockDim.y;
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int i = y * w + x;
    color[i].v.zero();
    color[i].r = fabs(ray[i].dir.x + 0.5f);
    color[i].g = fabs(ray[i].dir.y + 0.5f);
    color[i].b = fabs(ray[i].dir.z);
}

__global__ void scale_add(Color *color, Color *c2, float f)
{
    int w = gridDim.x * blockDim.x, h = gridDim.y * blockDim.y;
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int i = y * w + x;
    color[i].v += Vector(c2[i].v).scale(f);
}

__global__ void normal_map(Color *color, Ray *ray,
                          const Object *object, const size_t nobj)
{
    int w = gridDim.x * blockDim.x, h = gridDim.y * blockDim.y;
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int i = y * w + x;

    Ray *r = ray + i;
    Color *c = color + i;
    c->v.zero();

    const Object *obj = nullptr;
    Point hit;
    Normal nr;
    {
        float t = 1e10;
        ComputeHit ch;
        for (int n = 0; n < nobj; ++n)
        {
            ch.compute(r, object[n].shape);
            if (ch.isHit() && ch.t() < t)
            {
                t = ch.t();
                obj = object + n;
            }
        }
        if (obj)
        {
            hit = r->pos + Vector::Scale(r->dir, t);
            int strategy = *(int *)obj->shape;
            NormalStrategy[strategy](obj->shape, &hit, &nr);
        }
    }
    nr.add({1.0f, 1.0f, 1.0f}).scale(0.5f);
    c->r = fabs(nr.x) + fabs(1.0f - nr.y) + fabs(1.0f - nr.z);
    c->g = fabs(nr.y) + fabs(1.0f - nr.x) + fabs(1.0f - nr.z);
    c->b = fabs(nr.z) + fabs(1.0f - nr.x) + fabs(1.0f - nr.y);
    c->v.norm();
}
__global__ void ray_depth(Color *color, Ray *ray,
                          const Object *object, const size_t nobj)
{
    int w = gridDim.x * blockDim.x, h = gridDim.y * blockDim.y;
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int i = y * w + x;

    Ray *r = ray + i;
    Color *c = color + i;

    const Object *obj = nullptr;
    Point hit = {0.0f, 0.0f, 0.0f};
    float t = 1e10f;
    {
        ComputeHit ch;
        for (int n = 0; n < nobj; ++n)
        {
            ch.compute(r, object[n].shape);
            if (ch.isHit() && ch.t() < t)
            {
                t = ch.t();
                obj = object + n;
            }
        }
        if (obj)
        {
            hit = r->pos + Vector::Scale(r->dir, t);
        }
    }
    c->r = c->g = c->b = (t == 1e10f) ? 0.0f : t;
}
__global__ void ray_distance(Ray *ray, Ray *ray2, Color *color,
                             const Object *object, const size_t nobj,
                             BSDFEntity *bsdf_list,
                             curandState *_state)
{
    int w = gridDim.x * blockDim.x, h = gridDim.y * blockDim.y;
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int i = y * w + x;

    curandState &state = _state[i];

    Ray *r = ray2 + i;
    Ray *r2 = ray + i, *rtmp;
    Color *c = color + i;

    float D = 0.0f;
    for (int depth = 0; depth < 4; ++depth)
    {
        rtmp = r; r = r2; r2 = rtmp;
        if (r->factor.v.isZero())
            break;

        const Object *obj = nullptr;
        Point hit;
        Normal nr;
        {
            float t = 1e10;
            ComputeHit ch;
            for (int n = 0; n < nobj; ++n)
            {
                ch.compute(r, object[n].shape);
                if (ch.isHit() && ch.t() < t)
                {
                    t = ch.t();
                    obj = object + n;
                }
            }
            if (obj)
            {
                hit = r->pos + Vector::Scale(r->dir, t);
                int strategy = *(int *)obj->shape;
                NormalStrategy[strategy](obj->shape, &hit, &nr);
                D += t;
            }
        }

        if (obj)
        {
            if (obj->bsdf)
            {
                ComputeBSDF bsdf = {
                    nr,
                    -r->dir,
                    curand_uniform(&state),
                    curand_uniform(&state),
                    {},{},0.0f
                };
                bsdf.compute(bsdf_list + obj->bsdf->pick(curand_uniform(&state)));
                r2->pos = hit;
                r2->dir = bsdf.wi();
                r2->factor = bsdf.f();
                r2->factor.v.scale(bsdf.pdf());
            }
            else
                r2->factor.v.zero();
        }
    }
    c->v = {D, D, D};
    //c->v.scale(0.5f);
}

__global__ void trace_ray(Ray *ray, Ray *ray2, Color *color,
                          const Object *object, const size_t nobj,
                          BSDFEntity *bsdf_list,
                          curandState *_state)
{
    int w = gridDim.x * blockDim.x, h = gridDim.y * blockDim.y;
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int i = y * w + x;

    curandState &state = _state[i];
    Ray *r = ray2 + i;
    Ray *r2 = ray + i, *rtmp;
    Color *c = color + i;
    c->v.zero();

    for (int depth = 0; depth < 10; ++depth)
    {
        rtmp = r; r = r2; r2 = rtmp;
        if (r->factor.v.isZero())
            break;

        const Object *obj = nullptr;
        Point hit;
        Normal nr;
        {
            float t = 1e10;
            ComputeHit ch;
            for (int n = 0; n < nobj; ++n)
            {
                ch.compute(r, object[n].shape);
                if (ch.isHit() && ch.t() < t)
                {
                    t = ch.t();
                    obj = object + n;
                }
            }
            if (obj)
            {
                hit = r->pos + Vector::Scale(r->dir, t);
                int strategy = *(int *)obj->shape;
                NormalStrategy[strategy](obj->shape, &hit, &nr);
            }
        }

        if (obj)
        {
            if (obj->bsdf)
            {
                ComputeBSDF bsdf = {
                    nr,
                    -r->dir,
                    curand_uniform(&state), // TODO: this is slow!!!
                    curand_uniform(&state),
                    {},{},0.0f
                };
                bsdf.compute(bsdf_list + obj->bsdf->pick(curand_uniform(&state)));
                r2->pos = hit;
                r2->dir = bsdf.wi();
                // shader
                int strategy = *(int *)obj->shape;
                Color cl = bsdf.f();
                cl.v.mul(r->factor.v).scale(bsdf.pdf());
                r2->factor = ShaderStrategy[strategy](obj->shape, &hit, &nr, &cl);
            }
            else
                r2->factor.v.zero();

            if (obj->light)
            {
                ComputeLight light = {};
                light.compute(obj->light);
                //light.compute(hit, -r->dir);

                Color cv = light.L();
                cv.v.mul(r->factor.v);
                (*c).v += cv.v;

                // stop when hit light
                r2->factor.v.zero();
            }
        }
    }
}
