#pragma once

#include <cassert>
#include <cmath>
#include <cstdlib>

// +- x -->
// |
// y
// |
// v

void ray2color(Color *color, const MDSpace<2> &dim, const MDPoint<2> &pos,
               const Ray *ray)
{
    int w = dim[0], h = dim[1];
    int x = pos[0], y = pos[1];
    int i = y * w + x;
    color[i].v.zero();
    color[i].r = fabs(ray[i].dir.x + 0.5f);
    color[i].g = fabs(ray[i].dir.y + 0.5f);
    color[i].b = fabs(ray[i].dir.z);
}

// sampling (light has shape)
//      R + O       => obj & nr & hit & wo (add light color)
//      nr + wo + O.brdf (sampling)
//                  => f & wi & pdf
//      hit + wi => R.pos & R.dir
//      f + pdf => R.factor

// 1. BSDF, light -> pool + reference
// 2. Ray, Shape -> buffer (BVH)

// Camera => R.dir, R.pos, R.factor(1)
//
// R + Grid => Object
//          ^   R + Object.shape ==hit==> Object
//          |                    |
//          =======not hit========
// R + Object[] => pos, object
//                        .light => light
//                        .shape + pos => nr
//                        .brdf + R.dir + pos + nr => R'.pos, R'.dir,
//                        R'.factor(f*pdf)
//                  C += light.Sample_L(pos, R.dir) * R.factor
//
// R => brdf[] => R'[] ?

void init_ray(Ray *ray, const MDSpace<2> &dim, const MDPoint<2> &pos,
              const Camera *camera)
{
    int w = dim[0], h = dim[1];
    int x = pos[0], y = pos[1];
    int i = y * w + x;
    Ray r = {camera->pos, camera->dir, {1.0f, 1.0f, 1.0f}};
    float half_clip_x = tanf(0.5f * camera->fov_v);
    float half_clip_y = tanf(0.5f * camera->fov_h);
    r.dir.x += (float(x + frandom()) / float(w) - 0.5f) * half_clip_x;
    r.dir.y += (float(y + frandom()) / float(h) - 0.5f) * half_clip_y;
    r.dir.norm();
    ray[i] = r;
}

void trace_ray(Ray *ray2, Color *color, const MDSpace<2> &dim,
               const MDPoint<2> &pos, const Ray *ray, const Object *object,
               const size_t *nobj)
{
    int w = dim[0], h = dim[1];
    int x = pos[0], y = pos[1];
    int i = y * w + x;

    const Ray &r = ray[i];
    if (r.factor.v.isZero())
        return;

    // 1. What object hit ?
    const Object *obj = nullptr;
    Point hit;
    {
        float t = 1e10;
        for (int n = 0; n < *nobj; ++n)
        {
            float _t = 1e10;
            if (object[n].shape->intersect(r, _t) && _t < t)
            {
                t = _t;
                obj = object + n;
            }
        }
        hit = r.pos + Vector::Scale(r.dir, t);
    }

    // 2. Where ray reflected ?
    // reflection->do(nr, wo, &wi, &pdf) => f
    // 2. What's the color ?
    // light->do(pos, dir) => L
    Ray *r2 = ray2 + i;
    Color *c = color + i;
    if (obj)
    {
        ComputeBSDF *bsdf = obj->bsdf;
        if (bsdf)
        {
            bsdf->compute(0.0f, obj->shape->getNormal(hit), -r.dir);
            *r2 = {hit, bsdf->wi(), Vector::Scale(bsdf->f().v, bsdf->pdf())};
        }
        else
            r2->factor.v.zero();

        ComputeLight *light = obj->light;
        if (light)
        {
            light->compute(hit, -r.dir);
            (*c).v += Vector::Mul(light->L().v, r.factor.v);
        }
    }
}
