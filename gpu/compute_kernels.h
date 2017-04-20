#pragma once

#include <cassert>
#include <cmath>
#include <cstdlib>


/*
void kernel_simple(Color *img, const MDSpace<2> &dim, const MDPoint<2> &pos,
                   const Scene *scene, const Camera *camera)
{
    int h = dim[0];
    int w = dim[1];
    int y = pos[0];
    int x = pos[1];
    Color & c = img[y * w + x];
    c.v.zero();

    // shoot ray, get obj
    Ray r = {camera->pos, camera->dir};
    float len = tanf(camera->fov_h) * r.dir.z;
    r.dir.x += (x - w * 0.5f) / w * len;
    r.dir.y += (y - h * 0.5f) / h * len;
    r.dir.norm();
    float t = 0.f;
    Object *obj = nullptr;
    for (Object *s : scene->objs)
    {
        float t0;
        if (s->shape->intersect(r, t0) && (!obj || t > t0))
        {
            obj = s;
            t = t0;
        }
    }
    Vertex hit = r.pos + Vertex::Scale(r.dir, t);
    if (obj == nullptr) return;

    // calculate light
    Vector wi, wo;
    float pdf;
    for (PointLight *l : scene->lights)
    {
        c.v += l->Sample_L(hit, &wi, &pdf).v.mul(obj->bsdf->f(wo, wi).v);
    }

    // sample reflection
    // loop
}
*/

// +- x -->
// |
// y
// |
// v

void shoot_ray(Ray *ray, const MDSpace<2> &dim, const MDPoint<2> &pos,
               const Camera *camera)
{
    int w = dim[0], h = dim[1];
    int x = pos[0], y = pos[1];
    int i = y * w + x;
    Ray r = {camera->pos, camera->dir};
    float half_clip_x = tanf(0.5f * camera->fov_v);
    float half_clip_y = tanf(0.5f * camera->fov_h);
    r.dir.x += (float(x) / float(w) - 0.5f) * half_clip_x;
    r.dir.y += (float(y) / float(h) - 0.5f) * half_clip_y;
    r.dir.norm();
    ray[i] = r;
}

// collect objects, hit-pos
void calc_obj_pos(const Object **obj, Point *hit, const MDSpace<2> &dim,
                  const MDPoint<2> &pos, const Ray *rays, const Object *object,
                  size_t *nobj)
{
    int w = dim[0], h = dim[1];
    int x = pos[0], y = pos[1];
    int i = y * w + x;

    const Ray &r = rays[i];

    // find nearest object
    float t_min = 1e20, t = 1e20;
    const Object *o = nullptr;
    for (int n = 0; n < *nobj; ++n)
    {
        if (object[n].shape->intersect(r, t) && t < t_min)
        {
            t_min = t;
            o = object + n;
        }
    }
    obj[i] = o;
    hit[i] = r.pos + Vector::Scale(r.dir, t_min);
}

// collect normal
void calc_normal(Normal *nr, const MDSpace<2> &dim, const MDPoint<2> &pos,
                 const Point *hit, const Object **obj)
{
    int h = dim[0], w = dim[1];
    int y = pos[0], x = pos[1];
    int i = y * w + x;
    nr[i].zero();
    if (obj[i])
        nr[i] = obj[i]->shape->getNormal(hit[i]);
}

// collect wi
// count(wi) = N * nlight
void calc_wi(Vector *wi, const MDSpace<2> &dim, const MDPoint<2> &pos,
             const Point *hit, const Object **obj,
             const Light **light, const size_t *nlight)
{
    int w = dim[0], h = dim[1];
    int x = pos[0], y = pos[1];
    int i = y * w + x;
    if (obj[i])
    {
        wi = wi + (*nlight) * i;
        for (int L = 0; L < *nlight; ++L)
            wi[L] = light[L]->getDirection(hit[i]);
    }
}
void calc_wo(Vector *wo, const MDSpace<2> &dim, const MDPoint<2> &pos,
             const Ray *ray)
{
    int w = dim[0], h = dim[1];
    int x = pos[0], y = pos[1];
    int i = y * w + x;
    wo[i] = Vector::Scale(ray[i].dir, -1.0f);
}

// brdf(n, wi, wo) -> f
// count(f) = N * nlight
void calc_f(Color *f, const MDSpace<2> &dim, const MDPoint<2> &pos,
            const Normal *nr, const Vector *wi, const Vector *wo,
            const Object **obj, const size_t *nlight)
{
    int w = dim[0], h = dim[1];
    int x = pos[0], y = pos[1];
    int i = y * w + x;
    if (obj[i])
    {
        wi = wi + (*nlight) * i;
        f = f + (*nlight) * i;
        for (int L = 0; L < (*nlight); ++L)
        {
            f[L] = obj[i]->bsdf->f(nr[i], wi[L], wo[i]);
        }
    }
}

// f * L = color
// count(L) = N * nlight
void calc_color(Color *color, const MDSpace<2> &dim, const MDPoint<2> &pos,
                const Normal *nr, const Vector *wi, const Point *hit,
                const Color *f, const Light **light, const size_t *nlight)
{
    int w = dim[0], h = dim[1];
    int x = pos[0], y = pos[1];
    int i = y * w + x;
    f = f + (*nlight) * i;
    wi = wi + (*nlight) * i;
    Color c;
    for (int L = 0; L < *nlight; ++L)
    {
        c = f[L];
        c.v.mul(light[L]->Sample_L(hit[i], nullptr, nullptr).v)
            .scale(nr[i].dot(wi[L]));
        color[i].v += c.v;
    }
}

// DEBUG
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
void obj2color(Color *c, const MDSpace<2> &dim, const MDPoint<2> &pos,
               const Object **obj)
{
    int w = dim[0], h = dim[1];
    int x = pos[0], y = pos[1];
    int i = y * w + x;
    c[i].r = c[i].g = c[i].b = (obj[i]) ? 1.0f : 0.0f;
}
void pos2color(Color *c, const MDSpace<2> &dim, const MDPoint<2> &pos,
               const Point *hit, const Object **obj)
{
    int w = dim[0], h = dim[1];
    int x = pos[0], y = pos[1];
    int i = y * w + x;
    c[i].v.zero();
    if (obj[i])
        c[i].r = c[i].g = c[i].b = fabs(2.7f - hit[i].z);
}
void nr2color(Color *c, const MDSpace<2> &dim, const MDPoint<2> &pos,
              const Normal *nr, const Object **obj)
{
    int w = dim[0], h = dim[1];
    int x = pos[0], y = pos[1];
    int i = y * w + x;
    c[i].v.zero();
    if (obj[i])
    {
        c[i].r = fabs(nr[i].x);
        c[i].g = fabs(nr[i].y);
        c[i].b = fabs(nr[i].z);
    }
}
void wi2color(Color *c, const MDSpace<2> &dim, const MDPoint<2> &pos,
              const Vector *wi, const Object **obj, const size_t *nlight,
              const size_t *L)
{
    int h = dim[0], w = dim[1];
    int y = pos[0], x = pos[1];
    int i = y * w + x;
    wi = wi + (*nlight) * i + (*L);
    c[i].v.zero();
    if (obj[i])
    {
        c[i].r = fabs(wi->x + 0.5f);
        c[i].g = fabs(wi->y + 0.5f);
        c[i].b = fabs(wi->z);
    }
}
void f2color(Color *c, const MDSpace<2> &dim, const MDPoint<2> &pos,
             const Color *f, const size_t *nlight, const size_t *L)
{
    int w = dim[0], h = dim[1];
    int x = pos[0], y = pos[1];
    int i = y * w + x;
    c[i] = *(f + (*nlight) * i + (*L));
}
