#pragma once

#include <cmath>
#include <cstdlib>
#include <cassert>

/*
static unsigned short Xi[] = { 0, 0, 0 };

Vertex radiance(const Ray & r, int depth, const Scene * scene)
{
    if (depth > 6) return Vertex::Zero();

    // find intersect object
    Sphere * obj = nullptr;
    float t = 1e10;  // distance to intersection
    {
        float t0 = .0f;
        Sphere * tmp = nullptr;
        for (Sphere * s : scene->objs)
        {
            if (s->intersect(r, t0) && t0 < t)
            {
                t = t0;
                tmp = s;
            }
        }
        obj = (Sphere *)tmp;
    }
    if (obj == nullptr)
        return Vertex::Zero();  // if miss, return black

    Vertex x = r.pos + Vertex::Scale(r.dir, t);
    Vertex n = (x - obj->center).norm();
    Vertex nl = n.dot(r.dir) < 0 ? n : Vertex::Scale(n, -1);
    Vertex f = obj->c.v;

    double p =
        f.x > f.y && f.x > f.z ? f.x : f.y > f.z ? f.y : f.z;  // max refl
    if (++depth > 5)
    {
        if (erand48(Xi) < p)
            f.scale(1 / p);
        else
            return obj->e.v;  // R.R.
    }

    // Ideal DIFFUSE reflection
    double r1 = 2 * M_PI * erand48(Xi), r2 = erand48(Xi), r2s = sqrt(r2);
    Vertex w = nl;
    Vertex u = Vertex::Cross(fabs(w.x) > .1 ? Vertex(0, 1) : Vertex(1),
w).norm();
    Vertex v = Vertex::Cross(w, u);
    Vertex d =
        (u.scale(cos(r1) * r2s) + v.scale(sin(r1) * r2s) + w.scale(sqrt(1 -
r2))).norm();
    return obj->e.v + f.mul(radiance({x, d}, depth, scene));
}

// sampler
// radiance calc
void kernel_smallpt(Color * img, const MDSpace<2> & dim, const MDPoint<2> & pos,
                    const Scene * scene, const Camera * camera)
{
    int h = dim[0];
    int w = dim[1];
    int y = pos[0];
    int x = pos[1];
    int samps = 10;
    Color & c = img[y * w + x];

    Ray cam = {{50, 52, 295.6}, {0, -0.042612, -1}};  // cam pos, dir
    cam.dir.norm();

    Vertex cx = {(float)(w * .5135 / h), .0, .0};
    Vertex cy = Vertex::Cross(cx, cam.dir).norm().scale(0.5135f);
    Color r;

    for (int sy = 0; sy < 2; sy++)
    {
        for (int sx = 0; sx < 2; sx++)
        {
            r.v.zero();
            for (int s = 0; s < samps; s++)
            {
                double r1 = 2 * erand48(Xi);
                double dx = r1 < 1 ? sqrt(r1) - 1 : 1 - sqrt(2 - r1);
                double r2 = 2 * erand48(Xi);
                double dy = r2 < 1 ? sqrt(r2) - 1 : 1 - sqrt(2 - r2);

                Vertex d =
                    Vertex::Scale(cx, (((sx + .5 + dx) / 2 + x) / w - .5)) +
                    Vertex::Scale(cy, (((sy + .5 + dy) / 2 + y) / h - .5)) +
                    cam.dir;

                r.v += radiance({cam.pos + d.scale(139), Vertex::Norm(d)}, 0,
                                scene)
                           .scale(1. / samps);
            }
            c.v += Vertex(clamp(r.x), clamp(r.y), clamp(r.z)).scale(.25);
        }
    }
}
*/

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
    int h = dim[0], w = dim[1];
    int y = pos[0], x = pos[1];
    int i = y * w + x;
    Ray r = {camera->pos, camera->dir};
    float half_clip_x = tanf(0.5f * camera->fov_v);
    float half_clip_y = tanf(0.5f * camera->fov_h);
    r.dir.x += (float(x) / float(w) - 0.5f) * half_clip_x;
    r.dir.y += (float(y) / float(h) - 0.5f) * half_clip_y;
    r.dir.norm();
    ray[i] = r;
}
void ray2color(Color *color, const MDSpace<2> &dim, const MDPoint<2> &pos,
               const Ray *ray)
{
    int h = dim[0], w = dim[1];
    int y = pos[0], x = pos[1];
    int i = y * w + x;
    color[i].r = fabs(ray[i].dir.x);
    color[i].g = fabs(ray[i].dir.y);
    color[i].b = fabs(ray[i].dir.z);
}

// collect objects, hit-pos
void calc_obj_pos(const Object **obj, Point *hit, const MDSpace<2> &dim,
                  const MDPoint<2> &pos, const Ray *rays, const Object *object,
                  size_t *nobj)
{
    int h = dim[0], w = dim[1];
    int y = pos[0], x = pos[1];
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
void obj2color(Color *c, const MDSpace<2> &dim, const MDPoint<2> &pos,
               const Object **obj)
{
    int h = dim[0], w = dim[1];
    int y = pos[0], x = pos[1];
    int i = y * w + x;
    c[i].r = c[i].g = c[i].b = (obj[i]) ? 1.0f : 0.0f;
}
void pos2color(Color *c, const MDSpace<2> &dim, const MDPoint<2> &pos,
               const Point *hit, const Object **obj)
{
    int h = dim[0], w = dim[1];
    int y = pos[0], x = pos[1];
    int i = y * w + x;
    c[i].v.zero();
    if (obj[i]) c[i].r = c[i].g = c[i].b = fabs(2.7f - hit[i].z);
}

/*
// collect normal
void calc_normal(Normal *nr, const MDSpace<2> &dim, const MDPoint<2> &pos,
        const Point *pos, const Object **obj)
{
    int h = dim[0], w = dim[1];
    int y = pos[0], x = pos[1];
    int i = y * w + x;
    if (obj[i]) nr[i] = obj[i]->getNormal(pos[i]);
}

// collect wi
// count(wi) = count(nr) * count(light)
void calc_wi(Vector *wi, const MDSpace<2> &dim, const MDPoint<2> &pos,
        const Point *pos, const Light *light, size_t nlight)
{
    int h = dim[0], w = dim[1];
    int y = pos[0], x = pos[1];
    int i = y * w + x;
    wi = wi + nlight * i;
    for (int L = 0; L < nlight; ++L)
        wi[L] = light[L]->getDirection(pos[i]);
}

// brdf(n, wi, wo) -> f
// count(f) = count(wi)
void calc_f(Color *f, const MDSpace<2> &dim, const MDPoint<2> &pos,
        const Normal *nr, const Vector *wi, const Vector *wo, const Object
*object)
{
    int h = dim[0], w = dim[1];
    int y = pos[0], x = pos[1];
    int i = y * w + x;
    wi = wi + nlight * i;
    f = f + nlight * i;
    for (int L = 0; L < nlight; ++L)
    {
        f[L] = object[i]->bsdf->Sample_f(nr[i], wi[L], wo[i]);
    }
}

// f * L = color
void calc_color(Color *c, const MDSpace<2> &dim, const MDPoint<2> &pos,
        const Normal *nr, const Vector *wi,
        const Color *f, const Light *light, size_t nlight)
{
    int h = dim[0], w = dim[1];
    int y = pos[0], x = pos[1];
    int i = y * w + x;
    for (int L = 0; L < nlight; ++L)
        c[i] += f[i] * light[L]->L() * nr[i]->dot(wi[i]);
}
*/
