#pragma once

#include "mem.h"
#include "math.h"     // Math Library
#include "sampler.h"

#include <vector>
#include <fstream>
#include <sstream>
#include <string>

#include <cassert>

struct BBox
{
    float x0 = 1e10, y0 = 1e10, z0 = 1e10,
          x1 = -1e10, y1 = -1e10, z1 = -1e10;
    __device__ __host__ BBox() {}
    __device__ __host__ BBox (float x0_, float y0_, float z0_,
                              float x1_, float y1_, float z1_):
        x0(x0_), y0(y0_), z0(z0_),
        x1(x1_), y1(y1_), z1(z1_)
    {}
};


struct Color
{
    union
    {
        struct
        {
            float r, g, b;
        };
        struct
        {
            float x, y, z;
        };
        Vec3<float> v;
    };
    __host__ __device__ Color() : x(.0f), y(.0f), z(.0f)
    {}
    __host__ __device__ Color(float _x, float _y, float _z) : x(_x), y(_y), z(_z)
    {}
    __host__ __device__ Color(const Vec3<float> &_v) : v(_v)
    {}
};

struct Ray
{
    Vertex pos, dir;
    Color factor;
};

struct Camera
{
    Vertex pos, dir;
    float fov_h, fov_v; // radius
};

template <typename T>
struct RandomPicker
{
    T candidate[3];
    float cdf[3];

    __device__ T & pick(float p)
    {
        if (p <= cdf[0]) return candidate[0];
        else if (p <= cdf[1]) return candidate[1];
        else return candidate[2];
    }
};

// ---------------- Shape & Texture ----------------

typedef void Shape_t; // TODO: remove this

enum ShapeType
{
    SHAPE_SPHERE = 0, SHAPE_INVSPHERE = 1, SHAPE_RECT = 2, SHAPE_TRIA = 3
};

struct Texture
{
    size_t width, height, pixel;
    Color *tex;
};

// TODO: strategy -> ShapeType
struct Sphere
{
    int strategy;
    Vertex center;
    float radius;
};
struct Rectangle
{
    int strategy;
    Point pos;
    Vector a, b;
    Texture *tex;
};
struct Triangle
{
    int strategy;
    Point p1, p2, p3;
    UVPoint t1, t2, t3; // texture mapping
    Texture *tex;
};
union ShapeEntity
{
    struct Sphere sphere;
    struct Rectangle rectangle;
    struct Triangle triangle;
    ShapeEntity()
    {}
};

__device__ bool Intersect_ray2sphere(const void *ray, ShapeEntity *sphere, float *t)
{
    Ray &r = *(Ray *)ray;
    Sphere &s = *(Sphere *)sphere;

    Vertex op = s.center - r.pos;
    float eps = 1e-4;
    float b = op.dot(r.dir);
    float det = b * b - op.dot(op) + s.radius * s.radius;

    *t = 0.0f;
    if (det >= 0.0f)
    {
        det = sqrt(det);
        if (b - det > eps)
            *t = b - det;
        else if (b + det > eps)
            *t = b + det;
    }
    return *t != 0.0f;
}
__device__ bool Intersect_ray2rectangle(const void *ray, ShapeEntity *rectangle, float *t)
{
    Ray &r = *(Ray *)ray;
    struct Rectangle &rect = *(struct Rectangle *)rectangle;

    Normal nr = Vector::Cross(rect.a, rect.b).norm();
    float _t = (rect.pos - r.pos).dot(nr) / r.dir.dot(nr);
    if (_t < 1e-5f)
        return false;

    Point p = r.pos + Vector::Scale(r.dir, _t);
    Vector d = p - rect.pos;
    float alen2 = rect.a.dot(rect.a);
    float blen2 = rect.b.dot(rect.b);

    float ddota = d.dot(rect.a);
    if (ddota < 0.0 || ddota > alen2) return false;

    float ddotb = d.dot(rect.b);
    if (ddotb < 0.0 || ddotb > blen2) return false;

    *t = _t;
    return true;
}
__device__ bool Intersect_ray2triangle(const void *_ray, ShapeEntity *triangle, float *t)
{
    Ray &ray = *(Ray *)_ray;
    struct Triangle &tri = *(struct Triangle *)triangle;

    float a = tri.p1.x - tri.p2.x, b = tri.p1.x - tri.p3.x, c = ray.dir.x, d = tri.p1.x - ray.pos.x;
    float e = tri.p1.y - tri.p2.y, f = tri.p1.y - tri.p3.y, g = ray.dir.y, h = tri.p1.y - ray.pos.y;
    float i = tri.p1.z - tri.p2.z, j = tri.p1.z - tri.p3.z, k = ray.dir.z, l = tri.p1.z - ray.pos.z;

    float m = f * k - g * j, n = h * k - g * l, p = f * l - h * j;
    float q = g * i - e * k, s = e * j - f * i;

    float inv_denom = 1.0f / (a * m + b * q + c * s);

    float e1 = d * m - b * n - c * p;
    float beta = e1 * inv_denom;

    if (beta < 0)
        return false;

    float r = e * l - h * i;
    float e2 = a * n + d * q + c * r;
    float gamma = e2 * inv_denom;

    if (gamma < 0 || beta + gamma > 1) return false;

    float e3 = a * p - b * r + d * s;
    float _t = e3 * inv_denom;

    if (_t < 1e-5f) return false;

    *t = _t;
    return true;
}
typedef bool(*intersect_fun_t)(const void *, ShapeEntity *, float *);
__device__ intersect_fun_t IntersectStrategy[] = {
    Intersect_ray2sphere,
    Intersect_ray2sphere,
    Intersect_ray2rectangle,
    Intersect_ray2triangle
};

__device__ void Normal_sphere(ShapeEntity *sphere, void *pos, void *normal)
{
    Sphere &s = *(Sphere *)sphere;
    Point &p = *(Point *)pos;
    Normal &nr = *(Normal *)normal;
    nr = (p - s.center).norm();
}
__device__ void Normal_sphere2(ShapeEntity *sphere, void *pos, void *normal)
{
    Sphere &s = *(Sphere *)sphere;
    Point &p = *(Point *)pos;
    Normal &nr = *(Normal *)normal;
    nr = (s.center - p).norm();
}
__device__ void Normal_rectangle(ShapeEntity *rectangle, void *pos, void *normal)
{
    struct Rectangle &rect = *(struct Rectangle *)rectangle;
    Point &p = *(Point *)pos;
    Normal &nr = *(Normal *)normal;
    nr = Vector::Cross(rect.a, rect.b).norm();
}
__device__ void Normal_triangle(ShapeEntity *triangle, void *pos, void *normal)
{
    struct Triangle &tri = *(struct Triangle *)triangle;
    Point &p = *(Point *)pos;
    Normal &nr = *(Normal *)normal;
    Vector a = tri.p2 - tri.p1, b = tri.p3 - tri.p1;
    nr = Vector::Cross(a, b).norm();
}
typedef void(*normal_fun_t)(ShapeEntity *, void *, void *);
__device__ normal_fun_t NormalStrategy[] = {
    Normal_sphere,
    Normal_sphere2,
    Normal_rectangle,
    Normal_triangle
};

__device__ Color Shader_simple(ShapeEntity *shape, void *pos, void *normal, Color *factor)
{
    return *factor;
}
__device__ Color Shader_rectangle(ShapeEntity *shape, void *pos, void *normal, Color *factor)
{
    return *factor;
    //Point &p = *(Point *)pos;
    //struct Rectangle &rect = shape->rectangle;
    //if (!rect.tex) return *factor;

    //int x = 100.0f * (p - rect.pos).dot(rect.a) / rect.a.dot(rect.a);
    //int y = 100.0f * (p - rect.pos).dot(rect.b) / rect.b.dot(rect.b);
    //x = max(0, min(99, x));
    //y = max(0, min(99, y));
    //return Vector::Mul(rect.tex[100 * y + x].v, factor->v);
}
__device__ Color Shader_triangle(ShapeEntity *shape, void *pos, void *normal, Color *factor)
{
    Point &p = *(Point *)pos;
    struct Triangle &tri = shape->triangle;
    if (!tri.tex) return *factor;

    Vector p12 = tri.p2 - tri.p1;
    Vector p13 = tri.p3 - tri.p1;
    Vector p23 = tri.p3 - tri.p2;

    float area2 = Vector::Cross(p12, p13).magnitude();
    float u = Vector::Cross(p12, p - tri.p1).magnitude() / area2;
    float v = Vector::Cross(p23, p - tri.p2).magnitude() / area2;
    float w = 1.0f - u - v;

    UVPoint mp = Vector(tri.t1).scale(v) + Vector(tri.t2).scale(w) + Vector(tri.t3).scale(u);
    int x = tri.tex->width * mp.x;
    int y = tri.tex->height * mp.y;
    x = max(0, min((int)tri.tex->width - 1, x));
    y = max(0, min((int)tri.tex->height - 1, y));
    return Vector::Mul(tri.tex->tex[tri.tex->width * y + x].v, factor->v);
}
typedef Color(*shader_fun_t)(ShapeEntity *, void *, void *, Color *);
__device__ shader_fun_t ShaderStrategy[] = {
    Shader_simple,
    Shader_simple,
    Shader_rectangle,
    Shader_triangle
};

struct HitParam
{
    float t;
    bool is_hit;
};
struct ComputeHit
{
    HitParam param;
    __device__ inline void compute(const Ray *ray, ShapeEntity *shape)
    {
        int strategy = *(int *)shape;
        param.is_hit = IntersectStrategy[strategy](ray, shape, &param.t);
    }
    __device__ inline bool isHit() const
    {
        return param.is_hit;
    }
    __device__ inline float t() const
    {
        return param.t;
    }
};

typedef size_t shape_handle_t;

class BBox_Factory
{
    BBox bbox;
    std::vector<BBox> object_bboxs;
public:
    BBox_Factory():
        bbox(),
        object_bboxs()
    {}

    void update(void *shape)
    {
        BBox obj_bbox = calculate_bbox(shape);
        object_bboxs.push_back(obj_bbox);
        if (obj_bbox.x0 < bbox.x0) bbox.x0 = obj_bbox.x0;
        if (obj_bbox.y0 < bbox.y0) bbox.y0 = obj_bbox.y0;
        if (obj_bbox.z0 < bbox.z0) bbox.z0 = obj_bbox.z0;
        if (obj_bbox.x1 > bbox.x1) bbox.x1 = obj_bbox.x1;
        if (obj_bbox.y1 > bbox.y1) bbox.y1 = obj_bbox.y1;
        if (obj_bbox.z1 > bbox.z1) bbox.z1 = obj_bbox.z1;
    }

    BBox& getBBox()
    {
        return bbox;
    }

    std::vector<BBox>& getObjectBBoxs()
    {
        return object_bboxs;
    }

    static __device__ __host__ BBox calculate_bbox(void *shape)
    {
        switch (*(int *)shape)
        {
            case 0:
            case 1:
                {
                    Sphere &s = *(Sphere *)shape;

                    float dist = sqrtf(3 * s.radius * s.radius);
                    return{s.center.x - dist, s.center.y - dist, s.center.z - dist,
                        s.center.x + dist, s.center.y + dist, s.center.z + dist};
                }
            case 2:
                {
                    struct Rectangle &r = *(struct Rectangle *)shape;
                    Point p0 = r.pos;
                    Point p1 = r.pos + r.a;
                    Point p2 = r.pos + r.b;
                    Point p3 = p1 + r.b;

                    return{
                        fminf(fminf(p0.x, p1.x), fminf(p2.x, p3.x)),
                        fminf(fminf(p0.y, p1.y), fminf(p2.y, p3.y)),
                        fminf(fminf(p0.z, p1.z), fminf(p2.z, p3.z)),
                        fmaxf(fmaxf(p0.x, p1.x), fmaxf(p2.x, p3.x)),
                        fmaxf(fmaxf(p0.y, p1.y), fmaxf(p2.y, p3.y)),
                        fmaxf(fmaxf(p0.z, p1.z), fmaxf(p2.z, p3.z))
                    };
                }
            default:
                return BBox();
        }
    }
};

__constant__ ShapeEntity const_shape[100];
class Shape_Factory
{
    ConstVectorPool<ShapeEntity, 100> shapes;
public:
    Shape_Factory()
    {
        ShapeEntity *device_p;
        CheckCUDAError(cudaGetSymbolAddress((void **)&device_p, const_shape));
        shapes.setDevice(device_p);
    }
    shape_handle_t createShape(const ShapeEntity &s)
    {
        shapes.add(s);
        return shapes.getSize();
    }
    shape_handle_t createSphere(Vertex center, float radius)
    {
        ShapeEntity s;
        s.sphere = {0, center, radius};
        shapes.add(s);
        return shapes.getSize();
    }
    shape_handle_t createRectangle(Point pos, Vector a, Vector b)
    {
        ShapeEntity s;
        s.rectangle = {1, pos, a, b, nullptr};
        shapes.add(s);
        return shapes.getSize();
    }
    shape_handle_t createTriangle(Point a, Point b, Point c)
    {
        ShapeEntity s;
        s.triangle = {2, a, b, c, };
        shapes.add(s);
        return shapes.getSize();
    }
    ShapeEntity * getHost(shape_handle_t handle)
    {
        if (handle >= shapes.getSize()) return nullptr;
        else return shapes.getHost() + handle;
    }
    ShapeEntity * getDevice(shape_handle_t handle)
    {
        if (handle >= shapes.getSize()) return nullptr;
        else return shapes.getDevice() + handle;
    }
    void syncToDevice()
    {
        CheckCUDAError(cudaMemcpyToSymbol(const_shape,
                                          shapes.getHost(),
                                          shapes.getSize() * sizeof(ShapeEntity)));
        //shapes.syncToDevice();
    }
};

typedef size_t tex_handle_t;

class Texture_Factory
{
    VectorPool<Texture> textures;
    std::vector<Pool<Color>> texdata;
public:
    tex_handle_t createTexture(const char *ppm)
    {
        Texture t;

        std::ifstream in(ppm);
        in.ignore(3);
        in >> t.width >> t.height >> t.pixel;
        t.tex = nullptr;

        size_t size = t.width * t.height;
        texdata.emplace_back(size, IN_HOST | IN_DEVICE);

        Color *color = texdata.back().getHost();
        Color c;
        for (size_t i = 0; i < size; ++i)
        {
            in >> c.r >> c.g >> c.b;
            c.v.scale(1.0f / 255.0f);
            color[i] = c;
        }
        texdata.back().copyToDevice();

        t.tex = texdata.back().getDevice();
        textures.add(t);
        return textures.getSize();
    }
    void syncToDevice()
    {
        textures.syncToDevice();
    }
    Texture * getDevice(tex_handle_t handle)
    {
        if (handle >= textures.getSize()) return nullptr;
        else return textures.getDevice() + handle;
    }
};

// ---------------- BSDF ----------------

struct BSDFParam
{
    Normal nr;
    Vector wo;
    float u1, u2;

    Vector wi;
    Color f;
    float pdf;
};

// BSDF models
enum BSDFType
{
    LAMBERTIAN = 0, SPEC_REFL = 1, SPEC_TRANS = 2
};

struct Lambertian
{
    int strategy;
    Color R;
};
struct SpecularReflection
{
    int strategy;
    Color R;
};
struct SpecularTransmission
{
    int strategy;
    Color T;
};
union BSDFEntity
{
    Lambertian diff;
    SpecularReflection refl;
    SpecularTransmission trans;
    BSDFEntity()
    {}
};

// BSDF strategies
typedef void(*bsdf_fun_t)(BSDFParam *, const void *);
__device__ void BSDF_Lambertian(BSDFParam *param, const void *_model)
{
    const Lambertian &model = *(const Lambertian *)_model;
    param->wi = UniformSampleHemisphere(param->u1, param->u2);
    if (param->wi.dot(param->nr) < 0.0f) param->wi = -param->wi;
    param->f.v = Vector::Scale(model.R.v, 1.0f / 3.14159f);
    param->pdf = 1.0f / 3.14159f;
}
__device__ void BSDF_SpecRefl(BSDFParam *param, const void *_model)
{
    const SpecularReflection &model = *(const SpecularReflection *)_model;
    Vector nr = param->nr;
    Vector wo = param->wo;
    nr.norm();
    param->wi = Vector(nr).scale(wo.dot(nr)).scale(2.0f).sub(wo).norm();
    param->f.v = model.R.v;
    param->pdf = 1.0f;
}
__device__ void BSDF_SpecTrans(BSDFParam *param, const void *_model)
{
    const SpecularTransmission &model = *(const SpecularTransmission *)_model;
    Vector nr = param->nr;
    Vector wo = param->wo;
    nr.norm();

    bool into = wo.dot(nr) > 0.0f;
    float nc = 1.0f;
    float nt = 1.5f;
    float nnt = into ? nc / nt : nt / nc;
    float ddn = wo.dot(into ? -nr : nr);
    float cos2t = 1 - nnt*nnt*(1 - ddn*ddn);
    if (cos2t < 0) // Total internal reflection
    {
        param->wi = Vector(nr).scale(wo.dot(nr)).scale(2.0f).sub(wo).norm();
        param->f.v = model.T.v;
        param->pdf = 1.0f;
    }
    else
    {
        param->wi = (wo.scale(-nnt) - nr.scale((into ? 1.0f : -1.0f)*(ddn*nnt + sqrt(cos2t)))).norm();
        param->f.v = model.T.v;
        param->pdf = 1.0f;
    }
}
__device__ bsdf_fun_t BSDFStrategy[] = {
    BSDF_Lambertian,
    BSDF_SpecRefl,
    BSDF_SpecTrans
};

struct ComputeBSDF
{
    BSDFParam param;

    __device__ inline void compute(BSDFEntity *bsdf)
    {
        //BSDFStrategy[strategy](&param, bsdf);
        int strategy = *(int *)bsdf;
        if (strategy == 0)
            BSDF_Lambertian(&param, bsdf);
        else if (strategy == 1)
            BSDF_SpecRefl(&param, bsdf);
        else
            BSDF_SpecTrans(&param, bsdf);
    }
    __device__ inline Color f() const
    {
        return param.f;
    }
    __device__ inline float pdf() const
    {
        return param.pdf;
    }
    __device__ inline Vector wi() const
    {
        return param.wi;
    }
};

typedef size_t bsdf_handle_t;
typedef RandomPicker<bsdf_handle_t> bsdf_picker;

__constant__ BSDFEntity const_bsdf[100];
__constant__ bsdf_picker const_bsdf_picker[100];
class BSDF_Factory
{
    ConstVectorPool<BSDFEntity, 100> models;
    ConstVectorPool<bsdf_picker, 100> pickers;
public:
    BSDF_Factory()
    {
        void *device_p;
        CheckCUDAError(cudaGetSymbolAddress((void **)&device_p, const_bsdf));
        models.setDevice((BSDFEntity *)device_p);
        CheckCUDAError(cudaGetSymbolAddress((void **)&device_p, const_bsdf_picker));
        pickers.setDevice((bsdf_picker *)device_p);
    }
    bsdf_handle_t createLambertian(Color R)
    {
        BSDFEntity model;
        model.diff = {LAMBERTIAN, R};
        models.add(model);
        return models.getSize();
    }
    bsdf_handle_t createSpecRefl(Color R)
    {
        BSDFEntity model;
        model.refl = {SPEC_REFL, R};
        models.add(model);
        return models.getSize();
    }
    bsdf_handle_t createSpecTrans(Color T)
    {
        BSDFEntity model;
        model.trans = {SPEC_REFL, T};
        models.add(model);
        return models.getSize();
    }
    bsdf_handle_t createPicker(bsdf_handle_t models[3], float cdf[3])
    {
        bsdf_picker picker = {
            {models[0], models[1], models[2]},
            {cdf[0], cdf[1], cdf[2]}
        };
        pickers.add(picker);
        return pickers.getSize();
    }
    void syncToDevice()
    {
        CheckCUDAError(cudaMemcpyToSymbol(const_bsdf,
                                          models.getHost(),
                                          models.getSize() * sizeof(BSDFEntity)));
        CheckCUDAError(cudaMemcpyToSymbol(const_bsdf_picker,
                                          pickers.getHost(),
                                          pickers.getSize() * sizeof(bsdf_picker)));
        //models.syncToDevice();
        //pickers.syncToDevice();
    }
    BSDFEntity * getDevice_bsdf(bsdf_handle_t handle = 0u)
    {
        if (handle >= models.getSize()) return nullptr;
        else return models.getDevice() + handle;
    }
    bsdf_picker * getDevice_picker(bsdf_handle_t handle = 0u)
    {
        if (handle >= pickers.getSize()) return nullptr;
        else return pickers.getDevice() + handle;
    }
};

// ---------------- Light ----------------

enum LightType
{
    LIGHT_POINT
};

struct PointLight
{
    int strategy;
    Color intensity;
};
union LightEntity
{
    PointLight pointlight;
    LightEntity()
    {}
};

struct LightParam
{
    Color L;
};

typedef void(*light_fun_t)(LightParam &param, const LightEntity *light);
__device__ void L_pointlight(LightParam &param, const LightEntity *light)
{
    param.L = light->pointlight.intensity;
}
__device__ light_fun_t LightStategy[] = {
    L_pointlight
};

struct ComputeLight
{
    LightParam param;

    __device__ inline void compute(const LightEntity *light)
    {
        L_pointlight(param, light);
    }
    __device__ inline Color L() const
    {
        return param.L;
    }
};

typedef size_t light_handle_t;

__constant__ LightEntity const_light[10];
class Light_Factory
{
    ConstVectorPool<LightEntity, 10> lights;
public:
    Light_Factory()
    {
        void *device_p;
        CheckCUDAError(cudaGetSymbolAddress((void **)&device_p, const_light));
        lights.setDevice((LightEntity *)device_p);
    }
    light_handle_t createPointLight(Color L)
    {
        LightEntity l;
        l.pointlight = {LIGHT_POINT, L};
        lights.add(l);
        return lights.getSize();
    }
    void syncToDevice()
    {
        CheckCUDAError(cudaMemcpyToSymbol(const_light,
                                          lights.getHost(),
                                          lights.getSize() * sizeof(LightEntity)));
        //lights.syncToDevice();
    }
    LightEntity * getDevice(light_handle_t handle)
    {
        if (handle >= lights.getSize()) return nullptr;
        else return lights.getDevice() + handle;
    }
};

// ---------------- Object ----------------

struct Object
{
    ShapeEntity *shape;
    bsdf_picker *bsdf;
    LightEntity *light;
};

__constant__ Object const_objects[500];
class Object_Factory
{
    ConstVectorPool<Object, 500> objects;

public:
    Object_Factory()
    {
        void *device_p;
        CheckCUDAError(cudaGetSymbolAddress((void **)&device_p, const_objects));
        objects.setDevice((Object *)device_p);
    }
    void createObject(const Object &o)
    {
        objects.add(o);
    }
    
    void syncToDevice()
    {
        CheckCUDAError(cudaMemcpyToSymbol(const_objects,
                                          objects.getHost(),
                                          objects.getSize() * sizeof(Object)));
        //objects.syncToDevice();
    }
    Object * getHost()
    {
        return objects.getHost();
    }
    Object * getDevice()
    {
        return objects.getDevice();
    }
    
    size_t getSize() const
    {
        return objects.getSize();
    }
};

// ---------------- Scene ----------------

/*
    Scene File:
    [bsdf]
    <type id>   <params>
    0           0.75 0.75 0.75
    0           0.25 0.25 0.25

    [bsdf-picker]
    <bsdf id list>  <bsdf probability>
    0 1 2           1.0 0.0 0.0

    [shape]
    <type id>   <params>
    0           1.0 2.0 3.0 1000

    [light]
    <type id>   <params>
    0           12.0 12.0 12.0

    [texture]
    <file>
    grid.tex
    red.png

    [texture-mapping] // only triangles support texture
    <triangle id> <texture id> <t0> <t1> <t2>

    [object] // id = id + 1
    <shape id>  <bsdf-picker id>    <light id>
*/

class Scene
{
    BSDF_Factory bsdf_factory;
    Shape_Factory shape_factory;
    Light_Factory light_factory;
    Texture_Factory texture_factory;
    Object_Factory object_factory;
    Pool<Camera> *_camera;
    BBox_Factory bbox_factory;
public:
    Scene() : _camera(nullptr)
    {}
    BSDF_Factory & bsdf()
    {
        return bsdf_factory;
    }
    Shape_Factory & shape()
    {
        return shape_factory;
    }
    Light_Factory & light()
    {
        return light_factory;
    }
    Texture_Factory & texture()
    {
        return texture_factory;
    }
    Object_Factory & object()
    {
        return object_factory;
    }
    Pool<Camera> * &camera()
    {
        return _camera;
    }
    BBox_Factory bbox()
    {
        return bbox_factory;
    }

    void load(const char *file);
};


template <typename T>
std::istream  & operator >> (std::istream &in, Vec3<T> &v)
{
    in >> v.x >> v.y >> v.z;
    return in;
}
std::istream  & operator >> (std::istream &in, ShapeEntity &s)
{
    in >> s.sphere.strategy;
    switch (s.sphere.strategy)
    {
        case SHAPE_INVSPHERE:
        case SHAPE_SPHERE:
            in >> s.sphere.center
                >> s.sphere.radius;
            break;
        case SHAPE_RECT:
            in >> s.rectangle.pos
                >> s.rectangle.a
                >> s.rectangle.b;
            s.rectangle.tex = nullptr;
            break;
        case SHAPE_TRIA:
            in >> s.triangle.p1
                >> s.triangle.p2
                >> s.triangle.p3;
            s.triangle.tex = nullptr;
            break;
    }
    return in;
}
