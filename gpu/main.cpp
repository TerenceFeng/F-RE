// ------- Common --------
#include "compute.h"  // Compute-Engine Subsystem
#include "math.h"     // Math Library
typedef Vec3<float> Vertex;
typedef Vec3<float> Vector;
typedef Vec3<float> Point;
typedef Vec3<float> Normal;

// -------- Coordinate Subsystem -------
// #include "coordinate.h"

// -------- Visual Subsystem -------

/* common */
struct Ray
{
    Vertex pos, dir;
};

struct Color
{
    union {
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
    Color() : x(.0f), y(.0f), z(.0f)
    {
    }
    Color(float _x, float _y, float _z) : x(_x), y(_y), z(_z)
    {
    }
    Color(const Vec3<float> &_v) : v(_v)
    {
    }
};

#include "BRDF.h"  // Material: BSDF
// Texture
#include "light.h"
#include "shape.h"

// -------- Scene -------
#include "scene.h"

// ## debug helper ##
#include "compute_kernels.h"
double clamp(double d)
{
    return d > 1.0 ? 1.0 : (d < 0.0 ? 0.0 : d);
}

// ------- Renderer -------
class Render
{
   public:
    Render(size_t w, size_t h)
        : dim({w, h}),
          color_buffer(nullptr),
          ray_buffer(nullptr),
          obj_buffer(nullptr),
          pos_buffer(nullptr),
          nr_buffer(nullptr),
          wi_buffer(nullptr),
          wo_buffer(nullptr),
          f_buffer(nullptr)
    {
    }
    void update(const Scene &scene, const Camera &cam)
    {
        if (!color_buffer)
            color_buffer = new Color[dim[0] * dim[1]];
        if (!ray_buffer)
            ray_buffer = new Ray[dim[0] * dim[1]];
        if (!obj_buffer)
            obj_buffer = new const Object *[dim[0] * dim[1]];
        if (!pos_buffer)
            pos_buffer = new Point[dim[0] * dim[1]];
        if (!nr_buffer)
            nr_buffer = new Normal[dim[0] * dim[1]];
        if (!wi_buffer)
            wi_buffer = new Vector[dim[0] * dim[1] * scene.lights.size()];
        if (!wo_buffer)
            wo_buffer = new Vector[dim[0] * dim[1]];
        if (!f_buffer)
            f_buffer = new Color[dim[0] * dim[1] * scene.lights.size()];

        size_t nobj = scene.objs.size();
        size_t nlight = scene.lights.size();

        fprintf(stderr, "shoot_ray()...\n");
        ComputeEngine::Dispatch(CPU, shoot_ray, ray_buffer, dim, &cam);
        fprintf(stderr, "\ncalc_obj_pos()...\n");
        ComputeEngine::Dispatch(CPU, calc_obj_pos, obj_buffer, pos_buffer, dim,
                                ray_buffer, scene.objs.data(), &nobj);
        fprintf(stderr, "\ncalc_normal()...\n");
        ComputeEngine::Dispatch(CPU, calc_normal, nr_buffer, dim, pos_buffer,
                                obj_buffer);
        fprintf(stderr, "\ncalc_wi()...\n");
        ComputeEngine::Dispatch(CPU, calc_wi, wi_buffer, dim, pos_buffer,
                                obj_buffer, (const Light **)scene.lights.data(),
                                &nlight);
        fprintf(stderr, "\ncalc_wo()...\n");
        ComputeEngine::Dispatch(CPU, calc_wo, nr_buffer, dim, ray_buffer);
        fprintf(stderr, "\ncalc_f()...\n");
        ComputeEngine::Dispatch(CPU, calc_f, f_buffer, dim, nr_buffer,
                                wi_buffer, wo_buffer, obj_buffer, &nlight);
        fprintf(stderr, "\ncalc_color()...\n");
        ComputeEngine::Dispatch(CPU, calc_color, color_buffer, dim,
        nr_buffer,
                                wi_buffer, pos_buffer, f_buffer,
                                (const Light **)scene.lights.data(),
                                &nlight);
        display("xxx.ppm");

        // debug
        // ComputeEngine::Dispatch(CPU, ray2color, color_buffer, dim,
        // ray_buffer);
        // display("ray.ppm");
        // ComputeEngine::Dispatch(CPU, obj2color, color_buffer, dim,
        // obj_buffer);
        // display("object.ppm");
        // ComputeEngine::Dispatch(CPU, pos2color, color_buffer, dim,
        // pos_buffer, obj_buffer);
        // display("hit.ppm");
        // ComputeEngine::Dispatch(CPU, nr2color, color_buffer, dim, nr_buffer,
        //                         obj_buffer);
        // display("normal.ppm");

        // char name[] = "wi_.ppm";
        // for (size_t i = 0; i < nlight; ++i)
        // {
        //     ComputeEngine::Dispatch(CPU, wi2color, color_buffer, dim, wi_buffer,
        //                             obj_buffer, &nlight, &i);
        //     name[2] = i + '0';
        //     display(name);
        // }

        // size_t n = 2;
        // ComputeEngine::Dispatch(CPU, f2color, color_buffer, dim, f_buffer, &nlight, &n);
        // display("f.ppm");
    }
    void display(const char *filename)
    {
        int h = dim[0], w = dim[1];
        Color *c = color_buffer;
        FILE *f;
        f = fopen(filename, "w");
        fprintf(f, "P3\n%d %d\n%d\n", w, h, 255);
        // for (size_t i = 0; i < w * h; ++i)
        // {
        //     size_t p = (w - i % w - 1) + w * (h - i / w - 1);
        //     fprintf(f, "%d %d %d ",
        //             (int)(clamp(c[p].r) * 255),
        //             (int)(clamp(c[p].g) * 255),
        //             (int)(clamp(c[p].b) * 255));
        // }
        // Scale
        float maxv = 1e-10f;
        for (size_t i = 0; i < w * h; ++i)
        {
            maxv = fmax(maxv, c[i].r);
            maxv = fmax(maxv, c[i].g);
            maxv = fmax(maxv, c[i].b);
        }
        for (size_t i = 0; i < w * h; ++i)
        {
            size_t p = (i % w) + w * (h - 1 - i / w);
            fprintf(f, "%d %d %d ",
                    (int)(clamp(c[p].r / maxv) * 255),
                    (int)(clamp(c[p].g / maxv) * 255),
                    (int)(clamp(c[p].b / maxv) * 255));
        }
        // Pow
        // for (size_t i = 0; i < w * h; ++i)
        // {
        //     size_t p = (w - i % w - 1) + w * (h - i / w - 1);
        //     fprintf(f, "%d %d %d ",
        //             (int)(pow(clamp(c[p].r / maxv.r), 1 / 2.2) * 255 + .5),
        //             (int)(pow(clamp(c[p].g / maxv.g), 1 / 2.2) * 255 + .5),
        //             (int)(pow(clamp(c[p].b / maxv.b), 1 / 2.2) * 255 + .5));
        // }
    }

   private:
    MDSpace<2> dim;
    Color *color_buffer;
    Ray *ray_buffer;
    const Object **obj_buffer;
    Point *pos_buffer;
    Normal *nr_buffer;
    Vector *wi_buffer;
    Vector *wo_buffer;
    Color *f_buffer;
};

void A_Ball(Scene *scene, Camera *cam)
{
    DiffuseReflection green({0.8f, 0.8f, 0.8f});
    Sphere ball(1.0f, {0.0f, 0.0f, 3.0f});

    Object o = {new Sphere(ball), new DiffuseReflection(green)};
    scene->objs.push_back(o);

    scene->lights.push_back(
        new PointLight({-2.0f, 1.0f, -1.0f}, {1.0f, 0.0f, 0.0f}));
    scene->lights.push_back(
        new PointLight({2.0f, 1.0f, -1.0f}, {0.0f, 1.0f, 0.0f}));
    scene->lights.push_back(
        new PointLight({0.0f, -2.0f, -1.0f}, {0.0f, 0.0f, 1.0f}));

    Camera c = {{0.0f, 0.0f, 0.0f}, {0.0f, 0.0f, 1.0f}, 0.5 * M_PI, 0.5 * M_PI};
    *cam = c;
}
void Cornell_Box(Scene *scene, Camera *cam)
{
    DiffuseReflection grey({0.75f, 0.75f, 0.75f});
    DiffuseReflection green({0.1f, 1.0f, 0.1f});
    DiffuseReflection red({1.0f, 0.0f, 0.0f});
    DiffuseReflection blue({0.0f, 0.0f, 1.0f});
    Sphere left(999.0f, {-1000.0f, -3.0f, 5.0f});
    Sphere right(999.0f, {1000.0f, 3.0f, 5.0f});
    Sphere up(999.0f, {0.0f, 1000.0f, 5.0f});
    Sphere bottom(999.0f, {0.0f, -1000.0f, 5.0f});
    Sphere back(999.0f, {0.0f, 0.0f, 1001.0f});

    DiffuseReflection ball_color({1.0f, 1.0f, 1.0f});
    Sphere ball(0.3f, {0.0f, 0.0f, 0.95f});

    Object objs[] = {
        {&left, &red},
        {&right, &blue},
        {&up, &grey},
        {&bottom, &grey},
        {&back, &grey},
        // {&ball, &grey}
    };
    scene->objs.clear();
    for (Object &o : objs)
    {
        Object _o;
        _o.shape = new Sphere(*(Sphere *)o.shape);
        _o.bsdf = new DiffuseReflection(*(DiffuseReflection *)o.bsdf);
        scene->objs.push_back(_o);
    }

    PointLight lights[] = {
        {{-0.4f, -0.4f, 0.95f}, {1.0f, 0.9f, 1.0f}},
        {{0.4f, 0.4f, 0.95f}, {0.8f, 1.0f, 0.8f}},
        {{0.0f, 0.0f, -1.0f}, {1.0f, 1.0f, 1.0f}}
    };
    scene->lights.clear();
    for (PointLight &l : lights)
    {
        scene->lights.push_back(new PointLight(l));
    }

    Camera c = {{0.0f, 0.0f, 0.0f}, {0.0f, 0.0f, 1.0f}, 0.66 * M_PI, 0.66 * M_PI};
    *cam = c;
}

int main(int argc, char *argv[])
{
    Scene scene;
    Camera cam;

    Cornell_Box(&scene, &cam);
    // A_Ball(&scene, &cam);

    // create scene
    Render render(512, 512);
    // rendering loop
    render.update(scene, cam);

    return 0;
};
