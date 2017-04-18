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

// #include "BRDF.h"  // Material: BSDF
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
        : dim({h, w}),
          color_buffer(nullptr),
          ray_buffer(nullptr),
          obj_buffer(nullptr),
          pos_buffer(nullptr)
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
        // ComputeEngine::Dispatch(CPU, kernel_simple, color_buffer, dim,
        // &scene, &cam);
        fprintf(stderr, "shoot_ray()...\n");
        ComputeEngine::Dispatch(CPU, shoot_ray, ray_buffer, dim, &cam);
        fprintf(stderr, "\ncalc_obj_pos()...\n");
        size_t nobj = scene.objs.size();
        ComputeEngine::Dispatch(CPU, calc_obj_pos, obj_buffer, pos_buffer, dim,
                                ray_buffer, scene.objs.data(),
                                &nobj);
        // debug
        ComputeEngine::Dispatch(CPU, ray2color, color_buffer, dim, ray_buffer);
        display("ray.ppm");
        ComputeEngine::Dispatch(CPU, obj2color, color_buffer, dim, obj_buffer);
        ComputeEngine::Dispatch(CPU, pos2color, color_buffer, dim, pos_buffer, obj_buffer);
        display("hit.ppm");
    }
    void display(const char *filename)
    {
        int h = dim[0], w = dim[1];
        Color *c = color_buffer;
        FILE *f;
        f = fopen(filename, "w");
        fprintf(f, "P3\n%d %d\n%d\n", w, h, 255);
        // Scale
        Color maxv = {0.0f, 0.0f, 0.0f};
        for (size_t i = 0; i < w * h; ++i)
        {
            maxv.r = fmax(maxv.r, c[i].r); maxv.g = fmax(maxv.g, c[i].g); maxv.b = fmax(maxv.b, c[i].b);
        }
        for (size_t i = 0; i < w * h; ++i)
        {
            size_t p = (w - i % w - 1) + w * (h - i / w - 1);
            fprintf(f, "%d %d %d ",
                    (int)(clamp(c[p].r / maxv.r) * 255),
                    (int)(clamp(c[p].g / maxv.g) * 255),
                    (int)(clamp(c[p].b / maxv.b) * 255));
        }
        // Pow
        // for (size_t i = 0; i < w * h; ++i)
        // {
        //     size_t p = (w - i % w - 1) + w * (h - i / w - 1);
        //     fprintf(f, "%d %d %d ",
        //             (int)(pow(clamp(c[p].r), 1 / 2.2) * 255 + .5),
        //             (int)(pow(clamp(c[p].g), 1 / 2.2) * 255 + .5),
        //             (int)(pow(clamp(c[p].b), 1 / 2.2) * 255 + .5));
        // }
    }

   private:
    MDSpace<2> dim;
    Color *color_buffer;
    Ray *ray_buffer;
    const Object **obj_buffer;
    Point *pos_buffer;
    // Normal *nr_buffer;
    // Vector *wi_buffer;
    // Vector *wo_buffer;
};

void A_Ball(Scene *scene, Camera *cam)
{
    Sphere ball(1.0f, {0.0f, 0.0f, 3.0f});

    Object o = {new Sphere(ball)};
    scene->objs.push_back(o);

    Camera c = {
        {0.0f, 0.0f, 0.0f}, {0.0f, 0.0f, 1.0f}, 0.5 * M_PI, 0.5 * M_PI};
    *cam = c;
}
/*
void Cornell_Box(Scene *scene, Camera *cam)
{
    DiffuseReflection grey({0.25, 0.25f, 0.25f});
    DiffuseReflection red({0.75, 0.25, 0.25});
    DiffuseReflection green({0.1f, 1.0f, 0.1f});
    DiffuseReflection blue({0.25, 0.25, 0.75});
    Sphere left(999.0f, {-1000.0f, -3.0f, 5.0f});
    Sphere right(999.0f, {1000.0f, 3.0f, 5.0f});
    Sphere up(999.0f, {0.0f, -1000.0f, 5.0f});
    Sphere bottom(999.0f, {0.0f, 1000.0f, 5.0f});
    Sphere back(999.0f, {0.0f, 0.0f, 1001.0f});

    Object objs[] = {{&left, &red},
                     {&right, &blue},
                     {&up, &grey},
                     {&bottom, &grey},
                     {&back, &grey}};

    PointLight lights[] = {{{0.0f, 0.0f, 0.0f}, {1.0f, 1.0f, 1.0f}}};
    scene->objs.clear();
    scene->lights.clear();
    for (Object &o : objs)
    {
        Object *op = new Object();
        op->shape = new Sphere(*(Sphere *)o.shape);
        op->bsdf = new DiffuseReflection(*(DiffuseReflection *)o.bsdf);
        scene->objs.push_back(op);
    }
    for (PointLight &l : lights)
    {
        scene->lights.push_back(new PointLight(l));
    }

    Camera c = {{0.0f, 0.0f, 0.0f}, {0.0f, 0.0f, 1.0f}, 0.66f * M_PI, 0.5 *
M_PI};
    *cam = c;
}
*/

int main(int argc, char *argv[])
{
    Scene scene;
    Camera cam;

    // Cornell_Box(&scene, &cam);
    A_Ball(&scene, &cam);

    // create scene
    Render render(512, 512);
    // rendering loop
    render.update(scene, cam);
    // render.display();

    return 0;
};
