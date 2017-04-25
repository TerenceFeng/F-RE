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

struct Ray
{
    Vertex pos, dir;
    Color factor;
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
          ray2_buffer(nullptr)
    {
    }
    void update(const Scene &scene, const Camera &cam, int sample, int depth)
    {
        if (!color_buffer)
            color_buffer = new Color[dim[0] * dim[1]];
        if (!ray_buffer)
            ray_buffer = new Ray[dim[0] * dim[1]];
        if (!ray2_buffer)
            ray2_buffer = new Ray[dim[0] * dim[1]];
        size_t nobj = scene.objs.size();
        for (int s = 0; s < sample; ++s)
        {
            ComputeEngine::Dispatch(CPU, init_ray, ray_buffer, dim, &cam);
            for (int i = 1; i <= depth; ++i)
            {
                ComputeEngine::Dispatch(CPU, trace_ray, ray2_buffer, color_buffer, dim,
                        ray_buffer, scene.objs.data(), &nobj);
                std::swap(ray_buffer, ray2_buffer);
            }
        }
        display("xxx.ppm");
    }
    void display(const char *filename)
    {
        int h = dim[0], w = dim[1];
        Color *c = color_buffer;
        FILE *f;
        f = fopen(filename, "w");
        fprintf(f, "P3\n%d %d\n%d\n", w, h, 255);
        for (size_t i = 0; i < w * h; ++i)
        {
            size_t p = (i % w) + w * (h - 1 - i / w);
            fprintf(f, "%d %d %d ",
                    (int)(clamp(c[p].r) * 255),
                    (int)(clamp(c[p].g) * 255),
                    (int)(clamp(c[p].b) * 255));
        }
        // Scale
        // float maxv = 1e-10f;
        // for (size_t i = 0; i < w * h; ++i)
        // {
        //     maxv = fmax(maxv, c[i].r);
        //     maxv = fmax(maxv, c[i].g);
        //     maxv = fmax(maxv, c[i].b);
        // }
        // for (size_t i = 0; i < w * h; ++i)
        // {
        //     size_t p = (i % w) + w * (h - 1 - i / w);
        //     fprintf(f, "%d %d %d ",
        //             (int)(clamp(c[p].r / maxv) * 255),
        //             (int)(clamp(c[p].g / maxv) * 255),
        //             (int)(clamp(c[p].b / maxv) * 255));
        // }
        // Pow
        // for (size_t i = 0; i < w * h; ++i)
        // {
        //     size_t p = (i % w) + w * (h - 1 - i / w);
        //     fprintf(f, "%d %d %d ",
        //             (int)(pow(clamp(c[p].r / maxv.r), 1 / 2.2) * 255 + .5),
        //             (int)(pow(clamp(c[p].g / maxv.g), 1 / 2.2) * 255 + .5),
        //             (int)(pow(clamp(c[p].b / maxv.b), 1 / 2.2) * 255 + .5));
        // }
    }

   private:
    MDSpace<2> dim;
    Color *color_buffer;
    Ray *ray_buffer, *ray2_buffer;
};

void A_Ball(Scene *scene, Camera *cam)
{
    ComputeBSDF bsdf = {
        {},
        {1.0f, 0.0f, 0.0f},
        {{1.0f, 0.0f, 0.0f}},
        {{0.0f, 0.0f, 0.0f}},
        {{0.0f, 0.0f, 0.0f}}
    };
    ComputeLight light = {
        {},
        {{1.0f, 1.0f, 1.0f}}
    };
    Sphere s1(1.0f, {0.0f, 0.0f, 3.0f});
    Sphere s2(0.3f, {1.3f, 0.0f, 2.5f});

    Object o = {new Sphere(s1), new ComputeBSDF(bsdf), nullptr};
    Object l = {new Sphere(s2), nullptr, new ComputeLight(light)};
    scene->objs.push_back(o);
    scene->objs.push_back(l);

    // scene->lights.push_back(
    //     new PointLight({-2.0f, 1.0f, -1.0f}, {1.0f, 0.0f, 0.0f}));
    // scene->lights.push_back(
    //     new PointLight({2.0f, 1.0f, -1.0f}, {0.0f, 1.0f, 0.0f}));
    // scene->lights.push_back(
    //     new PointLight({0.0f, -2.0f, -1.0f}, {0.0f, 0.0f, 1.0f}));

    Camera c = {{0.0f, 0.0f, 0.0f}, {0.0f, 0.0f, 1.0f}, 0.5 * M_PI, 0.5 * M_PI};
    *cam = c;
}
void Cornell_Box(Scene *scene, Camera *cam)
{
    ComputeBSDF grey = {
        {}, {1.0f, 0.0f, 0.0f},
        {{0.75f, 0.75f, 0.75f}}, {{0.0f, 0.0f, 0.0f}}, {{0.0f, 0.0f, 0.0f}}
    };
    ComputeBSDF green = {
        {}, {1.0f, 0.0f, 0.0f},
        {{0.1f, 1.0f, 0.1f}}, {{0.0f, 0.0f, 0.0f}}, {{0.0f, 0.0f, 0.0f}}
    };
    ComputeBSDF red = {
        {}, {1.0f, 0.0f, 0.0f},
        {{1.0f, 0.0f, 0.0f}}, {{0.0f, 0.0f, 0.0f}}, {{0.0f, 0.0f, 0.0f}}
    };
    ComputeBSDF blue = {
        {}, {1.0f, 0.0f, 0.0f},
        {{0.0f, 0.0f, 1.0f}}, {{0.0f, 0.0f, 0.0f}}, {{0.0f, 0.0f, 0.0f}}
    };

    Sphere left(999.0f, {-1000.0f, -3.0f, 5.0f});
    Sphere right(999.0f, {1000.0f, 3.0f, 5.0f});
    Sphere up(999.0f, {0.0f, 1000.0f, 5.0f});
    Sphere bottom(999.0f, {0.0f, -1000.0f, 5.0f});
    Sphere back(999.0f, {0.0f, 0.0f, 1001.0f});

    // DiffuseReflection ball_color({1.0f, 1.0f, 1.0f});
    // Sphere ball(0.3f, {0.0f, 0.0f, 0.95f});

    // {{-0.4f, -0.4f, 0.95f}, {1.0f, 0.9f, 1.0f}},
    // {{0.4f, 0.4f, 0.95f}, {0.8f, 1.0f, 0.8f}},
    // {{0.0f, 0.0f, -1.0f}, {1.0f, 1.0f, 1.0f}}
    ComputeLight light = {
        {},
        {{1.0f, 1.0f, 1.0f}}
    };
    Sphere l1(0.2f, {0.0f, 0.0f, 1.0f});

    Object objs[] = {
        {new Sphere(left), new ComputeBSDF(red), nullptr},
        {new Sphere(right), new ComputeBSDF(blue), nullptr},
        {new Sphere(up), new ComputeBSDF(grey), nullptr},
        {new Sphere(bottom), new ComputeBSDF(grey), nullptr},
        {new Sphere(back), new ComputeBSDF(grey), nullptr},
        {new Sphere(l1), nullptr, new ComputeLight(light)},
    };
    scene->objs.clear();
    for (Object &o : objs)
    {
        scene->objs.push_back(o);
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
    render.update(scene, cam, 100, 6);

    return 0;
};
