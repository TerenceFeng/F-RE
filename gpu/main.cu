#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <curand_kernel.h>
//#include <thrust/host_vector.h>
//#include <thrust/device_vector.h>

// Fast Ray-AABB
// https://tavianator.com/fast-branchless-raybounding-box-intersections/
// Fast Intersection of all types, static/moving
// http://www.realtimerendering.com/intersections.html

#define USE_OPENGL

#include "common.h"
#include "struct.h"
#include "kernels.h"

#include "mem.h"

#ifdef USE_OPENGL
#include "display.h"
#endif

Color Clamp(const Color &c)
{
    float f = fmax(c.r, fmax(c.g, c.b));
    Color c2 = c;
    if (f > 1.0f)
    {
        c2.v.scale(1.0f / f);
        c2.r = fmin(1.0f, fmax(0.0f, c2.r));
        c2.g = fmin(1.0f, fmax(0.0f, c2.g));
        c2.b = fmin(1.0f, fmax(0.0f, c2.b));
    }
    return c2;
}

#define BLOCK_SIZE 8

class Render
{
public:
    Render(size_t w, size_t h)
        : W(w), H(h),
        color(w * h, IN_HOST | IN_DEVICE),
        states(BLOCK_SIZE * BLOCK_SIZE, IN_DEVICE),
        scene(nullptr), auto_clear(true)
    {}
    void init(Scene &_scene, size_t samp)
    {
        scene = &_scene;
        s = sample = _scene.sample_count;

        dim3 blockD(BLOCK_SIZE, BLOCK_SIZE);
        dim3 gridD(1, 1);

        init_rand <<<gridD, blockD>>> (states.getDevice());
        CheckCUDAError(cudaDeviceSynchronize());
    }
    bool update(unsigned char *data)
    {
        if (s == 0) return false;
        --s;

        dim3 blockD(BLOCK_SIZE, BLOCK_SIZE);
        dim3 gridD((W + BLOCK_SIZE - 1) / BLOCK_SIZE,
            (H + BLOCK_SIZE - 1) / BLOCK_SIZE);

        float px[] = {0.0f, 0.0f, 1.0f, 1.0f};
        float py[] = {0.0f, 1.0f, 1.0f, 0.0f};
        for (int corner = 0; corner < 4; ++corner)
        {
            //super_trace_ray <<<gridD, blockD >>>
            //    (color.getDevice(), scene->camera()->getDevice(),
            //     scene->object().getSize(),
            //     states.getDevice(),
            //     px[corner], py[corner], (float)sample);
            super_normal_map <<<gridD, blockD >>>
                (color.getDevice(), scene->camera()->getDevice(),
                 scene->object().getSize(),
                 states.getDevice(),
                 px[corner], py[corner], (float)sample);
            CheckCUDAError(cudaDeviceSynchronize());
        }

#ifdef USE_OPENGL
        if (s % 10 == 0)
        {
            color.copyFromDevice();
            //for (size_t i = 0; i < color.getSize(); ++i)
            //{
            //    data[i * 3] = (char)(pow(clamp(color.getHost()[i].r), 1 / 2.2) * 255 + .5);
            //    data[i * 3 + 1] = (char)(pow(clamp(color.getHost()[i].g), 1 / 2.2) * 255 + .5);
            //    data[i * 3 + 2] = (char)(pow(clamp(color.getHost()[i].b), 1 / 2.2) * 255 + .5);
            //}

            for (size_t i = 0; i < color.getSize(); ++i)
            {
                data[i * 3] = clamp((color.getHost()[i].r)) * 255;
                data[i * 3 + 1] = clamp((color.getHost()[i].g)) * 255;
                data[i * 3 + 2] = clamp((color.getHost()[i].b)) * 255;
            }

            //for (size_t i = 0; i < color.getSize(); ++i)
            //{
            //	Color c = Clamp(color.getHost()[i]);
            //	data[i * 3] = c.r * 255;
            //	data[i * 3 + 1] = c.g * 255;
            //	data[i * 3 + 2] = c.b * 255;
            //}

            //Color maxv = {0.0f, 0.0f, 0.0f};
            //for (size_t i = 0; i < color.getSize(); ++i)
            //{
            //    maxv.r = max(maxv.r, color.getHost()[i].r);
            //    maxv.g = max(maxv.g, color.getHost()[i].g);
            //    maxv.b = max(maxv.b, color.getHost()[i].b);
            //}
            //for (size_t i = 0; i < color.getSize(); ++i)
            //{
            //    data[i * 3] = clamp((color.getHost()[i].r) / (maxv.r)) * 255;
            //    data[i * 3 + 1] = clamp((color.getHost()[i].g) / (maxv.g)) * 255;
            //    data[i * 3 + 2] = clamp((color.getHost()[i].b) / (maxv.b)) * 255;
            //}

            //float maxv = 0.0f, minv = 1e10f;
            //for (size_t i = 0; i < color.getSize(); ++i)
            //{
            //    maxv = max(maxv, color.getHost()[i].r);
            //    if (!color.getHost()[i].v.isZero())
            //    {
            //        minv = min(minv, color.getHost()[i].r);
            //    }
            //}
            //for (size_t i = 0; i < color.getSize(); ++i)
            //{
            //    data[i * 3] = data[i * 3 + 1] = data[i * 3 + 2] = 0;
            //    if (!color.getHost()[i].v.isZero())
            //    data[i * 3] = data[i * 3 + 1] = data[i * 3 + 2] = 255 - clamp((color.getHost()[i].r - minv) / (maxv - minv)) * 255;
            //}
        }
#endif
        CheckCUDAError(cudaGetLastError());
        CheckCUDAError(cudaDeviceSynchronize());
        return true;
    }

    void toggleAutoClear()
    {
        auto_clear = !auto_clear;
    }
    const char * getInfo()
    {
        Color maxv = {0.0f, 0.0f, 0.0f}, minv = {1e5, 1e5, 1e5};
        for (size_t i = 0; i < color.getSize(); ++i)
        {
            maxv.r = max(maxv.r, color.getHost()[i].r);
            maxv.g = max(maxv.g, color.getHost()[i].g);
            maxv.b = max(maxv.b, color.getHost()[i].b);
            if (!color.getHost()[i].v.isZero())
            {
                minv.r = min(minv.r, color.getHost()[i].r);
                minv.g = min(minv.g, color.getHost()[i].g);
                minv.b = min(minv.b, color.getHost()[i].b);
            }
        }
        static char info[256];
        sprintf_s(info, "%d (%.2f, %.2f, %.2f) (%.4f, %.4f) min/max(%.2f/%.2f, %.2f/%.2f, %.2f/%.2f)",
                  sample - s,
                  scene->camera()->getHost()->pos.x, scene->camera()->getHost()->pos.y, scene->camera()->getHost()->pos.z,
                  scene->camera()->getHost()->fov_h, scene->camera()->getHost()->fov_v,
                  minv.x, maxv.x, minv.y, maxv.y, minv.z, maxv.z);
        return info;
    }
    void clearScreen()
    {
        for (size_t i = 0; i < color.getSize(); ++i)
            color.getHost()[i].v.zero();
        color.copyToDevice();
        s = sample;
    }
    void changFOV_h(float f)
    {
        scene->camera()->getHost()->fov_h += f;
        scene->camera()->copyToDevice();
        if (auto_clear) clearScreen();
    }
    void changFOV_v(float f)
    {
        scene->camera()->getHost()->fov_v += f;
        scene->camera()->copyToDevice();
        if (auto_clear) clearScreen();
    }
    void moveX(float v)
    {
        scene->camera()->getHost()->pos.x += v;
        scene->camera()->copyToDevice();
        if (auto_clear) clearScreen();
    }
    void moveY(float v)
    {
        scene->camera()->getHost()->pos.y += v;
        scene->camera()->copyToDevice();
        if (auto_clear) clearScreen();
    }
    void moveZ(float v)
    {
        scene->camera()->getHost()->pos.z += v;
        scene->camera()->copyToDevice();
        if (auto_clear) clearScreen();
    }
    float ratio()
    {
        return (float)W / H;
    }
    float progress()
    {
        return (float)(sample - s) / sample;
    }

    void done()
    {
        CheckCUDAError(cudaDeviceReset());
    }
    void display(const char *filename)
    {
        color.copyFromDevice();
        int h = H, w = W;
        Color *c = color.getHost();
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
    }

private:
    size_t W, H;
    Pool<Color> color;
    Pool<curandState> states;

    Scene *scene;

    // DEBUG
    size_t sample, s;
    bool auto_clear;
};

void Scene::load(const char *file)
{
    std::ifstream f(file, std::ifstream::in);
    std::istringstream iss;
    std::string line;
    int id;
    while (f.good())
    {
        std::getline(f, line);
        if (line == "[camera]")
        {
            LogInfo("Scene: load camera info...");
            std::getline(f, line);
            if (!line.empty())
            {
                iss = std::istringstream(line);
                _camera = new Pool<Camera>(1, IN_HOST | IN_DEVICE);
                Camera &cam = _camera->getHost()[0];
                iss >> cam.pos
                    >> cam.dir
                    >> cam.fov_h
                    >> cam.fov_v;
                cam.dir.norm();
                _camera->copyToDevice();
            }
        }
        else if (line == "[sample]")
        {
            LogInfo("Scene: load sample info...");
            std::getline(f, line);
            if (!line.empty())
            {
                iss = std::istringstream(line);
                iss >> sample_count;
            }
        }
        else if (line == "[bsdf]")
        {
            LogInfo("Scene: load bsdf info...");
            while (f.good())
            {
                std::getline(f, line);
                if (!line.empty())
                {
                    iss = std::istringstream(line);
                    iss >> id;
                    Color R = {0.0f, 0.0f, 0.0f};
                    switch (id)
                    {
                        case LAMBERTIAN:
                            iss >> R.r >> R.g >> R.b;
                            bsdf_factory.createLambertian(R);
                            break;
                        case SPEC_REFL:
                            iss >> R.r >> R.g >> R.b;
                            bsdf_factory.createSpecRefl(R);
                            break;
                        case SPEC_TRANS:
                            iss >> R.r >> R.g >> R.b;
                            bsdf_factory.createSpecTrans(R);
                            break;
                        default:
                            break;
                    }
                }
                else break;
            }
        }
        else if (line == "[bsdf-picker]")
        {
            LogInfo("Scene: load bsdf-picker info...");
            while (f.good())
            {
                std::getline(f, line);
                if (!line.empty())
                {
                    iss = std::istringstream(line);
                    bsdf_handle_t models[3];
                    float cdf[3];
                    iss >> models[0] >> models[1] >> models[2]
                        >> cdf[0] >> cdf[1] >> cdf[2];
                    bsdf_factory.createPicker(models, cdf);
                }
                else break;
            }
        }
        else if (line == "[shape]")
        {
            LogInfo("Scene: load shape info...");
            while (f.good())
            {
                std::getline(f, line);
                if (!line.empty())
                {
                    iss = std::istringstream(line);
                    ShapeEntity s;
                    iss >> s;
                    shape_factory.createShape(s);
                }
                else break;
            }
        }
        else if (line == "[light]")
        {
            LogInfo("Scene: load light info...");
            while (f.good())
            {
                std::getline(f, line);
                if (!line.empty())
                {
                    iss = std::istringstream(line);
                    iss >> id;
                    Color R = {0.0f, 0.0f, 0.0f};
                    iss >> R.r >> R.g >> R.b;
                    light_factory.createPointLight(R);
                }
                else break;
            }
        }
        else if (line == "[texture]")
        {
            LogInfo("Scene: load texture info...");
            while (f.good())
            {
                std::getline(f, line);
                if (!line.empty())
                {
                    texture_factory.createTexture(line.data());
                }
                else break;
            }
        }
        else if (line == "[texture-mapping]")
        {
            LogInfo("Scene: load texture-mapping info...");
            while (f.good())
            {
                std::getline(f, line);
                if (!line.empty())
                {
                    iss = std::istringstream(line);
                    shape_handle_t shape_id;
                    tex_handle_t tex_id;
                    iss >> shape_id >> tex_id;
                    ShapeEntity * shape = shape_factory.getHost(shape_id);
                    if (shape)
                    {
                        iss >> shape->triangle.t1
                            >> shape->triangle.t2
                            >> shape->triangle.t3;
                        shape->triangle.tex = texture_factory.getDevice(tex_id);
                    }
                    else
                    {
                        ShowErrorAndExit("Scene: bad texture mapping");
                    }
                }
                else break;
            }
        }
        else if (line == "[object]")
        {
            LogInfo("Scene: load object info...");
            while (f.good())
            {
                std::getline(f, line);
                if (!line.empty())
                {
                    iss = std::istringstream(line);
                    shape_handle_t shape_id;
                    bsdf_handle_t picker_id;
                    light_handle_t light_id;
                    iss >> shape_id >> picker_id >> light_id;
                    Object o = {
                        shape_id ? shape_factory.getDevice(shape_id - 1) : nullptr,
                        picker_id ? bsdf_factory.getDevice_picker(picker_id - 1) : nullptr,
                        light_id ? light_factory.getDevice(light_id - 1) : nullptr
                    };
                    object_factory.createObject(o);
                }
                else break;
            }
        }
    }

    if (object_factory.getSize() == 0)
        ShowErrorAndExit("Scene: no object defined.");
    bsdf().syncToDevice();
    shape().syncToDevice();
    light().syncToDevice();
    texture().syncToDevice();
    object().syncToDevice();
}

#ifndef USE_OPENGL

int main(void)
{
    Scene scene;
    scene.load("scene.txt");
    if (scene.camera() == nullptr)
    {
        scene.camera() = new Pool<Camera>(1, IN_HOST | IN_DEVICE);
        scene.camera()->getHost()[0] = {
            {50.0f, 52.0f, 169.9f},
            Vector(0.0f, -0.042612f, -1.0f).norm(),
            1.9043f, 2.0213f};
        scene.camera()->copyToDevice();
    }

    Render render(640, 480);
    render.init(scene, 100);
    while (render.update(nullptr))
    {
        printf("\r%.2f%", render.progress() * 100.0f);
    }
    printf("\n");
    render.display("image.ppm");
    render.done();
    return 0;
}

#else

static Render render(W_WIDTH, W_HEIGHT);
static Scene *g_scene;

void GLInitCallback()
{
    g_scene = new Scene();
    (*g_scene).load("scene.txt");
    render.init((*g_scene), 500);
    if ((*g_scene).camera() == nullptr)
    {
        (*g_scene).camera() = new Pool<Camera>(1, IN_HOST | IN_DEVICE);
        (*g_scene).camera()->getHost()[0] = {
            {50.0f, 52.0f, 169.9f},
            Vector(0.0f, -0.042612f, -1.0f).norm(), 
            1.9043f, 2.0213f};
        (*g_scene).camera()->copyToDevice();
    }
}
const char * GLWindowTitle()
{
    return render.getInfo();
}
void GLOnKeyPress(int key)
{
    switch (key)
    {
        case 0x57: // W
            render.moveZ(-1.f); break;
        case 0x53: // S
            render.moveZ(1.f); break;
        case 0x41: // A
            render.moveX(-1.f); break;
        case 0x44: // D
            render.moveX(1.f); break;
        case 0x45: // E
            render.moveY(1.f); break;
            //render.changFOV_v(-0.001f); break;
        case 0x51: // Q
            render.moveY(-1.f); break;
            //render.changFOV_v(0.001f); break;
        case 0x5A: // Z
            render.changFOV_h(-0.001f); break;
        case 0x43: // C
            render.changFOV_h(0.001f); break;
        case 0x46: // F
            render.clearScreen(); break;
        case 0x58: // X
            render.toggleAutoClear(); break;
    }
}
bool GLUpdateCallback()
{
    return render.update(texData);
}
void GLDoneCallback()
{
    render.done();
    DeviceMemoryManager::Summary();
}

#endif
