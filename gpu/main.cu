#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <curand_kernel.h>
//#include <thrust/host_vector.h>
//#include <thrust/device_vector.h>

//#define USE_OPENGL

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

class Render
{
public:
    Render(size_t w, size_t h)
        : W(w), H(h),
        color(w * h, IN_HOST | IN_DEVICE),
        c2(w * h, IN_DEVICE),
        ray0(w * h, IN_DEVICE),
        ray1(w * h, IN_DEVICE),
        state(w * h, IN_DEVICE),
        scene(nullptr), cam(0), auto_clear(true)
    {}
    void init(Scene &_scene, Pool<Camera> &cam, size_t samp)
    {
        this->scene = &_scene;
        this->cam.swap(cam);
        sample = samp;
        s = samp;

        const int edge = 8;
        dim3 blockD(edge, edge);
        dim3 gridD((W + edge - 1) / edge, (H + edge - 1) / edge);

        init_rand <<<gridD, blockD>>> (state.getDevice());
        CheckCUDAError(cudaGetLastError());
    }
    bool update(unsigned char *data)
    {
        if (s == 0) return false;
        --s;

        const int edge = 8;
        dim3 blockD(edge, edge);
        dim3 gridD((W + edge - 1) / edge, (H + edge - 1) / edge);

        init_ray <<<gridD, blockD>>> (ray0.getDevice(), cam.getDevice(), state.getDevice());
        //ray2color<<<gridD,blockD>>>(color.getDevice(), ray0.getDevice());
        //ray_depth<<<gridD,blockD>>>(color.getDevice(), ray0.getDevice(),
        //							  obj.getDevice(), obj.getSize());
        //ray_distance<<<gridD,blockD>>>(ray0.getDevice(), ray1.getDevice(), color.getDevice(),
        //                               obj.getDevice(), obj.getSize()
        //                               bsdf, state.getDevice());
        CheckCUDAError(cudaGetLastError());

        trace_ray <<<gridD, blockD>>> (ray0.getDevice(), ray1.getDevice(), c2.getDevice(),
                                       scene->object().getDevice(), scene->object().getSize(),
                                       scene->bsdf().getDevice_bsdf(),
                                       state.getDevice());
        CheckCUDAError(cudaGetLastError());
        scale_add <<<gridD, blockD>>> (color.getDevice(), c2.getDevice(), 50.0f / (float)sample);
        CheckCUDAError(cudaGetLastError());

        color.copyFromDevice();
#ifdef USE_OPENGL
        //for (size_t i = 0; i < color.getSize(); ++i)
        //{
        //	data[i * 3] = clamp((color.getHost()[i].r)) * 255;
        //	data[i * 3 + 1] = clamp((color.getHost()[i].g)) * 255;
        //	data[i * 3 + 2] = clamp((color.getHost()[i].b)) * 255;
        //}

        for (size_t i = 0; i < color.getSize(); ++i)
        {
            data[i * 3] = pow(clamp(color.getHost()[i].r), 1 / 2.2) * 255 + .5;
            data[i * 3 + 1] = pow(clamp(color.getHost()[i].g), 1 / 2.2) * 255 + .5;
            data[i * 3 + 2] = pow(clamp(color.getHost()[i].b), 1 / 2.2) * 255 + .5;
        }

        //for (size_t i = 0; i < color.getSize(); ++i)
        //{
        //	Color c = Clamp(color.getHost()[i]);
        //	data[i * 3] = c.r * 255;
        //	data[i * 3 + 1] = c.g * 255;
        //	data[i * 3 + 2] = c.b * 255;
        //}

        //Color maxv = { 0.0f, 0.0f, 0.0f };
        //for (size_t i = 0; i < color.getSize(); ++i)
        //{
        //	maxv.r = max(maxv.r, color.getHost()[i].r);
        //	maxv.g = max(maxv.g, color.getHost()[i].g);
        //	maxv.b = max(maxv.b, color.getHost()[i].b);
        //}
        //for (size_t i = 0; i < color.getSize(); ++i)
        //{
        //	data[i * 3] = clamp((/*maxv.r - */color.getHost()[i].r) / maxv.r) * 255;
        //	data[i * 3 + 1] = clamp((/*maxv.g - */color.getHost()[i].g) / maxv.g) * 255;
        //	data[i * 3 + 2] = clamp((/*maxv.b - */color.getHost()[i].b) / maxv.b) * 255;
        //}
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
        static char info[256];
        sprintf_s(info, "%d (%.2f, %.2f, %.2f) (%.4f, %.4f)",
                  sample - s,
                  cam.getHost()->pos.x, cam.getHost()->pos.y, cam.getHost()->pos.z,
                  cam.getHost()->fov_h, cam.getHost()->fov_v);
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
        cam.getHost()->fov_h += f;
        cam.copyToDevice();
        if (auto_clear) clearScreen();
    }
    void changFOV_v(float f)
    {
        cam.getHost()->fov_v += f;
        cam.copyToDevice();
        if (auto_clear) clearScreen();
    }
    void moveX(float v)
    {
        cam.getHost()->pos.x += v;
        cam.copyToDevice();
        if (auto_clear) clearScreen();
    }
    void moveY(float v)
    {
        cam.getHost()->pos.y += v;
        cam.copyToDevice();
        if (auto_clear) clearScreen();
    }
    void moveZ(float v)
    {
        cam.getHost()->pos.z += v;
        cam.copyToDevice();
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
    Pool<Color> color, c2;
    Pool<Ray> ray0, ray1;
    Pool<curandState> state;

    Scene *scene;
    Pool<Camera> cam;

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
        if (line == "[bsdf]")
        {
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
        if (line == "[bsdf-picker]")
        {
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
                        PrintError("Scene: bad texture mapping");
                        exit(1);
                    }
                }
                else break;
            }
        }
        else if (line == "[object]")
        {
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
}

#ifndef USE_OPENGL

int main(void)
{
    Scene scene;
    scene.load("scene.txt");
    Pool<Camera> cam(1, IN_HOST | IN_DEVICE);
    cam.getHost()[0] = {{50.0f, 52.0f, 169.9f},Vector(0.0f, -0.042612f, -1.0f).norm(), 1.9043f, 2.0213f};
    cam.copyToDevice();

    Render render(640, 480);
    render.init(scene, cam, 100);
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
static Scene scene;

void GLInitCallback()
{
    scene.load("scene.txt");
    Pool<Camera> cam(1, IN_HOST | IN_DEVICE);
    cam.getHost()[0] = {{50.0f, 52.0f, 169.9f},Vector(0.0f, -0.042612f, -1.0f).norm(), 1.9043f, 2.0213f};
    cam.copyToDevice();
    render.init(scene, cam, 500);
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
            render.moveZ(-0.1f); break;
        case 0x53: // S
            render.moveZ(0.1f); break;
        case 0x41: // A
            render.moveX(-0.1f); break;
        case 0x44: // D
            render.moveX(0.1f); break;
        case 0x45: // E
            render.changFOV_v(-0.001f); break;
        case 0x51: // Q
            render.changFOV_v(0.001f); break;
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
}

#endif