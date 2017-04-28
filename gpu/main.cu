#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <curand_kernel.h>
//#include <thrust/host_vector.h>
//#include <thrust/device_vector.h>

#include "common.h"
#include "struct.h"
#include "kernels.h"

#include "mem.h"

class Scene
{
public:
    Pool<Object> objs;
};

#include "display.h"

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
        obj(0), cam(0), auto_clear(true)
    {}
    void init(Pool<Object> &obj, Pool<Camera> &cam, _index_node *inode_list, size_t samp)
    {
        this->obj.swap(obj);
        this->cam.swap(cam);
        this->inode_list = inode_list;
        sample = samp;
        s = samp;

        const int edge = 4;
        dim3 blockD(edge, edge);
        dim3 gridD((W + edge - 1) / edge, (H + edge - 1) / edge);

        init_rand <<<gridD, blockD>>> (state.getDevice());
    }
    bool update(unsigned char *data)
    {
        if (s == 0) return false;
        --s;

        const int edge = 4;
        dim3 blockD(edge, edge);
        dim3 gridD((W + edge - 1) / edge, (H + edge - 1) / edge);

        init_ray <<<gridD, blockD>>> (ray0.getDevice(), cam.getDevice(), state.getDevice());
        //ray2color<<<gridD,blockD>>>(color.getDevice(), ray0.getDevice());
        //ray_depth<<<gridD,blockD>>>(color.getDevice(), ray0.getDevice(),
        //							  obj.getDevice(), obj.getSize());
        //ray_distance<<<gridD,blockD>>>(ray0.getDevice(), ray1.getDevice(), color.getDevice(),
        //                               obj.getDevice(), obj.getSize()
        //                               bsdf, state.getDevice());

        trace_ray <<<gridD, blockD>>> (ray0.getDevice(), ray1.getDevice(), c2.getDevice(),
                                       obj.getDevice(), obj.getSize(),
                                       inode_list, state.getDevice());
        scale_add <<<gridD, blockD>>> (color.getDevice(), c2.getDevice(), 50.0f / (float)sample);

        color.copyFromDevice();
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

    Pool<Object> obj;
    Pool<Camera> cam;

    _index_node *inode_list;

    // DEBUG
    size_t sample, s;
    bool auto_clear;
};

//void TwoBall(Render &render, size_t sample)
//{
//    //Pool<ComputeBSDF> bsdf(1, IN_HOST | IN_DEVICE);
//    //bsdf.getHost()[0] = {
//    //    {},
//    //    {1.0f, 0.0f, 0.0f},
//    //    {{1.0f, 1.0f, 1.0f}},
//    //    {{0.0f, 0.0f, 0.0f}},
//    //    {{0.0f, 0.0f, 0.0f}}
//    //};
//    //bsdf.copyToDevice();
//    BSDFFactory factory(1);
//    factory.createLambertian({1.0f, 1.0f, 1.0f});
//    factory.syncToDevice();
//    Pool<BSDFPicker> picker(1, IN_HOST | IN_DEVICE);
//    picker.getHost()[0] = {{0, 0, 0}, {1.0f, 0.0f, 0.0f}};
//    picker.copyToDevice();
//
//    Pool<Sphere> shape(2, IN_HOST | IN_DEVICE);
//    Pool<ComputeLight> light(1, IN_HOST | IN_DEVICE);
//
//    shape.getHost()[0] = {0,{ 0.0f, 0.0f, 5000.0f }, 4000.0f};
//    shape.getHost()[1] = {0,{ 0.0f, 0.0f, 500.0f }, 100.0f};
//    shape.copyToDevice();
//    light.getHost()[0] = {{},{ { 1.0f, 1.0f, 1.0f } }};
//    light.copyToDevice();
//
//    Pool<Object> object(2, IN_HOST | IN_DEVICE);
//    object.getHost()[0] = {shape.getDevice(), picker.getDevice(), nullptr}; // bsdf.getDevice(), light.getDevice()};
//    object.getHost()[1] = {shape.getDevice() + 1, nullptr, light.getDevice()}; // bsdf.getDevice(), light.getDevice()};
//    object.copyToDevice();
//
//    Pool<Camera> cam(1, IN_HOST | IN_DEVICE);
//    cam.getHost()[0] = {{ 0.0f, 0.0f, 0.0f },{ 0.0f, 0.0f, 1.0f }, 0.66f * M_PI, 0.66f * M_PI};
//    cam.copyToDevice();
//
//    render.init(object, cam, factory.getIndexNodeList(), sample);
//}

//void CornellBox(Render &render, size_t sample)
//{
//    Pool<Sphere> shape(7, IN_HOST | IN_DEVICE);
//    Pool<ComputeBSDF> bsdf(5, IN_HOST | IN_DEVICE);
//    Pool<ComputeLight> light(1, IN_HOST | IN_DEVICE);
//
//    shape.getHost()[0] = {0, { -10000.0f, -3.0f, 5.0f }, 9999.0f};
//    shape.getHost()[1] = {0, { 10000.0f, 3.0f, 5.0f }, 9999.0f};
//    shape.getHost()[2] = {0, { 0.0f, 10000.0f, 5.0f }, 9999.0f};
//    shape.getHost()[3] = {0, { 0.0f, -10000.0f, 5.0f }, 9999.0f};
//    shape.getHost()[4] = {0, { 0.0f, 0.0f, 10001.0f }, 9999.0f};
//    shape.getHost()[5] = {0, {0.0f, 1.5f, 1.8f}, 0.6f};
//    shape.getHost()[6] = {0,{ 0.0f, 0.0f, -5001.0f }, 5001.0f};
//
//    bsdf.getHost()[0] = { // grey
//        {}, {1.0f, 0.0f, 0.0f},
//        {{0.75f, 0.75f, 0.75f}}, {{0.0f, 0.0f, 0.0f}}, {{0.0f, 0.0f, 0.0f}}
//    };
//    bsdf.getHost()[1] = { // green
//        {}, {1.0f, 0.0f, 0.0f},
//        {{0.1f, 1.0f, 0.1f}}, {{0.0f, 0.0f, 0.0f}}, {{0.0f, 0.0f, 0.0f}}
//    };
//    bsdf.getHost()[2] = { // red
//        {}, {1.0f, 0.0f, 0.0f},
//        {{1.0f, 0.0f, 0.0f}}, {{0.0f, 0.0f, 0.0f}}, {{0.0f, 0.0f, 0.0f}}
//    };
//    bsdf.getHost()[3] = { // blue
//        {}, {1.0f, 0.0f, 0.0f},
//        {{0.0f, 0.0f, 1.0f}}, {{0.0f, 0.0f, 0.0f}}, {{0.0f, 0.0f, 0.0f}}
//    };
//    bsdf.getHost()[4] = { // white
//        {},{ 1.0f, 0.0f, 0.0f },
//        { { 1.0f, 1.0f, 1.0f } },{ { 0.0f, 0.0f, 0.0f } },{ { 0.0f, 0.0f, 0.0f } }
//    };
//    light.getHost()[0] = {{},{ { 1.0f, 1.0f, 1.0f } }};
//
//    shape.copyToDevice();
//    bsdf.copyToDevice();
//    light.copyToDevice();
//
//    Pool<Object> object(6, IN_HOST | IN_DEVICE);
//    object.getHost()[0] = {shape.getDevice(), bsdf.getDevice() + 2, nullptr}; // bsdf.getDevice(), light.getDevice()};
//    object.getHost()[1] = {shape.getDevice() + 1, bsdf.getDevice() + 3, nullptr}; // bsdf.getDevice(), light.getDevice()};
//    object.getHost()[2] = {shape.getDevice() + 2, bsdf.getDevice(), nullptr}; // bsdf.getDevice(), light.getDevice()};
//    object.getHost()[3] = {shape.getDevice() + 3, bsdf.getDevice(), nullptr}; // bsdf.getDevice(), light.getDevice()};
//    object.getHost()[4] = {shape.getDevice() + 4, bsdf.getDevice(), nullptr}; // bsdf.getDevice(), light.getDevice()};
//    object.getHost()[5] = {shape.getDevice() + 5, nullptr, light.getDevice()}; // bsdf.getDevice(), light.getDevice()};
//    //object.getHost()[6] = { shape.getDevice() + 6, bsdf.getDevice() + 4, nullptr }; // bsdf.getDevice(), light.getDevice()};
//    object.copyToDevice();
//
//    Pool<Camera> cam(1, IN_HOST | IN_DEVICE);
//    cam.getHost()[0] = {{ 0.0f, 0.0f, 0.0f },{ 0.0f, 0.0f, 1.0f }, 0.66f * M_PI, 0.66f * M_PI};
//    cam.copyToDevice();
//
//    render.init(object, cam, sample);
//}

void SmallPT(Render &render, size_t sample)
{
    BSDFFactory factor(5);
    factor.createLambertian({0.75f, 0.25f, 0.25f});
    factor.createLambertian({0.25f, 0.25f, 0.75f});
    factor.createLambertian({0.75f, 0.75f, 0.75f});
    factor.createLambertian({0.0f, 0.0f, 0.0f});
    factor.createSpecRefl({0.999f, 0.999f, 0.999f});
    factor.syncToDevice();
    Pool<BSDFPicker> picker(5, IN_DEVICE | IN_HOST);
    picker.getHost()[0] = {{0,0,0},{1.0f, 0.0f,0.0f}};
    picker.getHost()[1] = {{1,0,0},{1.0f, 0.0f,0.0f}};
    picker.getHost()[2] = {{2,0,0},{1.0f, 0.0f,0.0f}};
    picker.getHost()[3] = {{3,0,0},{1.0f, 0.0f,0.0f}};
    picker.getHost()[4] = {{0,4,0},{0.0f, 1.0f,0.0f}};
    picker.copyToDevice();

    Pool<Sphere> shape(9, IN_HOST | IN_DEVICE);
    Pool<ComputeLight> light(1, IN_HOST | IN_DEVICE);

    shape.getHost()[0] = {0,{ 1e3f + 1.0f, 40.8f, 81.6f }, 1e3f};
    shape.getHost()[1] = {0,{ -1e3f + 99.0f, 40.8f, 81.6f }, 1e3f};
    shape.getHost()[2] = {0,{ 50.0f, 40.8f, 1e3f }, 1e3f};
    shape.getHost()[3] = {0,{ 50.0f, 40.8f, -1e3f + 170.0f }, 1e3f};
    shape.getHost()[4] = {0,{ 50.0f, 1e3f, 81.6f }, 1e3f};
    shape.getHost()[5] = {0,{ 50.0f, -1e3f + 81.6f, 81.6f }, 1e3f};
    shape.getHost()[6] = {0,{ 27.0f, 16.5f, 47.0f }, 16.5f};
    shape.getHost()[7] = {0,{ 73.0f, 16.5f, 78.0f }, 16.5f};
    shape.getHost()[8] = {0,{ 50.0f, 681.6f - .27f,81.6f }, 600.0f};

    light.getHost()[0] = {{},{ { 12.0f, 12.0f, 12.0f } }};

    shape.copyToDevice();
    light.copyToDevice();

    Pool<Object> object(9, IN_HOST | IN_DEVICE);
    object.getHost()[0] = {shape.getDevice()    , picker.getDevice() + 0, nullptr};
    object.getHost()[1] = {shape.getDevice() + 1, picker.getDevice() + 1, nullptr};
    object.getHost()[2] = {shape.getDevice() + 2, picker.getDevice() + 2, nullptr};
    object.getHost()[3] = {shape.getDevice() + 3, picker.getDevice() + 3, nullptr};
    object.getHost()[4] = {shape.getDevice() + 4, picker.getDevice() + 2, nullptr};
    object.getHost()[5] = {shape.getDevice() + 5, picker.getDevice() + 2, nullptr};

    object.getHost()[6] = {shape.getDevice() + 6, picker.getDevice() + 4, nullptr};
    object.getHost()[7] = {shape.getDevice() + 7, picker.getDevice() + 4, nullptr};

    object.getHost()[8] = {shape.getDevice() + 8, nullptr, light.getDevice()};

    object.copyToDevice();

    Pool<Camera> cam(1, IN_HOST | IN_DEVICE);
    cam.getHost()[0] = {{50.0f, 52.0f, 169.9f},Vector(0.0f, -0.042612f, -1.0f).norm(), 1.9043f, 2.0213f};
    cam.copyToDevice();

    render.init(object, cam, factor.getIndexNodeList(), sample);
}

static Render render(W_WIDTH, W_HEIGHT);

void GLInitCallback()
{
    //TwoBall(render, 1000);
    //CornellBox(render, 1000);
    SmallPT(render, 1000);
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
