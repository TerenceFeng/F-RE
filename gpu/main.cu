// ------- Common --------
#include "common.h"
#include "math.h"
#include "struct.h"
#include "kernels.h"

#include "mem.h"

typedef Vec3<float> Vertex;
typedef Vec3<float> Vector;
typedef Vec3<float> Point;
typedef Vec3<float> Normal;

// ------- Renderer -------
/* #define USE_OPENGL */

#ifdef USE_OPENGL
#include "display.h"
#endif

#include "grid.h"

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

        /* setup space division */
        /* grid = Grid(obj); */

        const int edge = 8;
        dim3 blockD(edge, edge);
        dim3 gridD((W + edge - 1) / edge, (H + edge - 1) / edge);

        init_rand <<<gridD, blockD>>> (state.getDevice());
    }

    void init_grid()
    {
		Grid temp = Grid(obj);
		grid.swap(temp);
    }

    bool update(unsigned char *data)
    {
        if (s == 0) return false;
        --s;

        const int edge = 8;
        dim3 blockD(edge, edge);
        dim3 gridD((W + edge - 1) / edge, (H + edge - 1) / edge);

        init_ray <<<gridD, blockD>>> (ray0.getDevice(), cam.getDevice(), state.getDevice());
        /* ray2color<<<gridD,blockD>>>(color.getDevice(), ray0.getDevice()); */
        // ray_depth<<<gridD,blockD>>>(color.getDevice(), ray0.getDevice(),
        //                               obj.getDevice(), obj.getSize());
        //ray_distance<<<gridD,blockD>>>(ray0.getDevice(), ray1.getDevice(), color.getDevice(),
        //                               obj.getDevice(), obj.getSize()
        //                               bsdf, state.getDevice());

        trace_ray <<<gridD, blockD>>> (ray0.getDevice(), ray1.getDevice(), c2.getDevice(),
                                       obj.getDevice(), obj.getSize(),
                                       inode_list, state.getDevice());

        /*
         * trace_ray_in_grid <<<gridD, blockD>>> (ray0.getDevice(), ray1.getDevice(), c2.getDevice(),
         *                                        grid.cells.getDevice(), grid.cells_size.getDevice(),
         *                                        inode_list, state.getDevice(),
         *                                        grid.x0, grid.y0, grid.z0,
         *                                        grid.x1, grid.y1, grid.z1,
         *                                        grid.nx, grid.ny, grid.nz);
         */

        scale_add <<<gridD, blockD>>> (color.getDevice(), c2.getDevice(), 1.0f / (float)sample);

        color.copyFromDevice();
#ifdef USE_OPENGL
        //for (size_t i = 0; i < color.getSize(); ++i)
        //{
        //	data[i * 3] = clamp((color.getHost()[i].r)) * 255;
        //	data[i * 3 + 1] = clamp((color.getHost()[i].g)) * 255;
        //	data[i * 3 + 2] = clamp((color.getHost()[i].b)) * 255;
        //}

        //for (size_t i = 0; i < color.getSize(); ++i)
        //{
            //data[i * 3] = pow(clamp(color.getHost()[i].r), 1 / 2.2) * 255 + .5;
            //data[i * 3 + 1] = pow(clamp(color.getHost()[i].g), 1 / 2.2) * 255 + .5;
            //data[i * 3 + 2] = pow(clamp(color.getHost()[i].b), 1 / 2.2) * 255 + .5;
        //}

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
        sprintf(info, "%d (%.2f, %.2f, %.2f) (%.4f, %.4f)",
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
        for (size_t i = 0; i < color.getSize(); ++i)
        {
            color.getHost()[i] = clamp(color.getHost()[i]);
        }

        int h = H, w = W;
        Color *c = color.getHost();
        FILE *f;
        f = fopen(filename, "w");
        fprintf(f, "P3\n%d %d\n%d\n", w, h, 255);
        for (size_t i = 0; i < w * h; ++i)
        {
            size_t p = (i % w) + w * (h - 1 - i / w);
            /* Color clamped_c = clamp(c[p]); */
            fprintf(f, "%d %d %d ",
                    /* (int)(clamped_c.r * 255), */
                    /* (int)(clamped_c.g * 255), */
                    /* (int)(clamped_c.b * 255)); */
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

    Grid grid;

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
    BSDFFactory factor(6);
    factor.createLambertian({0.75f, 0.25f, 0.25f});
    factor.createLambertian({0.25f, 0.25f, 0.75f});
    factor.createLambertian({0.75f, 0.75f, 0.75f});
    factor.createLambertian({0.0f, 0.0f, 0.0f});
    factor.createSpecRefl({0.999f, 0.999f, 0.999f});
    factor.createSpecTrans({0.999f, 0.999f, 0.999f});
    factor.syncToDevice();
    Pool<BSDFPicker> picker(6, IN_DEVICE | IN_HOST);
    picker.getHost()[0] = {{0,0,0},{1.0f, 0.0f,0.0f}};
    picker.getHost()[1] = {{1,0,0},{1.0f, 0.0f,0.0f}};
    picker.getHost()[2] = {{2,0,0},{1.0f, 0.0f,0.0f}};
    picker.getHost()[3] = {{3,0,0},{1.0f, 0.0f,0.0f}};
    picker.getHost()[4] = {{0,4,0},{0.0f, 1.0f,0.0f}};
    picker.getHost()[5] = {{5,0,0},{1.0f, 0.0f,0.0f}};
    picker.copyToDevice();

    Pool<Sphere> shape(9, IN_HOST | IN_DEVICE);
    Pool<struct Rectangle> rect(1, IN_HOST | IN_DEVICE);
    Pool<ComputeLight> light(1, IN_HOST | IN_DEVICE);

    shape.getHost()[0] = {1,{1e3f + 1.0f, 40.8f, 81.6f}, 1e3f};
    shape.getHost()[1] = {1,{-1e3f + 99.0f, 40.8f, 81.6f}, 1e3f};
    shape.getHost()[2] = {1,{50.0f, 40.8f, 1e3f}, 1e3f};
    shape.getHost()[3] = {1,{50.0f, 40.8f, -1e3f + 170.0f}, 1e3f};
    shape.getHost()[4] = {1,{50.0f, 1e3f, 81.6f}, 1e3f};
    shape.getHost()[5] = {1,{50.0f, -1e3f + 81.6f, 81.6f}, 1e3f};
    shape.getHost()[6] = {0,{27.0f, 16.5f, 47.0f}, 16.5f};
    shape.getHost()[7] = {0,{73.0f, 16.5f, 78.0f}, 16.5f};
    shape.getHost()[8] = {0,{50.0f, 681.6f - .27f,81.6f}, 600.0f};
    rect.getHost()[0] = {2, {10.0f, 16.5f, 78.0f}, {20.0f, 0.0f, 0.0f}, {0.0f, 20.0f, 0.0f}};
    light.getHost()[0] = {{},{ { 12.0f, 12.0f, 12.0f } }};

    shape.copyToDevice();
    rect.copyToDevice();
    light.copyToDevice();

    Pool<Object> object(10, IN_HOST | IN_DEVICE);
    object.getHost()[0] = {shape.getDevice()    , picker.getDevice() + 0, nullptr};
    object.getHost()[1] = {shape.getDevice() + 1, picker.getDevice() + 0, nullptr};
    object.getHost()[2] = {shape.getDevice() + 2, picker.getDevice() + 2, nullptr};
    object.getHost()[3] = {shape.getDevice() + 3, picker.getDevice() + 3, nullptr};
    object.getHost()[4] = {shape.getDevice() + 4, picker.getDevice() + 2, nullptr};
    object.getHost()[5] = {shape.getDevice() + 5, picker.getDevice() + 2, nullptr};

    object.getHost()[6] = {shape.getDevice() + 6, picker.getDevice() + 4, nullptr};
    object.getHost()[7] = {shape.getDevice() + 7, picker.getDevice() + 4, nullptr};
    object.getHost()[9] = {rect.getDevice(), picker.getDevice() + 1, nullptr};

    object.getHost()[8] = {shape.getDevice() + 8, nullptr, light.getDevice()};

    object.copyToDevice();


    object.getHost()[0] = {shape.getHost()    , picker.getHost() + 0, nullptr};
    object.getHost()[1] = {shape.getHost() + 1, picker.getHost() + 1, nullptr};
    object.getHost()[2] = {shape.getHost() + 2, picker.getHost() + 2, nullptr};
    object.getHost()[3] = {shape.getHost() + 3, picker.getHost() + 3, nullptr};
    object.getHost()[4] = {shape.getHost() + 4, picker.getHost() + 2, nullptr};
    object.getHost()[5] = {shape.getHost() + 5, picker.getHost() + 2, nullptr};

    object.getHost()[6] = {shape.getHost() + 6, picker.getHost() + 4, nullptr};
    object.getHost()[7] = {shape.getHost() + 7, picker.getHost() + 4, nullptr};
    object.getHost()[9] = {rect.getHost(), picker.getHost() + 1, nullptr};

    object.getHost()[8] = {shape.getHost() + 8, nullptr, light.getHost()};


    Pool<Camera> cam(1, IN_HOST | IN_DEVICE);
    cam.getHost()[0] = {{50.0f, 52.0f, 169.9f},Vector(0.0f, -0.042612f, -1.0f).norm(), 1.9043f, 2.0213f};
    cam.copyToDevice();

    render.init(object, cam, factor.getIndexNodeList(), sample);
}

#ifndef USE_OPENGL

int main(void)
{
    Render render(640, 480);
    SmallPT(render, 100);
    render.init_grid();

	printf("MAKR: strat rendering\n");
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

void GLInitCallback()
{
    //TwoBall(render, 1000);
    //CornellBox(render, 1000);
    SmallPT(render, 500);
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
