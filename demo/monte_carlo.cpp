// c++ -o monte_carlo -O3 -Wall monte_carlo.cpp
#include <algorithm>
#include <cmath>
#include <fstream>
#include <iostream>
#include <random>

float bound(float v, float low, float high)
{
    return std::max(low, std::min(high, v));
}
void draw(int w, int h, float* r, float* g, float* b)
{
    std::ofstream ofs("./monte_carlo.ppm", std::ios::out | std::ios::binary);
    ofs << "P6\n" << w << " " << h << "\n255\n";
    for (unsigned i = 0; i < w * h; ++i)
    {
        ofs << (unsigned char)(bound(r[i], 0.0f, 1.0f) * 255.0f)
            << (unsigned char)(bound(g[i], 0.0f, 1.0f) * 255.0f)
            << (unsigned char)(bound(b[i], 0.0f, 1.0f) * 255.0f);
    }
    ofs.close();
}
float dist2(float x1, float y1, float x2, float y2)
{
    return (x2 - x1) * (x2 - x1) + (y2 - y1) * (y2 - y1);
}

void fill_color(int w, int h, float* r, float* g, float* b)
{
    for (int row = 0; row < h; ++row)
    {
        for (int col = 0; col < w; ++col)
        {
            int i = row * w + col;
            r[i] = (float)(h - row) / h;
            g[i] = (float)col / w;
            b[i] = (float)(row + col) / (w + h);
        }
    }
}
void fill_monte_carlo(int w, int h, float* r, float* g, float* b)
{
    std::default_random_engine generator;
    std::uniform_real_distribution<double> distribution(0.0, 1.0);

    int size = w * h;
    for (int i = 0; i < size; ++i)
    {
        int pos = i; //size * distribution(generator);
        pos = std::max(0, std::min(size - 1, pos));
        float x1 = pos % w, y1 = pos / w;
        float x2 = w / 2, y2 = h / 2;
        x1 /= w, y1 /= h;
        x2 /= w, y2 /= h;
        float x3 = x1, y3 = 0.5f + std::cos(4.0 * 3.141592 * x1) / 8.0;

        if (dist2(x1, y1, x2, y2) < 0.25f)
        {
            float d = dist2(x1, y1, x3, y3);
            // printf("%.4f\n", d);
            r[pos] = g[pos] = b[pos] = 0.001f / d;
        }
    }
}
void filter(int w, int h, float* rmask, float* gmask, float* bmask, float* r, float* g, float* b)
{
    for (int i = 0; i < w * h; ++i)
    {
        r[i] *= rmask[i];
        g[i] *= gmask[i];
        b[i] *= bmask[i];
    }
}

int main()
{
    const int D = 512;
    float r[D * D], g[D * D], b[D * D];
    float rmask[D * D], gmask[D * D], bmask[D * D];
    fill_monte_carlo(D, D, rmask, gmask, bmask);
    fill_color(D, D, r, g, b);
    filter(D, D, rmask, gmask, bmask, r, g, b);
    draw(D, D, r, g, b);
    return 0;
}
