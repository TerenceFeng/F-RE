#pragma once

#include <cmath>

struct Object
{
    Shape *shape;
    ComputeBSDF *bsdf;
    ComputeLight *light;
};

#include <vector>
/* scene */
class Scene
{
   public:
    std::vector<Object> objs;
};

/* view */
struct Camera
{
    Vertex pos, dir;
    float fov_h, fov_v; // radius
};
