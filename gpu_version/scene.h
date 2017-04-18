#pragma once

#include <cmath>

struct Object
{
    Shape *shape;
    // BxDF *bsdf;
};

#include <vector>
/* scene */
class Scene
{
   public:
    std::vector<Object> objs;
    std::vector<PointLight *> lights;
};

/* view */
struct Camera
{
    Vertex pos, dir;
    float fov_h, fov_v; // radius
};
