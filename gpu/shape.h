#pragma once

/* shape */
class Shape
{
   public:
    virtual bool intersect(const Ray &r, float &tHit) const = 0;
    virtual Normal getNormal(const Point &pos) const = 0;
};

class Sphere : public Shape
{
   public:
    float radius;
    Vertex center;

   public:
    Sphere(float r, const Vertex &c) : radius(r), center(c)
    {
    }
    virtual bool intersect(const Ray &r, float &tHit) const
    {
        Vertex op = center - r.pos;
        float eps = 1e-4;
        float b = op.dot(r.dir);
        float det = b * b - op.dot(op) + radius * radius;

        tHit = 0.0f;
        if (det >= 0.0f)
        {
            det = sqrt(det);
            if (b - det > eps)
                tHit = b - det;
            else if (b + det > eps)
                tHit = b + det;
        }
        return tHit != 0.0f;
    }
    virtual Normal getNormal(const Point &pos) const
    {
        return (pos - center).norm();
    }
};
