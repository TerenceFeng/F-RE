#include <cstdio>

#include "vcall.h"

class Shape
{
   public:
    bool intersect(float t) const
    {
        return VCALL(intersect, t);
    }

   protected:
    VCALL_DECL(Shape, bool, intersect, float t);
};

class SA : public Shape
{
   public:
    SA()
    {
        VCALL_INIT(intersect);
    }

   private:
    VCALL_IMPL(Shape, bool, intersect, float t)
    {
        puts("call SA::intersect");
        return false;
    }
};
class SB : public Shape
{
   public:
    SB()
    {
        VCALL_INIT(intersect);
    }

   private:
    VCALL_IMPL(Shape, bool, intersect, float t)
    {
        puts("call SB::intersect");
        return false;
    }
};

int main()
{
    float __t = .0f;
    SA sa;
    SB sb;
    Shape * s;
    s = &sa;
    s->intersect(__t);
    s = &sb;
    s->intersect(__t);
    return 0;
}
