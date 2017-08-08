
/* ====================================================
#   Author        : Terence (Yongxin) Feng
#   Email         : tyxfeng@gmail.com
#   File Name     : RGBColor.h
#   Last Modified : 2017-03-24 16:58
#  ====================================================*/

#ifndef  _RGBCOLOR_H
#define  _RGBCOLOR_H

class RGBColor
{
public:
	float r, g, b;

    RGBColor():
        r(0.0),
        g(0.0),
        b(0.0)
    {}

    RGBColor(float c):
        r(c),
        g(c),
        b(c)
    {}

    RGBColor(float _r, float _g, float _b):
        r(_r),
        g(_g),
        b(_b)
    {}

    RGBColor(const RGBColor& c):
        r(c.r),
        g(c.g),
        b(c.b)
    {}

	RGBColor& operator = (const RGBColor& rhs)
    {
        r = rhs.r;
        g = rhs.g;
        b = rhs.b;
        return (*this);
    }

    inline RGBColor operator * (const float k) const
    {
        return RGBColor(r * k, g * k, b * k);
    }

    inline RGBColor operator * (const RGBColor& color) const
    {
        return RGBColor(r * color.r, g * color.g, b * color.b);
    }

    inline RGBColor& operator *= (const float k)
    {
        r *= k;
        g *= k;
        b *= k;
        return (*this);
    }

    inline RGBColor& operator /= (const float k)
    {
        r /= k;
        g /= k;
        b /= k;
        return (*this);
    }

    inline RGBColor& operator += (const RGBColor& color)
    {
        r += color.r;
        g += color.g;
        b += color.b;
        return (*this);
    }

    inline RGBColor operator + (const RGBColor& color)
    {
        return RGBColor(r + color.r,
                g + color.g,
                b + color.b);
    }

    inline RGBColor operator / (const float f)
    {
        return RGBColor(r / f,
                g / f,
                b / f);
    }
};

/* constants */
const RGBColor BLACK(0.0);
const RGBColor WHITE(1.0);
const RGBColor RED(1.0, 0.0, 0.0);
const RGBColor GREEN(0.0, 1.0, 0.0);
const RGBColor BLUE(0.0, 0.0, 1.0);

#endif


