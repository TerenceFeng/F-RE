
/* ====================================================
#   Copyright (C)2017 All rights reserved.
#
#   Author        : Terence (Yongxin) Feng
#   Email         : tyxfeng@gmail.com
#   File Name     : RGBColor.h
#   Last Modified : 2017-03-24 16:58
#   Describe      :
#
#   Log           :
#
# ====================================================*/

#ifndef  _RGBCOLOR_H
#define  _RGBCOLOR_H

class RGBColor
{
public:
	float r, g, b;

	RGBColor();
	RGBColor(float c);
	RGBColor(float _r, float _g, float _b);
	RGBColor(const RGBColor& c);

	RGBColor&
		operator = (const RGBColor& rhs);

	inline RGBColor
		operator * (const float k) const
		{
			return RGBColor(r * k, g * k, b * k);
		}

	inline RGBColor
		operator * (const RGBColor& color) const
		{
			return RGBColor(r * color.r, g * color.g, b * color.b);
		}

	inline RGBColor&
		operator *= (const float k)
		{
			r *= k;
			g *= k;
			b *= k;
			return (*this);
		}

	inline RGBColor&
		operator /= (const float k)
		{
			r /= k;
			g /= k;
			b /= k;
			return (*this);
		}

	inline RGBColor&
		operator += (const RGBColor& color)
		{
			r += color.r;
			g += color.g;
			b += color.b;
			return (*this);
		}

	inline RGBColor
		operator + (const RGBColor& color)
		{
			return RGBColor(r + color.r,
							g + color.g,
							b + color.b);
		}

	inline RGBColor
		operator / (const float f)
		{
			return RGBColor(r / f,
							g / f,
							b / f);
		}

	/*
	 * inline friend RGBColor
	 *     operator * (const RGBColor& color, const float k) {
	 *         return RGBColor(color.r * k, color.g * k, color.b * k);
	 *     }
	 */

};

/* constants */
const RGBColor black(0.0f);
const RGBColor red(1.0f, 0.0f, 0.0f);
const RGBColor BLACK(0.0f);
const RGBColor RED(1.0f, 0.0f, 0.0f);
const RGBColor WHITE(1.0f);

#endif // _RGBCOLOR_H


