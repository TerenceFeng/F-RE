
/* ====================================================
#   Copyright (C)2017 All rights reserved.
#
#   Author        : Terence (Yongxin) Feng
#   Email         : tyxfeng@gmail.com
#   File Name     : Display.h
#   Last Modified : 2017-03-22 20:55
#   Describe      :
#
#   Log           :
#
# ====================================================*/

#ifndef  _DISPLAY_H
#define  _DISPLAY_H

#include "RGBColor.h"
#include "Utilities.h"
#include <vector>

class Display {
private:
	int x, y;
	int maxval;
	std::vector<std::vector<unsigned char>> pixels;

	RGBColor max_to_one(const RGBColor& c) const;
	RGBColor clamp_to_color(const RGBColor& raw_color) const;

public:
	Display();
	Display(int x_, int y_);
	Display(int x_, int y_, int maxval_);
	Display(const Display& d);

	inline Display&
	operator= (const Display& d) {
		x = d.x;
		y = d.y;
		maxval = d.maxval;
		pixels = d.pixels;
		return (*this);
	}

	void
	add_pixel(int r, int c, const RGBColor& color);

	void
	display();
};

#endif // _DISPLAY_H


