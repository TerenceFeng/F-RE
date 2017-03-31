
/* ====================================================
#   Copyright (C)2017 All rights reserved.
#
#   Author        : Terence (Yongxin) Feng
#   Email         : tyxfeng@gmail.com
#   File Name     : Display.cpp
#   Last Modified : 2017-03-22 20:44
#   Describe      :
#
#   Log           :
#
# ====================================================*/

#include "Display.h"

#include <fcntl.h>
#include <errno.h>

#include <cstdio>
#include <cstring>

#include <vector>
using namespace std;

RGBColor
Display::max_to_one(const RGBColor& c) const
{
	float max_val = (c.r > c.g) ? ((c.r > c.b) ? c.r : c.b) : ((c.g > c.b) ? c.g : c.b);
	if (max_val > 1.0)
		return (c * (1 / max_val));
	else
		return c;
}
RGBColor
Display::clamp_to_color(const RGBColor& raw_color) const
{
	RGBColor c(raw_color);

	if (raw_color.r > 1.0 || raw_color.g > 1.0 || raw_color.b > 1.0) {
		c.r = 1.0; c.g = c.b = 0.0;
	}
	return c;
}

Display::Display(): x(200), y(200), maxval(255) {}
Display::Display(int x_, int y_): x(x_), y(y_), maxval(255) {
	pixels = vector<vector<unsigned char>>(x_);
}
Display::Display(int x_, int y_, int maxval_): x(x_), y(y_), maxval(maxval_) {}
Display::Display(const Display& d): x(d.x), y(d.y), maxval(d.maxval) {}

void
Display::add_pixel(int r, int c, const RGBColor& color_)
{
	RGBColor color = max_to_one(color_);
	pixels[r].push_back((unsigned char)(int)(color.r * 255));
	pixels[r].push_back((unsigned char)(int)(color.g * 255));
	pixels[r].push_back((unsigned char)(int)(color.b * 255));
}


void
Display::display() {
	FILE *fp;
	fp = fopen("result.ppm", "wb");
	if (!fp) {
		fprintf(stderr, "ERROR: cannot open output file: %s\n", strerror(errno));
		return;
	}

	fprintf(fp, "P6\n");
	fprintf(fp, "%d %d\n%d\n", x, y, maxval);
	for(int r = pixels.size() - 1; r >= 0; r--) {
		for(int c = 0; c < pixels[r].size(); c++) {
			fprintf(fp, "%c", pixels[r][c]);
		}
	}
	fprintf(fp, "\n");
	fclose(fp);

}
