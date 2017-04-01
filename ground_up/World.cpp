
/* ====================================================
#   Copyright (C)2017 All rights reserved.
#
#   Author        : Terence (Yongxin) Feng
#   Email         : tyxfeng@gmail.com
#   File Name     : World.cpp
#   Last Modified : 2017-03-21 17:09
#   Describe      :
#
#   Log           :
#
# ====================================================*/

#include "World.h"

#include <fcntl.h>
#include <errno.h>
#include <cstdio>
#include <cfloat>
#include <cstring>
#include <iostream>
using namespace std;

World::World(void):
	background_color(black)
{}

World::~World(void)
{
	for (int i = 0; i < obj_ptrs.size(); i++) {
		delete obj_ptrs[i];
	}
	delete ambient_ptr;
	for (int i = 0; i < light_ptrs.size(); i++) {
		delete light_ptrs[i];
	}
}

void
World::
add_object(GeometricObject* obj_ptr)
{
	obj_ptrs.push_back(obj_ptr);
}

void
World::add_light(Light* light_)
{
	light_ptrs.push_back(light_);
}

