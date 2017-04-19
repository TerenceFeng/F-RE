
/* ====================================================
#   Author        : Terence (Yongxin) Feng
#   Email         : tyxfeng@gmail.com
#   File Name     : World.cpp
#   Last Modified : 2017-03-21 17:09
# ====================================================*/

#include "World.h"

World::World(void):
	background_color(BLACK)
{}

World::~World(void)
{
	for (int i = 0; i < obj_ptrs.size(); i++) {
		delete obj_ptrs[i];
	}
	for (int i = 0; i < light_ptrs.size(); i++) {
		delete light_ptrs[i];
	}
}

void
World::
add_object(Object* obj_ptr)
{
	obj_ptrs.push_back(obj_ptr);
}

void
World::add_light(Light* light_)
{
	light_ptrs.push_back(light_);
}

