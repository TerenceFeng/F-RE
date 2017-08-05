
/* ====================================================
#   Copyright (C)2017 All rights reserved.
#
#   Author        : Terence (Yongxin) Feng
#   Email         : tyxfeng@gmail.com
#   File Name     : World.h
#   Last Modified : 2017-03-21 17:08
#   Describe      :
#
#   Log           :
#
# ====================================================*/

#ifndef  _WORLD_H
#define  _WORLD_H

#include "Light.h"
#include "RGBColor.h"
#include "Utilities.h"
#include "object/Object.h"

#include <vector>

class World
{
public:
	RGBColor background_color;
	std::vector<Object *> obj_ptrs;
	std::vector<Light *> light_ptrs;
	AmbientOccluder *ambient_ptr;

    World(void):
        background_color(BLACK)
    {}

    ~World(void)
    {
        for (int i = 0; i < obj_ptrs.size(); i++) {
            delete obj_ptrs[i];
        }
        for (int i = 0; i < light_ptrs.size(); i++) {
            delete light_ptrs[i];
        }
    }

    void add_object(Object *obj_ptr)
    {
        obj_ptrs.push_back(obj_ptr);
    }
    void add_light(Light *light_ptr)
    {
        light_ptrs.push_back(light_ptr);
    }


};

#endif // _WORLD_H


