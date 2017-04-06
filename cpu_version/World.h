
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
#include "GeometricObject.h"

#include <vector>

class World
{
public:
	RGBColor background_color;
	Ambient* ambient_ptr;
	std::vector<GeometricObject*> obj_ptrs;
	std::vector<Light*> light_ptrs;

	World(void);
	~World(void);

	void add_object(GeometricObject* obj);
	void add_light(Light* light_ptr);

};

#endif // _WORLD_H

