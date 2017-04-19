
/* ====================================================
#   Copyright (C)2017 All rights reserved.
#   Author        : Terence (Yongxin) Feng
#   Email         : tyxfeng@gmail.com
#   File Name     : Grid.h
# ====================================================*/

#ifndef _GRID_H
#define _GRID_H

#include "BBox.h"
#include "Utilities.h"
#include "Object.h"

class Mesh
{
public:
	Mesh(void);
public:
	std::vector<Point3D> vertices;
	std::vector<int> indices;
	std::vector<Normal> normals;
	std::vector<std::vector<int>> vertex_faces;
	// std::vector<float> u; /* u texture coordinates */
	// std::vector<float> v; /* v texture coordinates */
	int num_vertices;
	int num_triangles;
	int num_indices;
};

class Grid: public Compound
{
public:
	Grid(void);
	~Grid(void);

	virtual BBox get_bounding_box(void);
	void setup_cells(void);
	virtual bool hit(const Ray&, float&, ShadeRec&);
	virtual bool shadow_hit(const Ray&, float&);
	void reverse_normals();
	void read_ply_file(char *);

private:
	std::vector<Object*> cells;
	BBox bbox;
	int nx, ny, nz;
	Mesh *mesh_ptr;

	Point3D min_coordinate(void);
	Point3D max_coordinate(void);
};

class MeshTriangle: public Object
{
public:
	MeshTriangle(void);
	MeshTriangle(Mesh *, const int, const int, const int);
	// Normal compute_normal(const int, const int, const int);
	Normal compute_normal(bool reverse_normal);
	bool hit(const Ray&, float&, ShadeRec&);
	bool shadow_hit(const Ray&, float&);
	BBox get_bounding_box(void);
	Normal get_normal(void) const;
public:
	Normal normal;
private:
	Normal interpolate_normals(const float, const float);
	Mesh *mesh_ptr;
	int index0, index1, index2;
};

#endif
