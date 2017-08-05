
/* ====================================================
#   Copyright (C)2017 All rights reserved.
#   Author        : Terence (Yongxin) Feng
#   Email         : tyxfeng@gmail.com
#   File Name     : Grid.cpp
# ====================================================*/

#include "../ply.h"
#include "Grid.h"
#include <cfloat>
#include <iostream>

// inline float
// clamp(float x, float min, float max)
// {
//     return (x < min ? min : (x > max ? max : x));
// }

Mesh::Mesh(void):
	vertices(),
	indices(),
	normals(),
	vertex_faces(),
	num_vertices(0),
	num_triangles(0),
	num_indices(0)
{}

/* NOTE: implementation of Grid */
Grid::Grid(void):
	cells(),
	bbox(),
	mesh_ptr(new Mesh),
	nx(0), ny(0), nz(0)
{}

Grid::~Grid(void)
{}

BBox
Grid::get_bounding_box(void)
{
	return bbox;
}

bool
Grid::hit(const Ray& ray, float& t, ShadeRec& sr)
{
	float ox = ray.o.x;
	float oy = ray.o.y;
	float oz = ray.o.z;
	float dx = ray.d.x;
	float dy = ray.d.y;
	float dz = ray.d.z;

	float x0 = bbox.x0;
	float y0 = bbox.y0;
	float z0 = bbox.z0;
	float x1 = bbox.x1;
	float y1 = bbox.y1;
	float z1 = bbox.z1;

	float tx_min, ty_min, tz_min;
	float tx_max, ty_max, tz_max;

	/* the following code includes modifications from Shirley and Morley (2003) */

	float a = 1.0 / dx;
	if (a >= 0) {
		tx_min = (x0 - ox) * a;
		tx_max = (x1 - ox) * a;
	}
	else {
		tx_min = (x1 - ox) * a;
		tx_max = (x0 - ox) * a;
	}

	float b = 1.0 / dy;
	if (b >= 0) {
		ty_min = (y0 - oy) * b;
		ty_max = (y1 - oy) * b;
	}
	else {
		ty_min = (y1 - oy) * b;
		ty_max = (y0 - oy) * b;
	}

	float c = 1.0 / dz;
	if (c >= 0) {
		tz_min = (z0 - oz) * c;
		tz_max = (z1 - oz) * c;
	}
	else {
		tz_min = (z1 - oz) * c;
		tz_max = (z0 - oz) * c;
	}

	float t0, t1;

	if (tx_min > ty_min)
		t0 = tx_min;
	else
		t0 = ty_min;

	if (tz_min > t0)
		t0 = tz_min;

	if (tx_max < ty_max)
		t1 = tx_max;
	else
		t1 = ty_max;

	if (tz_max < t1)
		t1 = tz_max;

	if (t0 > t1)
		return(false);

	/* initial cell coordinates */
	int ix, iy, iz;


	if (bbox.inside(ray.o)) {
		ix = clamp((ox - x0) * nx / (x1 - x0), 0, nx - 1);
		iy = clamp((oy - y0) * ny / (y1 - y0), 0, ny - 1);
		iz = clamp((oz - z0) * nz / (z1 - z0), 0, nz - 1);
	}
	else {
		Point3D p = ray.o + ray.d * t0;
		ix = clamp((p.x - x0) * nx / (x1 - x0), 0, nx - 1);
		iy = clamp((p.y - y0) * ny / (y1 - y0), 0, ny - 1);
		iz = clamp((p.z - z0) * nz / (z1 - z0), 0, nz - 1);
	}

	/* ray parameter increments per cell in the x, y, and z directions */
	float dtx = (tx_max - tx_min) / nx;
	float dty = (ty_max - ty_min) / ny;
	float dtz = (tz_max - tz_min) / nz;

	float 	tx_next, ty_next, tz_next;
	int 	ix_step, iy_step, iz_step;
	int 	ix_stop, iy_stop, iz_stop;

	if (dx > 0) {
		tx_next = tx_min + (ix + 1) * dtx;
		ix_step = +1;
		ix_stop = nx;
	}
	else {
		tx_next = tx_min + (nx - ix) * dtx;
		ix_step = -1;
		ix_stop = -1;
	}

	if (dx == 0.0) {
		tx_next = FLT_MAX;
		ix_step = -1;
		ix_stop = -1;
	}


	if (dy > 0) {
		ty_next = ty_min + (iy + 1) * dty;
		iy_step = +1;
		iy_stop = ny;
	}
	else {
		ty_next = ty_min + (ny - iy) * dty;
		iy_step = -1;
		iy_stop = -1;
	}

	if (dy == 0.0) {
		ty_next = FLT_MAX;
		iy_step = -1;
		iy_stop = -1;
	}

	if (dz > 0) {
		tz_next = tz_min + (iz + 1) * dtz;
		iz_step = +1;
		iz_stop = nz;
	}
	else {
		tz_next = tz_min + (nz - iz) * dtz;
		iz_step = -1;
		iz_stop = -1;
	}

	if (dz == 0.0) {
		tz_next = FLT_MAX;
		iz_step = -1;
		iz_stop = -1;
	}

	/* traverse the grid */
	while (true) {
		Object* object_ptr = cells[ix + nx * iy + nx * ny * iz];

		if (tx_next < ty_next && tx_next < tz_next)
		{
			if (object_ptr && object_ptr->hit(ray, t, sr) && t < tx_next)
			{
				material_ptr = object_ptr->material_ptr;
				return (true);
			}
			tx_next += dtx;
			ix += ix_step;
			if (ix == ix_stop)
				return (false);
		}
		else
		{
			if (ty_next < tz_next)
			{
				if (object_ptr && object_ptr->hit(ray, t, sr) && t < ty_next)
				{
					material_ptr = object_ptr->material_ptr;
					return (true);
				}
				ty_next += dty;
				iy += iy_step;
				if (iy == iy_stop)
					return (false);
		 	}
		 	else
			{
				if (object_ptr && object_ptr->hit(ray, t, sr) && t < tz_next)
				{
					material_ptr = object_ptr->material_ptr;
					return (true);
				}
				tz_next += dtz;
				iz += iz_step;
				if (iz == iz_stop)
					return (false);
		 	}
		}
	}
}

bool
Grid::shadow_hit(const Ray& ray, float& t)
{
	ShadeRec sr;
	return hit(ray, t, sr);
}

/* NOTE: following code is cited from "Ray Tricing from the Ground Up" */
void
Grid::read_ply_file(char *file_name)
{
	// Vertex definition
	typedef struct Vertex
	{
	  float x,y,z;      // space coordinates
	} Vertex;

	// Face definition. This is the same for all files but is placed here to keep all the definitions together
	typedef struct Face
	{
	  	unsigned char nverts;    // number of vertex indices in list
	  	int* verts;              // vertex index list
	} Face;

	// list of property information for a vertex
	// this varies depending on what you are reading from the file
	PlyProperty vert_props[] =
	{
	  {"x", PLY_FLOAT, PLY_FLOAT, offsetof(Vertex,x), 0, 0, 0, 0},
	  {"y", PLY_FLOAT, PLY_FLOAT, offsetof(Vertex,y), 0, 0, 0, 0},
	  {"z", PLY_FLOAT, PLY_FLOAT, offsetof(Vertex,z), 0, 0, 0, 0}
	};

	// list of property information for a face.
	// there is a single property, which is a list
	// this is the same for all files
	PlyProperty face_props[] =
	{
	  	{"vertex_indices", PLY_INT, PLY_INT, offsetof(Face,verts),
	   		1, PLY_UCHAR, PLY_UCHAR, offsetof(Face,nverts)}
	};

	// local variables
	int 			i,j;
  	PlyFile*		ply;
  	int 			nelems;		// number of element types: 2 in our case - vertices and faces
  	char**			elist;
	int 			file_type;
	float 			version;
	int 			nprops;		// number of properties each element has
	int 			num_elems;	// number of each type of element: number of vertices or number of faces
	PlyProperty**	plist;
	Vertex**		vlist;
	Face**			flist;
	char*			elem_name;
	int				num_comments;
	char**			comments;
	int 			num_obj_info;
	char**			obj_info;


  	// open a ply file for reading
	ply = ply_open_for_reading(file_name, &nelems, &elist, &file_type, &version);

  	// go through each kind of element that we learned is in the file and read them
  	for (i = 0; i < nelems; i++)
	{  // there are only two elements in our files: vertices and faces
	    // get the description of the first element

  	    elem_name = elist[i];
	    plist = ply_get_element_description (ply, elem_name, &num_elems, &nprops);

	    // if we're on vertex elements, read in the properties

    	if (equal_strings ("vertex", elem_name))
		{
	      	// set up for getting vertex elements
	      	// the three properties are the vertex coordinates

			ply_get_property (ply, elem_name, &vert_props[0]);
	      	ply_get_property (ply, elem_name, &vert_props[1]);
		  	ply_get_property (ply, elem_name, &vert_props[2]);

		  	// reserve mesh elements

		  	mesh_ptr->num_vertices = num_elems;
		  	mesh_ptr->vertices.reserve(num_elems);

		  	// grab all the vertex elements

		  	for (j = 0; j < num_elems; j++) {
				Vertex* vertex_ptr = new Vertex;

		        // grab an element from the file

				ply_get_element (ply, (void *) vertex_ptr);
		  		mesh_ptr->vertices.push_back(Point3D(vertex_ptr->x, vertex_ptr->y, vertex_ptr->z));
		  		delete vertex_ptr;
		  	}
    	}

	    // if we're on face elements, read them in

	    if (equal_strings ("face", elem_name))
		{
		    // set up for getting face elements

			ply_get_property (ply, elem_name, &face_props[0]);   // only one property - a list

		  	mesh_ptr->num_triangles = num_elems;
		  	object_ptrs.reserve(num_elems);  // triangles will be stored in Compound::objects

			// the following code stores the face numbers that are shared by each vertex

		  	mesh_ptr->vertex_faces.reserve(mesh_ptr->num_vertices);
			std::vector<int> faceList;

		  	for (j = 0; j < mesh_ptr->num_vertices; j++)
		  		mesh_ptr->vertex_faces.push_back(faceList); // store empty lists so that we can use the [] notation below

			// grab all the face elements

			int count = 0; // the number of faces read

			for (j = 0; j < num_elems; j++)
			{
			    // grab an element from the file
			    Face* face_ptr = new Face;

			    ply_get_element (ply, (void *) face_ptr);

			    // construct a mesh triangle of the specified type
			    	MeshTriangle* triangle_ptr = new MeshTriangle(mesh_ptr, face_ptr->verts[0], face_ptr->verts[1], face_ptr->verts[2]);
					// triangle_ptr->compute_normal(false); 	// the "flat triangle" normal is used to compute the average normal at each mesh vertex
					object_ptrs.push_back(triangle_ptr); 				// it's quicker to do it once here, than have to do it on average 6 times in compute_mesh_normals

					// the following code stores a list of all faces that share a vertex
					// it's used for computing the average normal at each vertex in order(num_vertices) time

					mesh_ptr->vertex_faces[face_ptr->verts[0]].push_back(count);
					mesh_ptr->vertex_faces[face_ptr->verts[1]].push_back(count);
					mesh_ptr->vertex_faces[face_ptr->verts[2]].push_back(count);
					count++;
			}
		}
	}  // end of for (i = 0; i < nelems; i++)

	// grab and print out the comments in the file
	comments = ply_get_comments (ply, &num_comments);

	// grab and print out the object information
/*
 *     obj_info = ply_get_obj_info (ply, &num_obj_info);
 * 
 *     for (i = 0; i < num_obj_info; i++)
 *         printf ("obj_info = '%s'\n", obj_info[i]);
 * 
 */
	// close the ply file
	ply_close (ply);
}

void
Grid::setup_cells(void)
{
	Point3D p0 = min_coordinate();
	Point3D p1 = max_coordinate();
	bbox.x0 = p0.x; bbox.y0 = p0.y; bbox.z0 = p0.z;
	bbox.x1 = p1.x; bbox.y1 = p1.y; bbox.z1 = p1.z;

	int num_objects = object_ptrs.size();
	float wx = p1.x - p0.x;
	float wy = p1.y - p0.y;
	float wz = p1.z - p0.z;
	const float multiplier = 2.0;
	float s = powf(wx * wy * wz / num_objects, 0.33333);
	nx = multiplier * wx / s + 1;
	ny = multiplier * wy / s + 1;
	nz = multiplier * wz / s + 1;

	int num_cells = nx * ny * nz;
	cells.reserve(num_cells);
	for (int i = 0; i < num_cells; i++)
	{
		cells.push_back(nullptr);
	}

	std::vector<int> count(num_cells, 0);

	BBox obj_bbox;
	int index;

	for (Object *obj_ptr: object_ptrs)
	{
		obj_bbox = obj_ptr->get_bounding_box();

		/* compute the cell indices for the corners of the bouding box of the object */
		int ixmin = clamp((obj_bbox.x0 - p0.x) * nx / (p1.x - p0.x), 0, nx - 1);
		int iymin = clamp((obj_bbox.y0 - p0.y) * ny / (p1.y - p0.y), 0, ny - 1);
		int izmin = clamp((obj_bbox.z0 - p0.z) * nz / (p1.z - p0.z), 0, nz - 1);
		int ixmax = clamp((obj_bbox.x1 - p0.x) * nx / (p1.x - p0.x), 0, nx - 1);
		int iymax = clamp((obj_bbox.y1 - p0.y) * ny / (p1.y - p0.y), 0, ny - 1);
		int izmax = clamp((obj_bbox.z1 - p0.z) * nz / (p1.z - p0.z), 0, nz - 1);

		/* add objects to cells */
		for (int iz = izmin; iz <= izmax; iz++)
			for (int iy = iymin; iy <= iymax; iy++)
				for (int ix = ixmin; ix <= ixmax; ix++)
				{
					index = ix + nx * iy + nx * ny * iz;

					if (count[index] == 0)
					{
						cells[index] = obj_ptr;
						count[index] += 1;
					}
					else
					{
						if (count[index] == 1)
						{
							Compound *compound_ptr = new Compound;
							compound_ptr->add_object(cells[index]);
							compound_ptr->add_object(obj_ptr);
							cells[index] = compound_ptr;
							count[index] += 1;
						}
						else
						{
							((Compound *)cells[index])->add_object(obj_ptr);
							count[index] += 1;
						}
					}
				}
	}
	count.erase(count.begin(), count.end());
}

Point3D
Grid::min_coordinate(void)
{
	BBox obj_bbox;
	Point3D p0(FLT_MAX);

	for (Object *obj_ptr: object_ptrs)
	{
		obj_bbox = obj_ptr->get_bounding_box();
		if (obj_bbox.x0 < p0.x) p0.x = obj_bbox.x0;
		if (obj_bbox.y0 < p0.y) p0.y = obj_bbox.y0;
		if (obj_bbox.z0 < p0.z) p0.z = obj_bbox.z0;
	}
	p0.x -= eps; p0.y -= eps; p0.z -= eps;
	bbox.x0 = p0.x; bbox.y0 = p0.y; bbox.z0 = p0.z;
	return p0;
}

Point3D
Grid::max_coordinate(void)
{
	BBox obj_bbox;
	Point3D p1(FLT_MIN);
	for (Object *obj_ptr: object_ptrs)
	{
		obj_bbox = obj_ptr->get_bounding_box();
		if (obj_bbox.x1 > p1.x) p1.x = obj_bbox.x1;
		if (obj_bbox.y1 > p1.y) p1.y = obj_bbox.y1;
		if (obj_bbox.z1 > p1.z) p1.z = obj_bbox.z1;
	}
	p1.x += eps; p1.y += eps; p1.z += eps;
	bbox.x1 = p1.x; bbox.y1 = p1.y; bbox.z1 = p1.z;
	return p1;
}

/* NOTE: implementation of MeshTriangle */
MeshTriangle::MeshTriangle(void)
{}

MeshTriangle::MeshTriangle(Mesh *mesh_ptr_, const int i0, const int i1, const int i2):
	Object(),
	mesh_ptr(mesh_ptr_),
	index0(i0), index1(i1), index2(i2)
{
	normal = (mesh_ptr->vertices[index1] - mesh_ptr->vertices[index0]) ^
		   (mesh_ptr->vertices[index2] - mesh_ptr->vertices[index0]);
	normal.normalize();
}

Normal
MeshTriangle::compute_normal(bool reverse_normal)
{
	if (reverse_normal)
		normal = -normal;
	return normal;
}

bool
MeshTriangle::hit(const Ray& ray, float& tmin, ShadeRec& sr)
{

	Point3D v0 = mesh_ptr->vertices[index0];
	Point3D v1 = mesh_ptr->vertices[index1];
	Point3D v2 = mesh_ptr->vertices[index2];

	float a = v0.x - v1.x, b = v0.x - v2.x, c = ray.d.x, d = v0.x - ray.o.x;
	float e = v0.y - v1.y, f = v0.y - v2.y, g = ray.d.y, h = v0.y - ray.o.y;
	float i = v0.z - v1.z, j = v0.z - v2.z, k = ray.d.z, l = v0.z - ray.o.z;

	float m = f * k - g * j, n = h * k - g * l, p = f * l - h * j;
	float q = g * i - e * k, s = e * j - f * i;

	float inv_denom = 1.0 / (a * m + b * q + c * s);

	float e1 = d * m - b * n - c * p;
	float beta = e1 * inv_denom;

	if (beta < 0)
		return false;

	float r = e * l - h * i;
	float e2 = a * n + d * q + c * r;
	float gamma = e2 * inv_denom;

	if (gamma < 0)
		return false;

	if (beta + gamma > 1)
		return false;

	float e3 = a * p - b * r + d * s;
	float t = e3 * inv_denom;

	if (t < eps)
		return false;

	tmin = t;
	sr.normal = normal;
	sr.local_hit_point = ray.o + ray.d * t;
	return true;
}

bool
MeshTriangle::shadow_hit(const Ray& ray, float& tmin)
{
	ShadeRec sr;
	return hit(ray, tmin, sr);
}

BBox
MeshTriangle::get_bounding_box(void)
{
	Point3D v0 = mesh_ptr->vertices[index0];
	Point3D v1 = mesh_ptr->vertices[index1];
	Point3D v2 = mesh_ptr->vertices[index2];

	return BBox(std::min(std::min(v0.x, v1.x), v2.x),
				std::min(std::min(v0.y, v1.y), v2.y),
				std::min(std::min(v0.z, v1.z), v2.z),
				std::max(std::max(v0.x, v1.x), v2.x),
				std::max(std::max(v0.y, v1.y), v2.y),
				std::max(std::max(v0.z, v1.z), v2.z));
}

Normal
MeshTriangle::get_normal(void) const
{
	return normal;
}
