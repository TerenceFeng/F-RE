
/* ====================================================
#   Copyright (C)2017 All rights reserved.
#   Author        : Terence (Yongxin) Feng
#   Email         : tyxfeng@gmail.com
#   File Name     : Grid.h
# ====================================================*/

#ifndef _GRID_H
#define _GRID_H

#include "BBox.h"
#include "../Utilities.h"
#include <cfloat>
#include "Object.h"

class Mesh
{
public:
	Mesh(void):
        vertices(),
        indices(),
        normals(),
        vertex_faces(),
        num_vertices(0),
        num_triangles(0),
        num_indices(0)
    {}
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
    Grid(void):
        cells(),
        bbox(),
        mesh_ptr(new Mesh),
        nx(0), ny(0), nz(0)
    {}

    virtual BBox get_bounding_box(void)
    {
        return bbox;
    }

    void setup_cells(void)
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

            // compute the cell indices for the corners of the bouding box of the object
                int ixmin = clamp((obj_bbox.x0 - p0.x) * nx / (p1.x - p0.x), 0, nx - 1);
            int iymin = clamp((obj_bbox.y0 - p0.y) * ny / (p1.y - p0.y), 0, ny - 1);
            int izmin = clamp((obj_bbox.z0 - p0.z) * nz / (p1.z - p0.z), 0, nz - 1);
            int ixmax = clamp((obj_bbox.x1 - p0.x) * nx / (p1.x - p0.x), 0, nx - 1);
            int iymax = clamp((obj_bbox.y1 - p0.y) * ny / (p1.y - p0.y), 0, ny - 1);
            int izmax = clamp((obj_bbox.z1 - p0.z) * nz / (p1.z - p0.z), 0, nz - 1);

            //  add objects to cells
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

    virtual bool hit(const Ray& ray, float& tmin, ShadeRec& sr)
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
                if (object_ptr && object_ptr->hit(ray, tmin, sr) && tmin < tx_next)
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
                    if (object_ptr && object_ptr->hit(ray, tmin, sr) && tmin < ty_next)
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
                    if (object_ptr && object_ptr->hit(ray, tmin, sr) && tmin < tz_next)
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

    virtual bool shadow_hit(const Ray& ray, float& tmin)
    {
        ShadeRec sr;
        return hit(ray, tmin, sr);
    }

    void reverse_normals()
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
	// void read_ply_file(char *);

private:
	std::vector<Object*> cells;
	BBox bbox;
	int nx, ny, nz;
	Mesh *mesh_ptr;

	Point3D min_coordinate(void)
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
	Point3D max_coordinate(void)
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
};

class MeshTriangle: public Object
{
public:
	MeshTriangle(void) {}
    MeshTriangle(Mesh *mesh_ptr_, const int i0, const int i1, const int i2):
        Object(),
        mesh_ptr(mesh_ptr_),
        index0(i0), index1(i1), index2(i2)
    {
        normal = (mesh_ptr->vertices[index1] - mesh_ptr->vertices[index0]) ^
            (mesh_ptr->vertices[index2] - mesh_ptr->vertices[index0]);
        normal.normalize();
    }

    // Normal compute_normal(const int, const int, const int);
    Normal compute_normal(bool reverse_normal)
    {
        if (reverse_normal)
            normal = -normal;
        return normal;
    }

    bool hit(const Ray& ray, float& tmin, ShadeRec& sr)
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

    bool shadow_hit(const Ray& ray, float& tmin)
    {
        ShadeRec sr;
        return hit(ray, tmin, sr);
    }

    BBox get_bounding_box(void)
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

	const Normal& get_normal(void) const
    {
        return normal;
    }
public:
	Normal normal;
private:
	Normal interpolate_normals(const float, const float);
	Mesh *mesh_ptr;
	int index0, index1, index2;
};

#endif
