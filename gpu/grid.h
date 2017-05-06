#pragma once

#include <vector>
#include "struct.h"
#include "mem.h"

__device__ __host__ float clamp(const float d, const float min, const float max)
{
    return d > max ? max : (d < min ? min : d);
}


class Grid
{
public:
	Pool<Object *> cells;
    Pool<int> cells_size;
	float x0 = 1e10, y0 = 1e10, z0 = 1e10,
		  x1 = -1e10, y1 = -1e10, z1 = -1e10;
	int nx, ny, nz;
    Grid():
        cells(0),
        cells_size(0)
    {}

    Grid(Grid &rhs):
        cells(rhs.cells),
        cells_size(rhs.cells_size),
        x0(rhs.x0), y0(rhs.y0), z0(rhs.z0),
        x1(rhs.x1), y1(rhs.y1), z1(rhs.z1),
        nx(rhs.nx), ny(rhs.ny), nz(rhs.nz)
    {}

    Grid (Grid &&rhs):
        cells(rhs.cells),
        cells_size(rhs.cells_size),
        x0(rhs.x0), y0(rhs.y0), z0(rhs.z0),
        x1(rhs.x1), y1(rhs.y1), z1(rhs.z1),
        nx(rhs.nx), ny(rhs.ny), nz(rhs.nz)
    {}

    Grid &operator = (Grid &rhs)
    {
		cells = rhs.cells;
		cells_size = rhs.cells_size;
		x0 = rhs.x0; y0 = rhs.y0; z0 = rhs.z0;
		x1 = rhs.x1; y1 = rhs.y1; z1 = rhs.z1;
		nx = rhs.nx; ny = rhs.ny; nz = rhs.nz;
        return (*this);
    }

    Grid &operator = (Grid &&rhs)
    {
		cells = rhs.cells;
		cells_size = rhs.cells_size;
		x0 = rhs.x0; y0 = rhs.y0; z0 = rhs.z0;
		x1 = rhs.x1; y1 = rhs.y1; z1 = rhs.z1;
		nx = rhs.nx; ny = rhs.ny; nz = rhs.nz;
        return (*this);
    }

    // Grid(Pool<Object>& objs):
    Grid(Object *objs, int num_objects, const BBox bbox, std::vector<BBox>& bboxs):
        cells(0),
        cells_size(0),
        x0(bbox.x0), y0(bbox.y0), z0(bbox.z0),
        x1(bbox.x1), y1(bbox.y1), z1(bbox.z1)
    {
        // std::vector<BBox> bboxs;
        BBox obj_bbox;

        float wx = x1 - x0;
        float wy = y1 - y0;
        float wz = z1 - z0;
        const float multiplier = 2.0;
        float s = powf(wx * wy * wz / num_objects, 0.33333);
        nx = multiplier * wx / s + 1;
        ny = multiplier * wy / s + 1;
        nz = multiplier * wz / s + 1;

        int num_cells = nx * ny * nz;

        /* construct cells and cells_size */

        cells = Pool<Object *>(num_cells, IN_HOST | IN_DEVICE);
        cells_size = Pool<int>(num_cells, IN_HOST | IN_DEVICE);


        std::vector<std::vector<Object>> _cells(num_cells, std::vector<Object>());
        int index;

        for (int i = 0; i < num_objects; i++)
        {
            int ixmin = clamp((bboxs[i].x0 - x0) * nx / (x1 - x0), 0, nx - 1);
            int iymin = clamp((bboxs[i].y0 - y0) * ny / (y1 - y0), 0, ny - 1);
            int izmin = clamp((bboxs[i].z0 - z0) * nz / (z1 - z0), 0, nz - 1);
            int ixmax = clamp((bboxs[i].x1 - x0) * nx / (x1 - x0), 0, nx - 1);
            int iymax = clamp((bboxs[i].y1 - y0) * ny / (y1 - y0), 0, ny - 1);
            int izmax = clamp((bboxs[i].z1 - z0) * nz / (z1 - z0), 0, nz - 1);

            for (int iz = izmin; iz <= izmax ; iz++)
                for (int iy = iymin; iy <= iymax; iy++)
                    for (int ix = ixmin; ix <= ixmax; ix++)
                    {
                        index = ix + nx * iy + nx * ny * iz;
                        _cells[index].push_back(objs[i]);
                    }
        }


        for (int i = 0; i < num_cells; i++)
        {
            cells_size.getHost()[i] = _cells[i].size();
            if (_cells[i].empty())
                cells.getHost()[i] = nullptr;
            else
            {
                CheckCUDAError(cudaMalloc((void **)&cells.getHost()[i], _cells[i].size() * sizeof(Object)));
                CheckCUDAError(cudaMemcpy(cells.getHost()[i], &_cells[i][0], _cells[i].size() * sizeof(Object), cudaMemcpyHostToDevice));
            }

        }

        cells_size.copyToDevice();
        cells.copyToDevice();

    }
};

/* intersect with a cell */
__device__ int
intersect_with_cell(Object *objs, int size,
        Ray *ray, float *t)
{
    int hit_index = -1;
    if (objs != nullptr)
    {
        ComputeHit ch;
        for (int i = 0; i < size; i++)
        {
            ch.compute(ray, objs[i].shape);
            if (ch.isHit() && ch.t() < *t)
            {
                *t = ch.t();
                hit_index = i;
            }

        }
    }
    return hit_index;
}

/* intersect with grid */
__device__ void
intersect_with_grid(Object **obj, Object **cells, int* cells_size,
                    Ray *ray, float *tmin,
                    float x0, float y0, float z0,
                    float x1, float y1, float z1,
                    int nx, int ny, int nz)
{
    float ox = ray->pos.x;
    float oy = ray->pos.y;
    float oz = ray->pos.z;
    float dx = ray->dir.x;
    float dy = ray->dir.y;
    float dz = ray->dir.z;

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
    {
        *obj = nullptr;
        return;
    }

    /* initial cell coordinates */
    int ix, iy, iz;


    if (ox >= x0 && oy >= y0 && oz >= z0 &&
        ox <= x1 && oz <= y1 && oz <= z1)
    {
        ix = clamp((ox - x0) * nx / (x1 - x0), 0, nx - 1);
        iy = clamp((oy - y0) * ny / (y1 - y0), 0, ny - 1);
        iz = clamp((oz - z0) * nz / (z1 - z0), 0, nz - 1);
    }
    else {
        Point p = ray->pos + Vector::Scale(ray->dir, t0);
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
        tx_next = 1e10;
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
        ty_next = 1e10;
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
        tz_next = 1e10;
        iz_step = -1;
        iz_stop = -1;
    }

    int hit_index = -1;
    /* traverse the grid */
    while (true) {

        int index = ix + nx * iy + nx * ny * iz;
        Object *cell = cells[index];
        int size = cells_size[index];

        if (tx_next < ty_next && tx_next < tz_next)
        {
            hit_index = intersect_with_cell(cell, size, ray, tmin);
            if (hit_index != -1)
            {
                *obj = cell + hit_index;
                return;
            }

            tx_next += dtx;
            ix += ix_step;
            if (ix == ix_stop)
            {
                *obj = nullptr;
                return;
            }
        }

        else
        {
            if (ty_next < tz_next)
            {
                hit_index = intersect_with_cell(cell, size, ray, tmin);
                if (hit_index != -1)
                {
                    *obj = cell + hit_index;
                    return;
                }

                ty_next += dty;
                iy += iy_step;
                if (iy == iy_stop)
                {
                    *obj = nullptr;
                    return;
                }
            }

            else
            {
                hit_index = intersect_with_cell(cell, size, ray, tmin);
                if (hit_index != -1)
                {
                    *obj = cell + hit_index;
                    return;
                }

                tz_next += dtz;
                iz += iz_step;
                if (iz == iz_stop)
                {
                    *obj = nullptr;
                    return;
                }
            }
        }
    }
}

