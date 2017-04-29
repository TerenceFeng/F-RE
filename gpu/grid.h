#pragma once

#include <vector>
#include <cfloat>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include "struct.h"
#include "mem.h"

struct BBox
{
    float x0 = FLT_MAX, y0 = FLT_MAX, z0 = FLT_MAX,
		  x1 = FLT_MAX, y1 = FLT_MAX, z1 = FLT_MAX;
};

struct Grid
{
    Pool<Object *> cells;
    Pool<int> cells_size;
	float x0 = FLT_MAX, y0 = FLT_MAX, z0 = FLT_MAX,
		  x1 = FLT_MAX, y1 = FLT_MAX, z1 = FLT_MAX;
	int nx, ny, nz;
    Grid():
        cells(0),
        cells_size(0)
    {}
    Grid(size_t _size):
        cells(_size),
        cells_size(_size)
    {}
};

__device__ __host__ BBox
get_bounded_box(const Object& obj)
{
    return BBox();
}

__device__ bool
inside(Grid* grid, const Vec3<float>& v)
{
    return (v.x >= grid->x0 && v.y >= grid->y0 && v.z >= grid->z0 &&
            v.x <= grid->x1 && v.y <= grid->y1 && v.z <= grid->z1);
}

__global__ void count_single_cell_kernel(Object *objs, int size, int *cells_count,
                                            int nx, int ny, int nz,
                                            int x0, int y0, int z0, int x1, int y1, int z1)
{

    int ix = blockDim.x * blockIdx.x + threadIdx.x;
    int iy = blockDim.y * blockIdx.y + threadIdx.y;
    // int iz = blockDim.z * blockIdx.z + threadIdx.z;

    /* boundry check */
    if (ix >= nx || iy >= ny/*  || iz >= nz */)
        return;

    // int index = ix + nx * iy + nx * ny * iz;
    int index = ix + nx * iy;
    int offset = nx * ny;

    while (index < size)
    {

        int local_count = 0;

        for (int i = 0; i < size; i++)
        {
            const BBox b = get_bounded_box(objs[i]);

            int ixmin = clamp((b.x0 - x0) * nx / (x1 - x0), 0, nx - 1);
            int iymin = clamp((b.y0 - y0) * ny / (y1 - y0), 0, ny - 1);
            int izmin = clamp((b.z0 - z0) * nz / (z1 - z0), 0, nz - 1);
            int ixmax = clamp((b.x1 - x0) * nx / (x1 - x0), 0, nx - 1);
            int iymax = clamp((b.y1 - y0) * ny / (y1 - y0), 0, ny - 1);
            int izmax = clamp((b.z1 - z0) * nz / (z1 - z0), 0, nz - 1);

            if (ixmin >= x0 && iymin >= y0 && izmin >= z0 &&
                    ixmax <= x1 && iymax <= y1 && izmax <= z1)
            {
                local_count += 1;
            }
        }

        index += nx * ny;
        cells_count[index] = local_count;
    }

}

__global__ void setup_single_cell_kernel(Object *objs, int size, Object **cells,
                                int nx, int ny, int nz,
                                int x0, int y0, int z0, int x1, int y1, int z1)
{
    int ix = blockDim.x * blockIdx.x + threadIdx.x;
    int iy = blockDim.y * blockIdx.y + threadIdx.y;
    // int iz = blockDim.z * blockIdx.z + threadIdx.z;

    /* boundry check */
    if (ix >= nx || iy >= ny/*  || iz >= nz */)
        return;

    // int index = ix + nx * iy + nx * ny * iz;
    int index = ix + nx * iy;
    int offset = nx * ny;

    while (index < size)
    {

        int local_count = 0;
        for (int i = 0; i < size; i++)
        {
            const BBox b = get_bounded_box(objs[i]);

            int ixmin = clamp((b.x0 - x0) * nx / (x1 - x0), 0, nx - 1);
            int iymin = clamp((b.y0 - y0) * ny / (y1 - y0), 0, ny - 1);
            int izmin = clamp((b.z0 - z0) * nz / (z1 - z0), 0, nz - 1);
            int ixmax = clamp((b.x1 - x0) * nx / (x1 - x0), 0, nx - 1);
            int iymax = clamp((b.y1 - y0) * ny / (y1 - y0), 0, ny - 1);
            int izmax = clamp((b.z1 - z0) * nz / (z1 - z0), 0, nz - 1);

            if (ixmin >= x0 && iymin >= y0 && izmin >= z0 &&
                    ixmax <= x1 && iymax <= y1 && izmax <= z1)
            {
                cells[index][local_count++] = objs[i];
            }
        }

        index += offset;

    }
}

Grid
setup_cells(std::vector<Object>& objs)
{
    std::vector<BBox> bboxs;
    BBox obj_bbox;

    Grid grid;

    for (int i = 0; i < objs.size(); i++)
    {
        obj_bbox = get_bounded_box(objs[i]);
        if (obj_bbox.x0 < grid.x0) grid.x0 = obj_bbox.x0;
        if (obj_bbox.y0 < grid.y0) grid.y0 = obj_bbox.y0;
        if (obj_bbox.z0 < grid.z0) grid.z0 = obj_bbox.z0;
        if (obj_bbox.x1 > grid.x1) grid.x1 = obj_bbox.x1;
        if (obj_bbox.y1 > grid.y1) grid.y1 = obj_bbox.y1;
        if (obj_bbox.z1 > grid.z1) grid.z1 = obj_bbox.z1;
        bboxs.push_back(obj_bbox);
    }

    int num_objects = objs.size();
    float wx = grid.x1 - grid.x0;
    float wy = grid.y1 - grid.y0;
    float wz = grid.z1 - grid.z0;
    const float multiplier = 2.0;
    float s = powf(wx * wy * wz / num_objects, 0.33333);
    grid.nx = multiplier * wx / s + 1;
    grid.ny = multiplier * wy / s + 1;
    grid.nz = multiplier * wz / s + 1;

    int num_cells = grid.nx * grid.ny * grid.nz;

    grid.cells = Pool<Object *>(num_cells);
    grid.cells_size = Pool<int>(num_cells);

    /* setup kernel dimension */
    const int BLOCK_WIDTH = 16;
    dim3 gridDim(grid.nx / BLOCK_WIDTH, grid.ny / BLOCK_WIDTH);
    dim3 blockDim(BLOCK_WIDTH, BLOCK_WIDTH);

    /* convert vector to array and copy to device*/
    Pool<Object> objs_arr(num_objects);
    objs_arr.getHost() = &objs[0];
    objs_arr.copyToDevice();


    /* count number of objects in each cell */
    count_single_cell_kernel<<<gridDim, blockDim>>>(objs_arr.getDevice(), num_objects, grid.cells_size.getDevice(),
                                                    grid.nx, grid.ny, grid.nz,
                                                    grid.x0, grid.y0, grid.z0,
                                                    grid.x1, grid.y1, grid.z1);

    grid.cells_size.copyFromDevice();


    /* allocate space for cells */
    for (int i = 0; i < num_cells; i++)
        CheckCUDAError(cudaMalloc((void **)&grid.cells.getDevice()[i], grid.cells_size.getHost()[i]));

    /* TODO: setup dimension
        * cell width*/
    setup_single_cell_kernel<<<gridDim, blockDim>>>(objs_arr.getDevice(), objs.size(), grid.cells.getDevice(),
                                                    grid.nx, grid.ny, grid.nz,
                                                    grid.x0, grid.y0, grid.z0,
                                                    grid.x1, grid.y1, grid.z1);

    for (int i = 0; i < num_cells; i++)
    {
        CheckCUDAError(cudaMemcpy((void **)&grid.cells.getHost()[i],
                                   grid.cells.getDevice()[i],
                                   grid.cells_size.getHost()[i] * sizeof(Object),
                                   cudaMemcpyDeviceToHost));
    }

    return grid;
}

/* intersect with a cell */
__device__ bool
intersect_with_cell(Object *objs, int size,
        Ray *ray, float *t, const Object* hitted_object)
{

    ComputeHit ch;
    for (int i = 0; i < size; i++)
    {
        ch.compute(ray, objs[i].shape);
        if (ch.isHit() && ch.t() < *t)
        {
            *t = ch.t();
            hitted_object = objs + i;
        }

    }

    return (hitted_object != NULL);
}

/* intersect with grid */
__device__ bool
intersect_with_grid(Grid *grid, Ray *ray,const  Object *hitted_object, float *tmin)
{
    float ox = ray->pos.x;
    float oy = ray->pos.y;
    float oz = ray->pos.z;
    float dx = ray->dir.x;
    float dy = ray->dir.y;
    float dz = ray->dir.z;

    /* TODO: shared */
    float x0 = grid->x0;
    float y0 = grid->y0;
    float z0 = grid->z0;
    float x1 = grid->x1;
    float y1 = grid->y1;
    float z1 = grid->z1;

    int nx = grid->nx;
    int ny = grid->ny;
    int nz = grid->nz;

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


    if (inside(grid, ray->pos)) {
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
    while (ix < ix_stop || iy < iy_stop || iz < iz_stop) {

        int index = ix + nx * iy + nx * ny * iz;
        Object *cell = grid->cells.getDevice()[index];
        int size = grid->cells_size.getDevice()[index];

        if (tx_next < ty_next && tx_next < tz_next)
        {
            if (intersect_with_cell(cell, size, ray, tmin, hitted_object))
            {
                return true;
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
                if (intersect_with_cell(cell, size, ray, tmin, hitted_object))
                {
                    return true;
                }

                ty_next += dty;
                iy += iy_step;
                if (iy == iy_stop)
                    return (false);
            }

            else
            {
                if (intersect_with_cell(cell, size, ray, tmin, hitted_object))
                {
                    return true;
                }
                tz_next += dtz;
                iz += iz_step;
                if (iz == iz_stop)
                    return (false);
            }
        }
    }
    return false;
}

