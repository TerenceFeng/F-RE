
#include <stdio.h>

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <curand_kernel.h>

//cudaError_t addWithCuda(int *c, const int *a, const int *b, unsigned int size);
#define SAMPLE 1000
#ifdef M_PI
#undef M_PI
#endif
#define M_PI 3.141592653
#define WIDTH 1024
#define HEIGHT 768

// CUDA math built-in functions
//__device__ double rsqrt(double);
//__device__ double fmax(double, double);
//__device__ double curand_uniform_double(curandStateXORWOW_t*);

// helper functions
__device__ double clamp(double x)
{
	return x < 0.0 ? 0.0 : x > 1.0 ? 1.0 : x;
}

// Vector
struct Vec
{
	double x, y, z;
};
__device__ void vec_zero(Vec *v)
{
	v->x = v->y = v->z = 0.0;
}
__device__ void vec_assign(Vec *v, double x, double y, double z)
{
	v->x = x;
	v->y = y;
	v->z = z;
}
__device__ void vec_copy(Vec *v1, const Vec *v2)
{
	v1->x = v2->x;
	v1->y = v2->y;
	v1->z = v2->z;
}
__device__ void vec_add(Vec *result, const Vec *v1, const Vec *v2)
{
	result->x = v1->x + v2->x;
	result->y = v1->y + v2->y;
	result->z = v1->z + v2->z;
}
__device__ void vec_sub(Vec *result, const Vec *v1, const Vec *v2)
{
	result->x = v1->x - v2->x;
	result->y = v1->y - v2->y;
	result->z = v1->z - v2->z;
}
__device__ void vec_mul(Vec *result, const Vec *v1, const Vec *v2)
{
	result->x = v1->x * v2->x;
	result->y = v1->y * v2->y;
	result->z = v1->z * v2->z;
}
__device__ void vec_scale(Vec *v, double factor)
{
	v->x *= factor;
	v->y *= factor;
	v->z *= factor;
}
__device__ __host__ void vec_norm(Vec *v)
{
	double mod = rsqrt(v->x * v->x + v->y * v->y + v->z * v->z);
	v->x *= mod;
	v->y *= mod;
	v->z *= mod;
}
__device__ double vec_dot(const Vec *v1, const Vec *v2)
{
	return v1->x * v2->x + v1->y * v2->y + v1->z * v2->z;
}
__device__ void vec_cross(Vec *result, const Vec *v1, const Vec *v2)
{
	result->x = v1->y * v2->z - v1->z * v2->y;
	result->y = v1->z * v2->x - v1->x * v2->z;
	result->z = v1->x * v2->y - v1->y * v2->x;
}

// Ray
struct Ray
{
	Vec o, d;
};
__device__ void ray_assign(Ray *r, const Vec *o, const Vec *d)
{
	vec_copy(&r->o, o);
	vec_copy(&r->d, d);
}

// Sphere
enum EReflType
{
	DIFF, SPEC, REFR
};
struct Sphere
{
	double rad;       // radius
	Vec p, e, c;      // position, emission, color
	EReflType refl;      // reflection type (DIFFuse, SPECular, REFRactive)
	/* Sphere(double rad_, Vec p_, Vec e_, Vec c_, EReflType refl_) : */
	/*     rad(rad_), p(p_), e(e_), c(c_), refl(refl_) */
	/* {} */
};

// Intersection Test
__device__ double intersect_sr(const Sphere *s, const Ray *r)
{
	const double eps = 1e-4;
	Vec op;
	double b, det, t;

	vec_sub(&op, &s->p, &r->o);
	b = vec_dot(&op, &r->d);
	det = b * b - vec_dot(&op, &op) + s->rad * s->rad;
	if (det < 0.0)
	{
		return 0.0;
	}
	else
	{
		// nearest positive
		det = sqrt(det);
		t = b - det;
		if (t <= eps) t = b + det;
		return fmax(t, 0.0);
	}
}
__device__ bool intersect_all(const Ray *r, double *t, int *id, const Sphere *spheres)
{
	double d, inf;

	inf = *t = 1e20;

	for (int i = 9; i--;)
	{
		d = intersect_sr(spheres + i, r);
		if (d != 0.0 && d < *t)
		{
			*t = d; *id = i;
		}
	}
	return *t < inf;
}

__device__ void radiance(Vec *radiance, const Ray *_r, curandState *state, Sphere *spheres)
{
	/* vec_assign(radiance, 1.0, 1.0, 0.0); */
	/* return; */
	Vec prev_f = { 1.0, 1.0, 1.0 };
	Ray r = *_r;
	for (int depth = 0; ; ++depth)
	{
		double t; // distance to intersection
		const Sphere *obj; // the hit object
		{
			int id = 0;
			if (!intersect_all(&r, &t, &id, spheres))
			{
				break;
			}
			obj = spheres + id;
		}

		Vec x, n, nl, f;
		//double p;  // max refl
		{
			// x
			vec_copy(&x, &r.d);
			vec_scale(&x, t);
			vec_add(&x, &x, &r.o);
			// n
			vec_copy(&n, &x);
			vec_sub(&n, &n, &obj->p);
			vec_norm(&n);
			// nl
			if (vec_dot(&n, &r.d) >= 0.0)
				vec_scale(&n, -1.0);
			vec_copy(&nl, &n);
			// f
			vec_copy(&f, &obj->c);
			// p
			//p = f.x > f.y && f.x > f.z ? f.x : f.y > f.z ? f.y : f.z;
		}

		if (depth > 5) //R.R.
		{
			// assert( frame >= 1 );
			vec_mul(&prev_f, &prev_f, &obj->e);
			vec_add(radiance, radiance, &prev_f);
			break;

			//if (curand_uniform_double(state) < p)
			//	vec_scale(&f, 1.0 / p);
			//else
			//{
			//	vec_copy(radiance, &obj->e);
			//	return;
			//}
		}

		if (obj->refl == DIFF) // Ideal DIFFUSE reflection
		{
			Vec d;
			{
				double r1 = 2.0 * M_PI * curand_uniform_double(state);
				double r2 = curand_uniform_double(state);
				double r2s = sqrt(r2);
				Vec w, u, v;
				// w
				w = nl;
				// u
				vec_zero(&u);
				if (fabs(w.x) > 0.1) u.y = 1.0;
				else u.x = 1.0;
				vec_norm(&u);
				// v
				vec_copy(&v, &w);
				vec_cross(&v, &v, &u);
				// d
				vec_scale(&u, cos(r1) * r2s);
				vec_scale(&v, sin(r1) * r2s);
				vec_scale(&w, sqrt(1 - r2));
				vec_copy(&d, &u);
				vec_add(&d, &d, &v);
				vec_add(&d, &d, &w);
				vec_norm(&d);
			}

			vec_mul(&prev_f, &prev_f, &obj->e);
			vec_add(radiance, radiance, &prev_f);
			ray_assign(&r, &x, &d);
			prev_f = f;
			continue;
		}
		else if (obj->refl == SPEC) // Ideal SPECULAR reflection
		{
			Vec d;
			{
				d = n;
				vec_scale(&d, -2.0 * vec_dot(&n, &r.d));
				vec_add(&d, &d, &r.d);
			}
			
			vec_mul(&prev_f, &prev_f, &obj->e);
			vec_add(radiance, radiance, &prev_f);
			ray_assign(&r, &x, &d);
			prev_f = f;
			continue;
		}
		else // Ideal dielectric REFRACTION
		{
			// Not supported yet.
			break;
			//Ray reflRay(x, r.d - n * 2 * n.dot(r.d));
			//bool into = n.dot(nl) > 0;                // Ray from outside going in?
			//double nc = 1, nt = 1.5, nnt = into ? nc / nt : nt / nc, ddn = r.d.dot(nl), cos2t;
			//if ((cos2t = 1 - nnt*nnt*(1 - ddn*ddn)) < 0)    // Total internal reflection
			//	return obj.e + f.mult(radiance(reflRay, depth, state, spheres));
			//Vec tdir = (r.d*nnt - n*((into ? 1 : -1)*(ddn*nnt + sqrt(cos2t)))).norm();
			//double a = nt - nc, b = nt + nc, R0 = a*a / (b*b), c = 1 - (into ? -ddn : tdir.dot(n));
			//double Re = R0 + (1 - R0)*c*c*c*c*c, Tr = 1 - Re, P = .25 + .5*Re, RP = Re / P, TP = Tr / (1 - P);
			//return obj.e + f.mult(depth > 2 ? (curand_uniform(&state) < P ?   // Russian roulette
			//								   radiance(reflRay, depth, state, spheres)*RP : radiance(Ray(x, tdir), depth, state, spheres)*TP) :
			//					  radiance(reflRay, depth, state, spheres)*Re + radiance(Ray(x, tdir), depth, state, spheres)*Tr);
		}
	}
	
}

__global__ void radiance_kernel(Vec *c, Sphere *spheres, Ray *cam)
{
	const int w = WIDTH, h = HEIGHT;
	unsigned int seed = threadIdx.x;
	curandState state;
	curand_init(seed, 0, 0, &state);
	
	////blockIdx, blockDim, warpSize
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;

	// parameters
	Vec cx, cy;
	Vec r, d, tmp, rad;
	Ray rtmp;

	vec_assign(&cx, w*.5135 / h, .0, .0);
	vec_cross(&cy, &cx, &cam[0].d);
	vec_norm(&cy);
	vec_scale(&cy, 0.5135);

	// 2x2 subpixel
	double r1, r2, dx, dy;
	for (int sy = 0, i = y*w + x/*(h - y - 1)*w + x*/; sy < 2; sy++)
	{
		for (int sx = 0; sx < 2; sx++)
		{
			vec_zero(&r);
			for (int s = 0; s < SAMPLE; s++)
			{
				r1 = 2 * curand_uniform_double(&state);
				dx = r1 < 1 ? sqrt(r1) - 1 : 1 - sqrt(2 - r1);
				r2 = 2 * curand_uniform_double(&state);
				dy = r2 < 1 ? sqrt(r2) - 1 : 1 - sqrt(2 - r2);

				d = cx;
				tmp = cy;
				vec_scale(&d, ((sx + .5 + dx) / 2 + x) / w - .5);
				vec_scale(&tmp, ((sy + .5 + dy) / 2 + y) / h - .5);
				vec_add(&d, &d, &tmp);
				vec_add(&d, &d, &cam[0].d);

				tmp = d;
				vec_scale(&tmp, 140.0);
				vec_add(&tmp, &tmp, &cam[0].o);
				vec_norm(&d);

				ray_assign(&rtmp, &tmp, &d);

				radiance(&rad, &rtmp, &state, spheres);
				vec_scale(&rad, 1.0 / SAMPLE);
				vec_add(&r, &r, &rad);

			} // Camera rays are pushed ^^^^^ forward to start in interior
			vec_assign(&r, clamp(r.x) * 0.25, clamp(r.y) * 0.25, clamp(r.z) * 0.25);
			vec_add(c + i, c + i, &r);
		}
	}
}

__host__ cudaError_t PathTracing(Vec *c, const int w, const int h)
{
	const int size = w * h;
	Vec *dev_c;
	Sphere *spheres;
	Ray *cam;

	Sphere spheres_h[] = {//Scene: radius, position, emission, color, material
		{1e5, {1e5 + 1,40.8,81.6},  {.0,.0,.0},{.75,.25,.25},DIFF},//Left
		{1e5, {-1e5 + 99,40.8,81.6},{.0,.0,.0},{.25,.25,.75},DIFF},//Rght
		{1e5, {50,40.8, 1e5},       {.0,.0,.0},{.75,.75,.75},DIFF},//Back
		{1e5, {50,40.8,-1e5 + 170}, {.0,.0,.0},{.0,.0,.0},DIFF},//Frnt
		{1e5, {50, 1e5, 81.6},      {.0,.0,.0},{.75,.75,.75},DIFF},//Botm
		{1e5, {50,-1e5 + 81.6,81.6},{.0,.0,.0},{.75,.75,.75},DIFF},//Top
		/*
		 * {16.5,{27,16.5,47},         {.0,.0,.0},{0.999,0.999,0.999}, SPEC},//Mirr
		 * {16.5,{73,16.5,78},         {.0,.0,.0},{ 0.999,0.999,0.999 }, REFR},//Glas
		 */
		{600, {50,681.6 - .27,81.6},{12,12,12},  {.0,.0,.0}, DIFF} //Lite
	};
	Ray cam_h = { {50, 52, 295.6}, {0, -0.042612, -1} };
	vec_norm(&cam_h.d);

	cudaError_t cudaStatus;
	// Choose which GPU to run on, change this on a multi-GPU system.
	cudaStatus = cudaSetDevice(0);
	if (cudaStatus != cudaSuccess)
	{
		fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
		cudaFree(dev_c); return cudaStatus;
	}

	// Allocate GPU buffers for three vectors (two input, one output)    .
	cudaStatus = cudaMalloc((void**)&dev_c, size * sizeof(Vec));
	if (cudaStatus != cudaSuccess)
	{
		fprintf(stderr, "cudaMalloc failed!");
		cudaFree(dev_c); return cudaStatus;
	}
	cudaStatus = cudaMalloc((void**)&spheres, 9 * sizeof(Sphere));
	if (cudaStatus != cudaSuccess)
	{
		fprintf(stderr, "cudaMalloc failed!");
		cudaFree(dev_c); return cudaStatus;
	}
	cudaStatus = cudaMalloc((void**)&cam, sizeof(Ray));
	if (cudaStatus != cudaSuccess)
	{
		fprintf(stderr, "cudaMalloc failed!");
		cudaFree(dev_c); return cudaStatus;
	}

	cudaStatus = cudaMemcpy(spheres, spheres_h, 9 * sizeof(Sphere), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess)
	{
		fprintf(stderr, "cudaMemcpy failed!");
		cudaFree(dev_c); return cudaStatus;
	}
	cudaStatus = cudaMemcpy(cam, &cam_h, sizeof(Ray), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess)
	{
		fprintf(stderr, "cudaMemcpy failed!");
		cudaFree(dev_c); return cudaStatus;
	}

	// Copy input vectors from host memory to GPU buffers.

	// Launch a kernel on the GPU with one thread for each element.
	/* const int THREADS_PER_BLOCK = 16; */
	dim3 blockD(8, 8);
	/* const int BLOCK_COUNT = ((w + 15) / 4) * ((h + 15) / 4); */
	dim3 gridD((w + 15) / 8, (h + 15) / 8);
	/* radiance_kernel<<<THREADS_PER_BLOCK, BLOCK_COUNT>>>(dev_c, spheres, cam); */
	radiance_kernel<<<gridD, blockD>>>(dev_c, spheres, cam);

	// Check for any errors launching the kernel
	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess)
	{
		fprintf(stderr, "radiance_kernel() launch failed: %s\n", cudaGetErrorString(cudaStatus));
		cudaFree(dev_c); return cudaStatus;
	}

	// cudaDeviceSynchronize waits for the kernel to finish, and returns
	// any errors encountered during the launch.
	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess)
	{
		fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching radiance_kernel()!\n", cudaStatus);
		cudaFree(dev_c); return cudaStatus;
	}

	// Copy output vector from GPU buffer to host memory.
	cudaStatus = cudaMemcpy(c, dev_c, size * sizeof(Vec), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess)
	{
		fprintf(stderr, "cudaMemcpy failed!");
		cudaFree(dev_c); return cudaStatus;
	}


	return cudaStatus;
}

void draw(Vec *c, int w, int h)
{
	FILE *f;
	f = fopen("image.ppm", "w");
	fprintf(f, "P3\n%d %d\n%d\n", w, h, 255);
	for (int i = 0; i < w*h; i++)
		fprintf(f, "%d %d %d ", (int)(c[i].x * 255.0), (int)(c[i].y * 255.0), (int)(c[i].z * 255.0));
}

int main()
{
	const int w = WIDTH, h = HEIGHT;
	Vec *c = new Vec[w * h];
	// Add vectors in parallel.
	cudaError_t cudaStatus = PathTracing(c, w, h);
	if (cudaStatus != cudaSuccess)
	{
		fprintf(stderr, "PathTracing() failed!");
		return 1;
	}

	// cudaDeviceReset must be called before exiting in order for profiling and
	// tracing tools such as Nsight and Visual Profiler to show complete traces.
	cudaStatus = cudaDeviceReset();
	if (cudaStatus != cudaSuccess)
	{
		fprintf(stderr, "cudaDeviceReset failed!");
		return 1;
	}

	draw(c, w, h);
	delete[] c;

	return 0;
}

