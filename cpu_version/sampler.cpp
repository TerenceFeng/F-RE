
/* ====================================================
#   Copyright (C)2017 All rights reserved.
#   Author        : Terence (Yongxin) Feng
#   Email         : tyxfeng@gmail.com
#   File Name     : sampler.cpp
# ====================================================*/

#include "sampler.h"
#include <ctime>
#include <cstdlib>
#include <algorithm>

#define PI 3.141592

float
rand_float()
{
	return (float)(rand() + clock()) / (float)(RAND_MAX);
}

Sampler::Sampler():
	num_samples(0),
	num_sets(0),
	count(0),
	jump(0),
	samples(),
	samples_disk(),
	shuffled_indices()
{}

Sampler::Sampler(int num_samples_):
	num_samples(num_samples_),
	num_sets(83),
	count(0),
	jump(0),
	samples(),
	samples_disk(),
	shuffled_indices()
{}

Sampler::~Sampler()
{}

Point2D
Sampler::sample_unit_square(void)
{
	if (count % num_samples == 0)
		jump = (rand() % num_sets) * num_samples;
	return samples[jump + count++ % num_samples];
}

Point2D
Sampler::sample_unit_disk(void)
{
	if (count % num_samples == 0)
		jump = (rand() % num_sets) * num_samples;
	return samples_disk[jump + count++ % num_samples];
}

Point3D
Sampler::sample_unit_hemisphere(void)
{
	if (count % num_samples == 0)
		jump = (rand() % num_sets) * num_samples;
	return samples_hemisphere[jump + count++ % num_samples];
}

void
Sampler::map_samples_to_unit_disk(void)
{
	float r, phi;
	float x, y;
	for (Point2D& p: samples)
	{
		x = 2 * p.x - 1;
		y = 2 * p.y - 1;
		if (x > -y)
		{
			if (x > y)
			{
				r = x;
				if (x != 0)
				phi = y / x;
				else
					phi = 0;
			}
			else
			{
				r = y;
				if (y != 0)
				phi = 2 - x / y;
				else
					phi = 0;
			}
		}
		else
		{
			if (x < y)
			{
				r = -x;
				if (x != 0)
				phi = 4 + y / x;
				else
					phi = 0;
			}
			else
			{
				r = y;
				if (y != 0)
					phi = 6 - x / y;
				else
					phi = 0;
			}
		}
		phi *= PI / 4.0;
		samples_disk.push_back(Point2D(r * cosf(phi), r * sinf(phi)));
	}
}

void
Sampler::map_samples_to_hemisphere(const float e = 1)
{
	for (Point2D& p: samples)
	{
		float cos_phi = cosf(2 * PI * p.x);
		float sin_phi = sinf(2 * PI * p.x);
		float cos_theta = powf((1 - p.y), 1 / (e + 1));
		float sin_theta = sqrtf(1 - cos_theta * cos_theta);
		samples_hemisphere.push_back(Point3D(sin_theta * cos_phi,
											 sin_theta * sin_phi,
											 cos_theta));
	}
}

/* Implementation of Jittered */
Jittered::Jittered():
	Sampler(0)
{}
Jittered::Jittered(int num_samples_):
	Sampler(num_samples_)
{
	generate_samples();
}

void
Jittered::generate_samples(void)
{
	int n = (int)sqrtf(num_samples) + 1;
	for (int p = 0; p < num_sets; p++)
		for (int j = 0; j < n; j++)
			for (int k = 0; k < n; k++)
			{
				Point2D sp((k + rand_float()) / n, (j + rand_float() / n));
				samples.push_back(sp);
			}
}

/* inplementation of NRooks */
NRooks::NRooks():
	Sampler()
{
	generate_samples();
}
NRooks::NRooks(int num_samples_):
	Sampler(num_samples_)
{
	generate_samples();
}

void
NRooks::generate_samples(void)
{
	for (int p = 0; p < num_sets; p++)
		for (int j = 0; j < num_samples; j++)
		{
			Point2D sp((j + rand_float()) / num_samples, (j + rand_float()) / num_samples);
			samples.push_back(sp);
		}
	shuffle_x_coordinates();
	shuffle_y_coordinates();
}

void
NRooks::shuffle_x_coordinates()
{
	for (int p = 0; p < num_sets; p++)
		for (int i = 0; i < num_samples; i++)
		{
			int target = (int)rand() % num_samples + p * num_samples;
			float temp = samples[i + p * num_samples + 1].x;
			samples[i + p * num_samples + 1].x = samples[target].x;
			samples[target].x = temp;
		}
}

void
NRooks::shuffle_y_coordinates()
{
	for (int p = 0; p < num_sets; p++)
		for (int i = 0; i < num_samples; i++)
		{
			int target = (int)rand() % num_samples + p * num_samples;
			float temp = samples[i + p * num_samples + 1].y;
			samples[i + p * num_samples + 1].y = samples[target].y;
			samples[target].y = temp;
		}
}

/* NOTE: implementation of Hammersley */
Hammersley::Hammersley(): Sampler(0){}
Hammersley::Hammersley(int num_samples_): Sampler(num_samples_)
{
	generate_samples();
}

Point2D
Hammersley::sample_unit_square(void)
{
	if (count % num_samples == 0)
		jump = (rand() % num_sets) * num_samples;
	return samples[jump + shuffled_indices[count++ % num_samples]];
}

void
Hammersley::generate_samples(void)
{
	float x = 1 / num_samples;
	for (int p = 0; p < num_sets; p++)
		for (int i = 0; i < num_samples; i++)
		{
			Point2D sp(x, phi(i));
			samples.push_back(sp);
		}
	shuffle_indices();
}

void
Hammersley::shuffle_indices(void)
{
	std::vector<int> indices;

	for (int j = 0; j < num_sets; j++)
		indices.push_back(j);
	for (int p = 0; p < num_samples; p++)
	{
		random_shuffle(indices.begin(), indices.end());
		for (int j = 0; j < num_samples; j++)
			shuffled_indices.push_back(indices[j]);
	}
}

float
Hammersley::phi(int i)
{
	float x = 0;
	float base = 0.5;
	while (i) {
		x += base * (float)(!i & 1);
		i /= 2;
		base *= 0.5;
	}
	return x;
}
