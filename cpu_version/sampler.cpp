
/* ====================================================
#   Copyright (C)2017 All rights reserved.
#   Author        : Terence (Yongxin) Feng
#   Email         : tyxfeng@gmail.com
#   File Name     : sampler.cpp
# ====================================================*/

#include "sampler.h"
#include <cstdlib>
#include <algorithm>

float
rand_float()
{
	return (float)rand() / (float)(RAND_MAX);
}

Sampler::Sampler():
	num_samples(0),
	num_sets(0),
	count(0),
	jump(0),
	samples(),
	shuffled_indices()
{}

Sampler::Sampler(int num_samples_):
	num_samples(num_samples_),
	num_sets(83),
	count(0),
	jump(0),
	samples(),
	shuffled_indices()
{}

Point2D
Sampler::sample_unit_square(void)
{
	if (count % num_samples == 0)
		jump = (rand() % num_sets) * num_samples;
	return samples[jump + count++ % num_samples];
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
		for (int i = 0; i < num_samples - 1; i++)
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
		for (int i = 0; i < num_samples - 1; i++)
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
