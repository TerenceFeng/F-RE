
/* ====================================================
#   Copyright (C)2017 All rights reserved.
#   Author        : Terence (Yongxin) Feng
#   Email         : tyxfeng@gmail.com
#   File Name     : sampler.h
# ====================================================*/

#ifndef _SAMPLER_H
#define _SAMPLER_H

#include "Utilities.h"
#include <vector>

float rand_float();

class Sampler
{
public:
	int num_samples;

	Sampler();
	Sampler(int num_samples_);
	virtual ~Sampler();
	virtual void generate_samples(void) = 0;
	void setup_shuffled_indices(void);
	void map_samples_to_unit_disk(void);
	void map_samples_to_hemisphere(const float);
	Point2D sample_unit_square(void);
	Point2D sample_unit_disk(void);
	Point3D sample_unit_hemisphere(void);

protected:
	int num_sets;
	std::vector<Point2D> samples;
	std::vector<Point2D> samples_disk;
	std::vector<Point3D> samples_hemisphere;
	std::vector<int> shuffled_indices;
	unsigned long count;
	int jump;

	void shuffle_samples(void);
};

class NRooks: public Sampler
{
public:
	NRooks();
	NRooks(int);
private:
	virtual void generate_samples(void);
	void shuffle_x_coordinates();
	void shuffle_y_coordinates();
};

class Hammersley: public Sampler
{
public:
	Hammersley();
	Hammersley(int);
	Point2D sample_unit_square(void);
private:
	virtual void generate_samples(void);
	void shuffle_indices(void);
	float phi(int);
};

#endif
