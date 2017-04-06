
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

class Sampler
{
public:
	int num_samples;

	Sampler();
	Sampler(int num_samples_);
	virtual void generate_samples(void) = 0;
	void setup_shuffled_indices(void);
	Point2D sample_unit_square(void);

protected:
	int num_sets;
	std::vector<Point2D> samples;
	std::vector<int> shuffled_indices;
	unsigned long count;
	int jump;

	void shuffle_samples(void);
};

class Jittered: public Sampler
{
public:
	Jittered();
	Jittered(int);
private:
	virtual void generate_samples(void);
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
