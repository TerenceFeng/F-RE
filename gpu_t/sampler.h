#pragma once

#include "math.h"
#include "utilities.h"
#include <random>

namespace sampler
{
	std::random_device rd;
	std::mt19937 gen(rd());
	std::uniform_real_distribution<float> dist(0, 1);
};

static float
sample()
{
	return sampler::dist(sampler::gen);
}

static Point2D
sample_square()
{
	return Point2D(sample(), sample());
}

static Point2D
sample_disk()
{
	Point2D p = sample_square();
	float r, phi;
	float x = 2 * p.x - 1;
	float y = 2 * p.y - 1;

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
	return Point2D(r * cosf(phi), r * sinf(phi));

}

static Point3D
sample_hemisphere(const float e = 1)
{
	Point2D p = sample_square();
	float cos_phi = cosf(2 * PI * p.x);
	float sin_phi = sinf(2 * PI * p.y);
	float cos_theta = powf((1 - p.y) , 1 / (e + 1));
	float sin_theta = sqrtf(1 - cos_theta * cos_theta);
	return Point3D(sin_theta * cos_phi,
			sin_theta * sin_phi,
			cos_theta);
}

