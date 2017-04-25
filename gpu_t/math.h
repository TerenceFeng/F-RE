#pragma once

#include <random>
#include <ctime>

float
rand_float()
{
	return (float)(rand() + clock()) / (float)(RAND_MAX);
}

float
clamp(const float f)
{
	return (f > 1.0) ? 1.0 : ((f < 0.0) ? 0.0 : f);
}
