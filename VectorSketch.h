#pragma once

#include <vector>
#include "CountMinSketch.h"

class VectorSketch
{
	std::vector<CountMinSketch> sketchs;
public:
	VectorSketch(int width, int depth, int num_hh);

	void expand();
	void update(uint32_t item, int diff);
	int query(uint32_t item);
};

