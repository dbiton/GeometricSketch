#include "DynamicSketch.h"

DynamicSketch::DynamicSketch(int width, int depth) : Dictionary()
{
	sketches.push_back(CM_Init(width, depth, rand()));
}

void DynamicSketch::update(uint32_t item, int f)
{
	CM_type *sketch = sketches.back();
	CM_Update(sketch, item, f);
}

int DynamicSketch::query(uint32_t item)
{
	int estimate = 0;
	for (const auto &sketch : sketches)
	{
		estimate += CM_PointEst(sketch, item);
	}
	return estimate;
}
void DynamicSketch::expand(int width)
{
	int depth = sketches.back()->depth;
	int prev_width = sketches.back()->width;
	assert(width > prev_width);
	sketches.push_back(CM_Init(width, depth, rand()));
}

int DynamicSketch::getSize() const{
    return sketches.size();
}

void DynamicSketch::shrink(int bytes)
{
	throw std::runtime_error("DynamicSketch::shrink not allowed");
}

int DynamicSketch::getMemoryUsage() const
{
	int memory_usage = sizeof(int); // depth
	for (const auto &sketch : sketches)
	{
		memory_usage += (1 + sketch->width * sketch->depth) * sizeof(int); // counters + width
	}
	return memory_usage;
}
