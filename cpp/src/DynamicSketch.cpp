#include "DynamicSketch.h"

#define IS_SAME_SEED 0

int get_seed(){
    if (IS_SAME_SEED){
        return 0xDEADBEEF;
    }
    else{
        return rand();
    }
}

DynamicSketch::DynamicSketch(int width, int depth) : Dictionary()
{
    sketches.push_back(CM_Init(width, depth, get_seed()));
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

int DynamicSketch::expand(int width)
{
	int depth = sketches.back()->depth;
	int prev_width = sketches.back()->width;
	assert(width > prev_width);
    sketches.push_back(CM_Init(width, depth, get_seed()));
    return width;
}

int DynamicSketch::getSize() const{
    return sketches.size();
}

void DynamicSketch::mergeCountMin(CM_type* cm0, CM_type* cm1){
    int width0 = cm0->width;
    int width1 = cm1->width;
    double factor = (double)width1 / (double)width0;
    int depth = cm0->depth;
    for (int row=0; row < depth; row++){
        for (int col0=0; col0<width0; col0++){
            int col1_begin =  std::floor(factor * col0);
            for (int col1 = col1_begin; col1 < factor + col1_begin; col1++){
                cm0->counts[row][col0] += cm1->counts[row][col1];
            }
        }
    }
}


int DynamicSketch::shrink(int n)
{
    int bytes_removed = 0;
    int sketchCount = getSize();
    if (sketchCount > 1){
        auto sketch_large = sketches.back();
        auto sketch_small = sketches[sketchCount-2];
        if (sketch_large->width <= n) {
            bytes_removed = sketch_large->width;
            mergeCountMin(sketch_small, sketch_large);
            delete sketch_large;
            sketches.pop_back();
        }
    }
    return bytes_removed;
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
