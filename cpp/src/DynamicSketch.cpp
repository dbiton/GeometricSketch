#include "DynamicSketch.h"

int get_seed(bool is_same_seed){
    if (is_same_seed){
        return 0;
    }
    else{
        int seed = rand();
        return seed;
    }
}

DynamicSketch::DynamicSketch(int width, int depth, bool _is_same_seed) : 
    Dictionary(),
    is_same_seed(_is_same_seed)
{
    sketches.push_back(CM_Init(width, depth, get_seed(is_same_seed)));
}

DynamicSketch::~DynamicSketch()
{
    while (sketches.size() > 0) {
        CM_Destroy(sketches.back());
        sketches.pop_back();
    }
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
    CM_type* sketch = CM_Init(width, depth, get_seed(is_same_seed));
    if (!sketch || width <= prev_width) {
        throw std::runtime_error("DynamicSketch runtime error");
    }
    sketches.push_back(sketch);
    return width;
}

int DynamicSketch::getSize() const{
    return (int)sketches.size();
}

void DynamicSketch::mergeCountMin(CM_type* cm0, CM_type* cm1){
    int width0 = cm0->width;
    int width1 = cm1->width;
    assert(width1 % width0 == 0);
    int factor = width1 / width0;
    int depth = cm0->depth;
    for (int row=0; row < depth; row++){
        for (int col0=0; col0<width0; col0++){
            for (int col1 = col0; col1 < width1; col1+=width0){
                cm0->counts[row][col0] += cm1->counts[row][col1];
            }
        }
    }
}


int DynamicSketch::shrink(int n)
{
    int bytes_removed = 0;
    while (getSize() > 1){
        auto sketch_large = sketches.back();
        auto sketch_small = sketches[getSize()-2];
        if (sketch_large->width <= n) {
            n -= sketch_large->width;
            bytes_removed = sketch_large->depth * sketch_large->width;
            mergeCountMin(sketch_small, sketch_large);
            CM_Destroy(sketch_large);
            sketches.pop_back();
        }
        else {
            break;
            break;
        }
    }
    return bytes_removed;
}

uint64_t DynamicSketch::getMemoryUsage() const
{
    uint64_t memory_usage = sizeof(int); // depth
	for (const auto &sketch : sketches)
	{
		memory_usage += (1 + sketch->width * sketch->depth) * sizeof(int); // counters + width
	}
	return memory_usage;
}
