#include "VectorSketch.h"
constexpr uint32_t SEED = 0x1337C0D3;

VectorSketch::VectorSketch(int width, int depth, int num_hh)
{
	sketchs.push_back(CountMinSketch(width, depth, SEED, num_hh));
}

void VectorSketch::expand()
{
	sketchs.push_back(CountMinSketch(sketchs[0].getWidth(), sketchs[0].getDepth(), SEED, sketchs[0].getNumHH()));
}

void VectorSketch::update(uint32_t item, int diff)
{
	auto sketch = sketchs[0];
	for (int i = 1; i < sketchs.size(); i++) if (sketchs[i].numEvents() < sketch.numEvents()) sketch = sketchs[i];
	sketch.update(item, diff);
}

int VectorSketch::query(uint32_t item)
{
	int res = 0;
	for (auto& s : sketchs) res += s.pointEst(item);
	return res;
}

