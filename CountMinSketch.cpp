#include "CountMinSketch.h"
#include <string>
#include <cmath>
#include <algorithm>

CountMinSketch::CountMinSketch(int _width, int _depth, int _seed, int _num_hh) :
	width(_width),
	depth(_depth),
	seed(_seed),
	num_events(0),
	heavy_hitters(_num_hh, std::pair<uint32_t, uint32_t>(0,0))
{
	sketch = CM_Init(width, depth, seed);
}

CountMinSketch::CountMinSketch(const CountMinSketch& o) : 
	width(o.width),
	depth(o.depth),
	seed(o.seed),
	heavy_hitters(o.heavy_hitters),
	num_events(o.num_events),
	err_amount(o.err_amount)
{
	sketch = CM_Copy(o.sketch);
}

CountMinSketch* CountMinSketch::newEmptyLike()
{
	return new CountMinSketch(width, depth, seed, heavy_hitters.size());
}

CountMinSketch::~CountMinSketch() {
	delete sketch;
}

// Note that we do not consider negative queries for now (diff being negative)
void CountMinSketch::update(unsigned int item, int diff) {
	num_events += diff;
	int item_est = CM_PointEst(sketch, item) + diff;
	CM_Update(sketch, item, diff);
	updateHeavyHitters(item, item_est);
}

int CountMinSketch::pointEst(unsigned int query) {
	int count;
	if (isHeavyHitter(query, count)) {
		return count;
	}
	else {
		count = CM_PointEst(sketch, query);
		updateHeavyHitters(query, count);
		return count;
	}
}

int CountMinSketch::getWidth() const
{
	return width;
}

int CountMinSketch::getDepth() const
{
	return depth;
}

int CountMinSketch::getNumHH() const
{
	return heavy_hitters.size();
}

unsigned CountMinSketch::numEvents() const {
	return num_events;
}

void CountMinSketch::merge(const CountMinSketch& o)
{
	num_events = o.numEvents() + numEvents();
	
	// merge heavy hitters - we want to keep the most heavy from both sketchs
	heavy_hitters.reserve(o.heavy_hitters.size() + heavy_hitters.size());
	heavy_hitters.insert(heavy_hitters.end(), o.heavy_hitters.begin(), o.heavy_hitters.end());
	
	// used for sorting heavy hitters
	auto hh_pred = [](std::pair<uint32_t, uint32_t> hh0, std::pair<uint32_t, uint32_t> hh1) {
		return hh0.second > hh1.second;
	};
	
	std::sort(heavy_hitters.begin(), heavy_hitters.end(), hh_pred);
	// get rid of the least heavy hitters
	heavy_hitters.erase(heavy_hitters.begin() + o.heavy_hitters.size(), heavy_hitters.end());

	CM_Merge(sketch, o.sketch);
}

void CountMinSketch::clear()
{
	// set heavy hitters to an empty vector
	heavy_hitters = std::vector<std::pair<uint32_t, uint32_t>>(heavy_hitters.size(), std::pair<uint32_t, uint32_t>(0, 0));
	//CM_HalveCounts(sketch);
	CM_Clear(sketch);
	num_events = 0;
}

CountMinSketch* CountMinSketch::split(sketchFilter filter)
{
	// save heavy hitters so they won't get erased when we clear this sketch
	auto hhs = heavy_hitters;
	clear();
	CountMinSketch* o = new CountMinSketch(*this);
	o->clear();
	// init indexes for hhs in both sketchs
	int n0 = 0;
	int n1 = 0;
	for (uint32_t i = 0; i < hhs.size(); i++) {
		auto hh = hhs[i];
		// filter is given by the user, we keep only values the filter returns a non zero value for
		int select = filter(hh.first);
		if (select == 0) {
			CM_Update(o->sketch, hh.first, hh.second);
			o->num_events += hh.second;
			o->heavy_hitters[n0] = hh;
			n0++;
		}
		else {
			CM_Update(sketch, hh.first, hh.second);
			num_events += hh.second;
			heavy_hitters[n1] = hh;
			n1++;
		}
	}
	return o;
}

void CountMinSketch::updateHeavyHitters(uint32_t e, uint32_t count)
{
	uint32_t min = UINT32_MAX;
	uint32_t min_i = 0;
	for (uint32_t i = 0; i < heavy_hitters.size(); i++) {
		// if it already is a heavy hitter, we just need to update it's count and leave
		if (heavy_hitters[i].first == e) {
			heavy_hitters[i].second = count;
			return;
		}
		// also get heavy_hitter's minimum while you're at it
		if (heavy_hitters[i].second < min) {
			min = heavy_hitters[i].second;
			min_i = i;
		}
	}
	// if this value's count is larger than our smallest heavy hitter, we can replace it
	if (count > min) {
		heavy_hitters[min_i].first = e;
		heavy_hitters[min_i].second = count;
	}
}

bool CountMinSketch::isHeavyHitter(unsigned int query, int& counts)
{
	for (const auto& hh : heavy_hitters) {
		if (hh.first == query) {
			counts = hh.second;
			return true;
		}
	}
	return false;
}
