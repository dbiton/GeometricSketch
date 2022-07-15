#include "RingSketch.h"
#include <cassert>
#include <algorithm>
#include <iostream>
#include <string>

RingSketch::RingSketch(double epsilon, double gamma, int ring_size)
{
	uint32_t chunk_size = UINT32_MAX / ring_size;
	for (int i = 0; i < ring_size-1; i++) {
		sketchs[i*chunk_size] = new CountSketch(epsilon, gamma);
	}
	sketchs[UINT32_MAX] = new CountSketch(epsilon, gamma);
}

void RingSketch::increment(int key)
{
	sketchs[getSketchIndexForKey(key)]->addInt(key);
}

int RingSketch::query(int key)
{
	CountSketch* sketch = sketchs[getSketchIndexForKey(key)];
	return sketch->getIntFrequency(key);
}

void RingSketch::expand()
{
	uint32_t index = getFullestSketchIdx();
	auto it_next = sketchs.find(index);
	auto it_prev = it_next;
	it_next++;

	CountSketch* sketch = it_prev->second->split();

	if (it_next == sketchs.end()) {
		it_next = sketchs.begin();
		uint32_t key = it_prev->first + (UINT32_MAX - it_prev->first)/2 + it_next->first/2;
		sketchs[key] = sketch;
	}
	else {
		uint32_t key = it_next->first / 2 + it_prev->first / 2;
		sketchs[key] = sketch;
	}
}

void RingSketch::shrink()
{
	if (sketchs.size() > 1) {
		uint32_t index = getEmptiestSketchIdx();
		if (index == UINT32_MAX) {
			return;
		}
		auto it_next = sketchs.find(index);
		auto it_prev = it_next;
		it_next++;
		it_next->second->merge(*it_prev->second);
		sketchs.erase(it_prev);
	}
}

int RingSketch::numSketchs()
{
	return sketchs.size();
}

uint32_t RingSketch::getSketchIndexForKey(uint32_t k)
{
	auto s = sketchs.lower_bound(k);
	if (s == sketchs.end()) {
		s = sketchs.begin();
	}
	return s->first;
}

uint32_t RingSketch::getFullestSketchIdx()
{
	int max_count = INT_MIN;
	uint32_t max_index = UINT32_MAX;
	for (auto& pair : sketchs) {
		uint32_t cur_index = pair.first;
		CountSketch* sketch = pair.second;
		if (!sketch->isBalanced()) {
			continue;
		}
		int cur_count = sketch->getTotalCount();
		if (max_count < cur_count) {
			max_count = cur_count;
			max_index = cur_index;
		}
	}
	return max_index;
}

uint32_t RingSketch::getEmptiestSketchIdx()
{
	uint32_t min_index = UINT32_MAX;
	int min_count = INT_MAX;
	auto it = sketchs.begin();
	while(it != sketchs.end()) {
		uint32_t curr_index = it->first;
		CountSketch* curr_sketch = it->second;
		it++;
		if (it == sketchs.end()) {
			break;
		}
		uint32_t next_index = it->first;
		CountSketch* next_sketch = it->second;

		if (curr_sketch->isBalanced() || next_sketch->isBalanced()) {
			continue;
		}

		int curr_count = curr_sketch->getTotalCount() + next_sketch->getTotalCount();
		if (min_count > curr_count) {
			min_count = curr_count;
			min_index = curr_index;
		}
	}
	return min_index;
}
