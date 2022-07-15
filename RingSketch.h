#pragma once

#include "CountSketch.h"

#include <map>


class RingSketch {
	std::map<uint32_t, CountSketch*> sketchs;
public:
	RingSketch(double epsilon, double gamma, int ring_size);

	void increment(int key);
	int query(int key);

	void expand();
	void shrink();
	int numSketchs();
private:
	uint32_t getSketchIndexForKey(uint32_t k);

	uint32_t getFullestSketchIdx();
	uint32_t getEmptiestSketchIdx();
};