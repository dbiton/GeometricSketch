#pragma once


#include <map>
#include <vector>

#include "countmin.h"


typedef int (*sketchFilter)(int);

class CountMinSketch {
	int width, depth, seed;
	std::vector<std::pair<uint32_t, uint32_t>> heavy_hitters;
	CM_type* sketch;
	unsigned num_events;
	float err_amount;
public:
	CountMinSketch(int width, int depth, int seed, int num_hh);
	CountMinSketch(const CountMinSketch& o);
	CountMinSketch* newEmptyLike();


	~CountMinSketch();

	void update(unsigned int item, int diff);
	int pointEst(unsigned int query);

	int getWidth() const;
	int getDepth() const;
	int getNumHH() const;

	unsigned numEvents() const;
	
	void merge(const CountMinSketch& sketch);	
	void clear();
	CountMinSketch* split(sketchFilter filter);
private:
	void updateHeavyHitters(uint32_t e, uint32_t count);
	bool isHeavyHitter(unsigned int query, int& counts);
};