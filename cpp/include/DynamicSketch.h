#pragma once

#include "countmin.h"
#include "Dictionary.h"

class DynamicSketch : public Dictionary
{
	std::vector<CM_type*> sketches;
public:
    DynamicSketch(int width, int depth);

	void update(uint32_t key, int amount);
	int query(uint32_t item);

	void expand(int width);
	void shrink(int n);
	void compress(int n);

    int getSize() const;
    int getMemoryUsage() const; // minimum
private:
	void mergeCountMin(CM_type* cm0, CM_type* cm1);
};
