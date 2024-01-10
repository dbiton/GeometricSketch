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
	void shrink(int bytes);

    int getSize() const;
    int getMemoryUsage() const; // minimum
};
