#pragma once

#include <random>
#include <map>
#include <vector>

#include "countmin.h"


class NaiveSketch
{
	struct Node {
		CM_type* sketch;
		int num_events;

		Node(int width, int depth, int seed);
	};

	std::vector<Node*> nodes;
	int seed;
public:
	NaiveSketch(int width, int depth, int seed);

	void update(uint32_t item, int diff);
	int query(uint32_t item);

	void expand();
	void shrink();

	int sketchCount() const;
private:
	int emptiestNodeIndex();
};
