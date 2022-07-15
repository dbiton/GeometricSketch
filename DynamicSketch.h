#pragma once

#include <random>
#include <map>
#include <vector>

#include "countmin.h"


class DynamicSketch
{
	struct Node {
		CM_type* sketch;

		Node* parent;
		uint32_t updates_average;
		uint32_t updates_counter;
		int num_events;

		bool flip_flop;
		uint32_t min_key;
		uint32_t max_key;

		Node(int width, int depth, int seed, uint32_t min_key, uint32_t max_key, Node* parent);
		static bool compareMinKey(Node* n0, Node* n1);
	};
	std::vector<Node*> nodes_vector;

	int seed;
public:
	DynamicSketch(int width, int depth, int seed);

	void update(uint32_t item, int diff);
	int query(uint32_t item);

	void expand();
	void shrink();

	int sketchCount() const;
	int byteSize() const;
private:
	static bool nodeComp(Node* n0, Node* n1);
	int firstAt(int value);
	int nextAt(int index, int value);
};
