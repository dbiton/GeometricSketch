#pragma once

#include <random>
#include <map>
#include <vector>

#include "countmin.h"
#include "Dictionary.h"

class DynamicSketch : public Dictionary
{
	struct Node {
		CM_type* sketch;

		std::vector<int> buckets;

		int num_events;

		uint32_t min_key;
		uint32_t max_key;

		Node(int width, int depth, int seed, uint32_t min_key, uint32_t max_key);
		void clearBuckets();
		void updateMedianSinceLastClear(uint32_t key, int amount); 
		uint32_t estimateMedianSinceLastClear() const;
		int updatesSinceLastClear() const;

		static bool compareMinKey(Node* n0, Node* n1);
	};
	std::vector<Node*> nodes_vector;

	int seed;
public:

	DynamicSketch(int width, int depth, int seed);

	void update(uint32_t key, int amount);
	int query(uint32_t item);

	void expand();
	void shrink();

    int getSize() const;
    int getMemoryUsage() const; // minimum
private:
	static bool nodeComp(Node* n0, Node* n1);
	int firstAt(int value);
	int nextAt(int index, int value);
	void clearAllBuckets();
};
