#pragma once

#include <random>
#include <map>
#include <vector>

#include "CountSketch.h"
#include "Dictionary.h"

class DynamicSketch : public Dictionary
{
	struct Node {
        CountSketch sketch;

		std::vector<int> buckets;

		int num_events;

		uint32_t min_key;
		uint32_t max_key;

		Node(int width, int depth, int seed, uint32_t min_key, uint32_t max_key);
		void clearBuckets();
		void updateMedianSinceLastClear(uint32_t key, int amount); 
		std::pair<uint32_t, uint32_t> getRangeWithHalfOfUpdates() const;
		int updatesSinceLastClear() const;

		static bool compareMinKey(Node* n0, Node* n1);
	};
	std::vector<Node*> nodes_vector;
	int width;
	int depth;
	int seed;
public:

	DynamicSketch(int width, int depth, int seed);

	void update(uint32_t key, int amount);
	int query(uint32_t item);

	void expand();
	void shrink();

    int getSize() const;
    int getMemoryUsage() const; // minimum

    void printInfo(int index) const;
private:
	static bool nodeComp(Node* n0, Node* n1);
	int firstAt(uint32_t item);
	int nextAt(int sketch_index, uint32_t item);
	void clearAllBuckets();
};
