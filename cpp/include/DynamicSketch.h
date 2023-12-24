#pragma once

#include <random>
#include <map>
#include <vector>

#include "HeavyPart.h"
#include "LightPart.h"
#include "Dictionary.h"

class DynamicSketch : public Dictionary
{
	struct Node {
        LightPart sketch;
		std::vector<bool> bitmask;
		int num_events;

		Node(int width, int depth, int seed, const std::vector<bool>& bitmask);

		bool keyInRange(uint32_t key);
		void update(uint32_t key, int amount);
	};
	HeavyPart heavy_part;
	std::vector<Node> nodes_vector;
	std::vector<uint32_t> distribution;
	int width;
	int depth;
	int seed;
public:
    DynamicSketch(int width, int depth, int seed, int heavy_part_bucket_num, int dist_buckets_count);

	void update(uint32_t key, int amount);
	int query(uint32_t item);

	void expand();
	void shrink();

    int getSize() const;
    int getMemoryUsage() const; // minimum

    void printInfo(int index) const;
private:
	void getBitmaskForNewSketch(std::vector<bool>& bitmask);
	int distributionBucketIndex(uint32_t key);
	int querySketches(uint32_t key);
	void saveInSketches(uint32_t key, int amount);
	static bool nodeComp(Node* n0, Node* n1);
	int firstAt(uint32_t item);
	int nextAt(int sketch_index, uint32_t item);
	void clearAllBuckets();
};
