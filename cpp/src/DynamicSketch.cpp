#include "DynamicSketch.h"

#include <algorithm>

int bucket_count = 5;

DynamicSketch::DynamicSketch(int width, int depth, int _seed) : Dictionary(), seed(_seed)
{
	Node *node = new Node(width, depth, seed, 0, UINT32_MAX);
	nodes_vector.push_back(node);
}

void DynamicSketch::update(uint32_t item, int diff)
{
	int num_events = INT_MAX;
	Node *node = nullptr;

	for (int i = firstAt(item); i >= 0; i = nextAt(i, item))
	{
		Node *node_curr = nodes_vector[i];
		int num_events_curr = node_curr->num_events;
		if (num_events_curr < num_events)
		{
			num_events = num_events_curr;
			node = node_curr;
		}
	}

	assert(node && "DynamicSketch::update - node in range wasn't found");
	node->updateMedianSinceLastClear(item, diff);
	CM_Update(node->sketch, item, diff);
}

int DynamicSketch::query(uint32_t item)
{
	int sum = 0;
	for (int i = firstAt(item); i >= 0; i = nextAt(i, item))
	{
		auto &sketch = nodes_vector[i]->sketch;
		sum += CM_PointEst(sketch, item);
	}
	return sum;
}

void DynamicSketch::expand()
{
	Node *node_max;
	node_max = nodes_vector[0];
	std::pair<uint32_t, uint32_t> range_max;
	range_max.first = nodes_vector[0]->min_key;
	range_max.second = nodes_vector[0]->max_key;

	int index_max;
	for (int i = 1; i < nodes_vector.size(); i++)
	{
		Node *node = nodes_vector[i];
		if (node->updatesSinceLastClear() > node_max->updatesSinceLastClear())
		{
			node_max = node;
			range_max.first = node->min_key;
			range_max.second = node->max_key;
			index_max = i;
		}
	}
	auto range_child = node_max->getRangeWithHalfOfUpdates();
	Node *node_child = new DynamicSketch::Node(node_max->sketch->width, node_max->sketch->depth, seed, range_child.first, range_child.second);

	// sorted insertion into nodes_vector

	auto it = std::lower_bound(nodes_vector.begin(), nodes_vector.end(), node_child, Node::compareMinKey);
	nodes_vector.insert(it, node_child);

	clearAllBuckets();
}

void DynamicSketch::shrink()
{
	if (getSize() == 1)
		return;

	auto contains = [](std::pair<uint32_t, uint32_t> r0, std::pair<uint32_t, uint32_t> r1) -> bool
	{
		return r0.first <= r1.first && r0.second >= r1.second;
	};

	int node_child_index = -1;
	Node *node_child_min = nullptr, *node_parent_min = nullptr;
	uint32_t min_num_events = UINT32_MAX;
	for (int i = 1; i < nodes_vector.size(); i++)
	{
		Node *n0 = nodes_vector[i];
		auto n0_range = std::make_pair(n0->min_key, n0->max_key);
		for (int j = 0; j < i; j++)
		{
			Node *n1 = nodes_vector[j];
			auto n1_range = std::make_pair(n1->min_key, n1->max_key);
			int cur_num_events = n0->num_events + n1->num_events;
			if (cur_num_events < min_num_events)
			{
				if (contains(n0_range, n1_range))
				{
					min_num_events = cur_num_events;
					node_parent_min = n0;
					node_child_min = n1;
					node_child_index = j;
				}
				else if (contains(n1_range, n0_range))
				{
					min_num_events = cur_num_events;
					node_parent_min = n1;
					node_child_min = n0;
					node_child_index = i;
				}
			}
		}
	}

	if (node_child_min && node_parent_min)
	{
		nodes_vector.erase(nodes_vector.begin() + node_child_index);
		CM_Merge(node_parent_min->sketch, node_child_min->sketch);
		node_parent_min->num_events += node_child_min->num_events;
		delete node_child_min;
	}
	clearAllBuckets();
}

int DynamicSketch::getSize() const
{
	return nodes_vector.size();
}

int DynamicSketch::getMemoryUsage() const
{
	auto sketch_size = CM_Size(nodes_vector[0]->sketch);
	return sizeof(DynamicSketch) + getSize() * (sizeof(DynamicSketch::Node) + sketch_size);
}

bool DynamicSketch::nodeComp(Node *n0, Node *n1)
{
	if (n0->min_key < n1->min_key)
		return true;
	else if (n0->min_key == n1->min_key)
		return n0->max_key > n1->max_key;
	return false;
}

int DynamicSketch::firstAt(uint32_t key)
{
	return nextAt(-1, key);
}

int DynamicSketch::nextAt(int sketch_index, uint32_t key)
{
	while (++sketch_index < nodes_vector.size())
	{
		auto node = nodes_vector[sketch_index];
		if (node->min_key <= key)
		{
			if (node->max_key >= key) {
				return sketch_index;
			}
		}
		else {
			return -1;
		}
	}
	return -1;
}

DynamicSketch::Node::Node(int width, int depth, int seed, uint32_t _min_key, uint32_t _max_key) : sketch(nullptr),
																								  num_events(0),
																								  min_key(_min_key),
																								  max_key(_max_key),
																								  buckets(bucket_count, 0)
{
	sketch = CM_Init(width, depth, seed);
}

bool DynamicSketch::Node::compareMinKey(Node* n0, Node* n1)
{
	if (n0->min_key == n1->min_key) {
		return n0->max_key < n1->max_key;
	}
	else {
		return n0->min_key < n1->min_key;
	}
}

void DynamicSketch::Node::clearBuckets()
{
	buckets = std::vector<int>(bucket_count, 0);
}

void DynamicSketch::Node::updateMedianSinceLastClear(uint32_t key, int amount)
{
	num_events+=amount;
	uint32_t bucket_width = (max_key - min_key) / bucket_count;
	uint32_t bucket_index = std::min(((key - min_key)/bucket_width), (uint32_t)bucket_count - 1U);
	buckets[(int)bucket_index] += amount;
}

int DynamicSketch::Node::updatesSinceLastClear() const
{
	int n = 0;
	for (int i = 0; i < bucket_count; i++)
	{
		n += buckets[i];
	}
	return n;
}

template <typename T> 
T length(std::pair<T, T> pair) {
	return pair.second - pair.first;
}

// smallest subrange containing closest to half the number of total events, while being > 0
std::pair<uint32_t, uint32_t> DynamicSketch::Node::getRangeWithHalfOfUpdates() const
{
	int min_range_size = bucket_count;

	int updates_total = updatesSinceLastClear();
	if (updates_total == 0)
	{
		return std::make_pair(min_key, (min_key + max_key) / 2);
	}
	uint32_t bucket_width = (max_key - min_key) / bucket_count;
	auto range_closest = std::make_pair(min_key, (min_key + max_key) / 2);
	int double_delta_closest = INT_MAX;
	int double_delta_closest_abs = INT_MAX;

	for (int i = 0; i < bucket_count; i++) {
		int range_updates_current = 0;
		for (int j = i+1; j < bucket_count+1; j++)
		{
			auto range_current = make_pair(min_key + i * bucket_width, min_key + j * bucket_width);
			if (length(range_current) < min_range_size) {
				continue;
			}
			range_updates_current += buckets[j-1];
			int double_delta_current = 2 * range_updates_current - updates_total;
			int double_delta_current_abs = std::abs(double_delta_current);
			if (
				double_delta_current_abs < double_delta_closest_abs ||
				double_delta_current_abs == double_delta_closest_abs && (length(range_current) < length(range_closest) || double_delta_current > double_delta_closest)
			) {
				double_delta_closest = double_delta_current;
				double_delta_closest_abs = std::abs(double_delta_closest);
				range_closest = range_current;
			}
		}
	}
	return range_closest;
}

void DynamicSketch::clearAllBuckets(){
	for (const auto& node : nodes_vector){
		node->clearBuckets();
	}
}

void DynamicSketch::printInfo() const {
	std::cout << "{";
	for (int j = 0; j < nodes_vector.size(); j++) {
		auto node = nodes_vector[j];
		if (j > 0) std::cout << ",";
		std::cout << "{";
		std::cout << "\"min_key\":" << node->min_key << ",\"max_key\":" << node->max_key << ",\"updates_since_last_clear\":" << node->updatesSinceLastClear();
		std::cout << ",\"num_events\":" << node->num_events << ",\"buckets\":[" << std::endl;
		uint32_t bucket_width = (node->max_key - node->min_key) / bucket_count;
		for (int i = 0; i < node->buckets.size(); i++) {
			uint32_t min_key = node->min_key + i * bucket_width;
			uint32_t max_key = node->min_key + (i + 1) * bucket_width;
			std::cout << "	{\"min_key\":" << min_key << ",\"max_key\":" << max_key << ",\"counter\":" << node->buckets[i] << "}" << std::endl;
		}
		std::cout << "]}";
	}
	std::cout << "}" << std::endl;
}
