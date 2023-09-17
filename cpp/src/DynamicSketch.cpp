#include "DynamicSketch.h"
#include <algorithm>

DynamicSketch::DynamicSketch(int width, int depth, int _seed) : Dictionary(), seed(_seed)
{
	Node *node = new Node(width, depth, seed, 0, UINT32_MAX);
	nodes_vector.push_back(node);
}

void DynamicSketch::update(uint32_t item, int diff)
{
	uint32_t num_events = UINT32_MAX;
	Node *node = nullptr;

	for (int i = firstAt(item); i >= 0; i = nextAt(i, item))
	{
		Node *node_curr = nodes_vector[i];
		uint32_t num_events_curr = node_curr->num_events;
		if (num_events_curr <= num_events)
		{
			num_events = num_events_curr;
			node = node_curr;
		}
	}

	assert(node);
	node->updates_counter++;
	node->num_events++;
	if (node->updates_counter == 1)
	{
		node->updates_average = item;
	}
	else
	{
		node->updates_average += item / node->updates_counter - node->updates_average / node->updates_counter;
	}
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
		if (node->updates_counter > node_max->updates_counter)
		{
			node_max = node;
			range_max.first = node->min_key;
			range_max.second = node->max_key;
			index_max = i;
		}
	}
	auto range_child = node_max->flip_flop ? std::make_pair(node_max->updates_average, range_max.second) : std::make_pair(range_max.first, node_max->updates_average);
	Node *node_child = new DynamicSketch::Node(node_max->sketch->width, node_max->sketch->depth, seed, range_child.first, range_child.second);

	// sorted insertion into nodes_vector

	auto it = std::lower_bound(nodes_vector.begin(), nodes_vector.end(), node_child, Node::compareMinKey);
	nodes_vector.insert(it, node_child);

	node_max->flip_flop = !node_max->flip_flop;
	node_max->updates_counter = 0;
	node_max->updates_average = range_max.first / 2 + range_max.second / 2;
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

int DynamicSketch::firstAt(int value)
{
	return nextAt(-1, value);
}

int DynamicSketch::nextAt(int index, int value)
{
	while (++index < nodes_vector.size())
	{
		auto node = nodes_vector[index];
		if (node->min_key <= value && node->max_key >= value)
		{
			return index;
		}
		else
			return -1;
	}
	return -1;
}

DynamicSketch::Node::Node(int width, int depth, int seed, uint32_t _min_key, uint32_t _max_key) : sketch(nullptr),
																								  num_events(0),
																								  updates_counter(0),
																								  updates_average(0),
																								  flip_flop(rand()),
																								  min_key(_min_key),
																								  max_key(_max_key)
{
	sketch = CM_Init(width, depth, seed);
}

bool DynamicSketch::Node::compareMinKey(Node *n0, Node *n1)
{
	return n0->min_key < n1->min_key;
}
