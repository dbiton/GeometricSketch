#include "DynamicSketch.h"
#include "extract_subrange.h"

#include <algorithm>


int bucket_count = 8;

DynamicSketch::DynamicSketch(int width, int depth, int _seed, int bucket_num) : 
	Dictionary(), 
	seed(_seed), 
	width(width), 
	depth(depth),
	heavy_part(bucket_num)
{
	Node *node = new Node(width, depth, seed, 0, UINT32_MAX);
	nodes_vector.push_back(node);
}

void DynamicSketch::update(uint32_t item, int f)
{
	uint8_t *key = (uint8_t*)&item;
    uint8_t swap_key[KEY_LENGTH_4];
    uint32_t swap_val = 0;
    int result = heavy_part.insert(key, swap_key, swap_val, f);
    switch(result)
    {
        case 0: return;
        case 1:{
			// highest bit of swap_val used to indicate if we should swap
            if(HIGHEST_BIT_IS_1(swap_val))
                saveInSketches((uint32_t)*swap_key, GetCounterVal(swap_val));
            else{
				int amount_swap_key = querySketches((uint32_t)*swap_key);
				int amount_increase = amount_swap_key - swap_val;
				// replace here if we make sketch hold something else (chars etc)
				int max_sketches_value = INT_MAX;
				if (amount_increase > 0 && amount_swap_key < max_sketches_value){
					saveInSketches((uint32_t)*swap_key, amount_increase);
				}
			}
			return;
        }
        case 2: saveInSketches(item, 1);  return;
        default:
            printf("error return value !\n");
            exit(1);
    }
}

int DynamicSketch::query(uint32_t item)
{
	uint8_t *key = (uint8_t*)&item;
    uint32_t heavy_result = heavy_part.query(key);
    if(heavy_result == 0 || HIGHEST_BIT_IS_1(heavy_result))
    {
        int light_result = querySketches(item);
        return (int)GetCounterVal(heavy_result) + light_result;
    }
    return heavy_result;
}

void DynamicSketch::saveInSketches(uint32_t item, int diff)
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
	uint8_t* item_key = (uint8_t*)&item;
	node->sketch.insert(item_key, diff);
}

int DynamicSketch::querySketches(uint32_t item)
{
	uint8_t* item_key = (uint8_t*)&item;
	int sum = 0;
	for (int i = firstAt(item); i >= 0; i = nextAt(i, item))
	{
		auto &sketch = nodes_vector[i]->sketch;
		sum += sketch.query(item_key);
	}
	return sum;
}

void DynamicSketch::expand()
{
    Node *node_max = nodes_vector[0];
    int max_amount = node_max->num_events;

    //std::cout << "expand" << std::endl;
    for (int i = 1; i < nodes_vector.size(); i++)
	{
        Node *node = nodes_vector[i];
        int amount = node->num_events;
        //std::cout << node->min_key << "," << node->max_key << std::endl;
        //std::cout << "best:" << max_amount << " current:" << amount << std::endl;
        if (amount > max_amount)
		{
            max_amount = amount;
			node_max = node;
		}
	}
	auto range_child = node_max->getRangeWithHalfOfUpdates();
	Node *node_child = new DynamicSketch::Node(width, depth, seed, range_child.first, range_child.second);

	// sorted insertion into nodes_vector

	auto it = std::lower_bound(nodes_vector.begin(), nodes_vector.end(), node_child, Node::compareMinKey);
	nodes_vector.insert(it, node_child);
    node_max->clearBuckets();
    //clearAllBuckets();
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
		node_parent_min->sketch.merge(node_child_min->sketch);
		node_parent_min->num_events += node_child_min->num_events;
		delete node_child_min;
	}
    //clearAllBuckets();
}

int DynamicSketch::getSize() const
{
	return nodes_vector.size();
}

int DynamicSketch::getMemoryUsage() const
{
	auto sketch_size = 0; // CM_Size(nodes_vector[0]->sketch);
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
            if (node->max_key >= key)
            {
                return sketch_index;
            }
		}
        else
        {
            return -1;
        }
	}
	return -1;
}

DynamicSketch::Node::Node(int width, int depth, int seed, uint32_t _min_key, uint32_t _max_key) : sketch(LightPart(width*depth, seed)),
																								  num_events(0),
																								  min_key(_min_key),
																								  max_key(_max_key),
																								  buckets(bucket_count, 0)
{
}

bool DynamicSketch::Node::compareMinKey(Node *n0, Node *n1)
{
	if (n0->min_key == n1->min_key)
	{
		return n0->max_key < n1->max_key;
	}
	else
	{
		return n0->min_key < n1->min_key;
	}
}

void DynamicSketch::Node::clearBuckets()
{
	buckets = std::vector<int>(bucket_count, 0);
}

void DynamicSketch::Node::updateMedianSinceLastClear(uint32_t key, int amount)
{
	num_events += amount;
    double bucket_width = (double)(max_key - min_key) / (double)bucket_count;
    int bucket_index = (int)std::floor((double)(key - min_key) / (double)bucket_width);
    buckets[bucket_index] += amount;
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
T length(std::pair<T, T> pair)
{
	return pair.second - pair.first;
}

// smallest subrange containing closest to half the number of total events, while being > 0
std::pair<uint32_t, uint32_t> DynamicSketch::Node::getRangeWithHalfOfUpdates() const
{
    int min_range = bucket_count * bucket_count;
    if (max_key - min_key < min_range){
        return std::make_pair(min_key, max_key);
    }
    auto bucket_range = extract_subrange(buckets);
    double bucket_width = (double)(max_key - min_key) / (double)bucket_count;
    uint32_t range_begin = bucket_range.first*bucket_width+min_key;
    uint32_t range_end = bucket_range.second*bucket_width+min_key;
    return std::make_pair(range_begin, range_end);
}

void DynamicSketch::clearAllBuckets()
{
	for (const auto &node : nodes_vector)
	{
		node->clearBuckets();
	}
}

void DynamicSketch::printInfo(int packet_index) const
{
	std::cout << "{\"index\":" << packet_index << ",\"loads\":[";
	for (int j = 0; j < nodes_vector.size(); j++)
	{
		auto node = nodes_vector[j];
		std::cout << "{";
		std::cout << "\"min_key\":" << node->min_key << ",\"max_key\":" << node->max_key << ",\"updates_since_last_clear\":" << node->updatesSinceLastClear();
		std::cout << ",\"num_events\":" << node->num_events << ",\"buckets\":[" << std::endl;
		uint32_t bucket_width = (node->max_key - node->min_key) / bucket_count;
		for (int i = 0; i < node->buckets.size(); i++)
		{
			uint32_t min_key = node->min_key + i * bucket_width;
			uint32_t max_key = node->min_key + (i + 1) * bucket_width;
			std::cout << "	{\"min_key\":" << min_key << ",\"max_key\":" << max_key << ",\"counter\":" << node->buckets[i] << "}" << std::endl;
			if (i < node->buckets.size() - 1)
			{
				std::cout << ",";
			}
		}
		std::cout << "]}";
		if (j < nodes_vector.size() - 1)
		{
			std::cout << ",";
		}
	}
	std::cout << "]}," << std::endl;
}
