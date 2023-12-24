#include "DynamicSketch.h"
#include "extract_subrange.h"

#include <algorithm>


int bucket_count = 8;

int bucketIndexForKey(uint32_t key, int num_buckets)
{
	return (int)(key / (uint32_t)num_buckets);
}

DynamicSketch::DynamicSketch(int width, int depth, int seed, int heavy_part_bucket_num, int dist_buckets_count) : 
	Dictionary(), 
	seed(seed), 
	width(width), 
	depth(depth),
	heavy_part(heavy_part_bucket_num),
	distribution(dist_buckets_count, 0)
{
	auto node_mask = std::vector<bool>(true, dist_buckets_count);
	nodes_vector.push_back(Node(width, depth, seed, node_mask));
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
				// also should consider possible OVERFLOW, save in multiple sketches?
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
	Node* node = nullptr;

	for (int i = firstAt(item); i >= 0; i = nextAt(i, item))
	{
		Node* node_curr = &nodes_vector[i];
		int num_events_curr = node_curr->num_events;
		if (num_events_curr < num_events)
		{
			num_events = num_events_curr;
			node = node_curr;
		}
	}
	assert(node && "DynamicSketch::update - node in range wasn't found");
	int bucket_index = bucketIndexForKey(item, distribution.size());
	distribution[bucket_index] += diff;
	node->update(item, diff);
}

int DynamicSketch::querySketches(uint32_t item)
{
	uint8_t* item_key = (uint8_t*)&item;
	int sum = 0;
	for (int i = firstAt(item); i >= 0; i = nextAt(i, item))
	{
		auto &sketch = nodes_vector[i].sketch;
		sum += sketch.query(item_key);
	}
	return sum;
}

void DynamicSketch::getBitmaskForNewSketch(std::vector<bool>& bitmask)
{
	int n = distribution.size();
	std::vector<double> loads;
	std::vector<double> coverage;
	for (const auto& node : nodes_vector){
		double node_range = 0;
		for (int i=0; i<n; i++) node_range += node.bitmask[i];
		for (int i=0; i<n; i++){
			coverage[i] += (double) node.bitmask[i] / node_range;
		}
	}
	double total_num_events = 0;
	for (const auto& v : distribution){
		total_num_events += v;
	}
	for (int i=0; i<n; i++){
		loads[i] = (double) distribution[i] / total_num_events;
	}

	bitmask = std::vector<bool>(n, false);

	// replace this with a linear equation solver or something
	for (int num_bits_on = 1; num_bits_on <= n; num_bits_on++){
		
	}

}

void DynamicSketch::expand()
{
    Node *node_max = &nodes_vector[0];
    int max_amount = node_max->num_events;

    for (int i = 1; i < nodes_vector.size(); i++)
	{
        Node *node = &nodes_vector[i];
        int amount = node->num_events;
        if (amount > max_amount)
		{
            max_amount = amount;
			node_max = node;
		}
	}
	std::vector<bool> bitmask;
	getBitmaskForNewSketch(bitmask);
	nodes_vector.push_back(DynamicSketch::Node(width, depth, seed, bitmask));
}

void DynamicSketch::shrink()
{
	throw std::runtime_error("DynamicSketch::shrink implement!");
}

int DynamicSketch::getSize() const
{
	return nodes_vector.size();
}

int DynamicSketch::getMemoryUsage() const
{
	auto sketch_size = 0; // FIXME CM_Size(nodes_vector[0]->sketch);
	return sizeof(DynamicSketch) + getSize() * (sizeof(DynamicSketch::Node) + sketch_size);
}

int DynamicSketch::firstAt(uint32_t key)
{
	return nextAt(-1, key);
}

int DynamicSketch::nextAt(int sketch_index, uint32_t key)
{
    int key_bitmask_index = bucketIndexForKey(key, distribution.size());
	while (++sketch_index < nodes_vector.size())
	{
		auto node = &nodes_vector[sketch_index];
        if (node->bitmask[key_bitmask_index])
		{
			return sketch_index;
		}
        else
        {
            return -1;
        }
	}
	return -1;
}

DynamicSketch::Node::Node(int width, int depth, int seed, const std::vector<bool>& bitmask) : 
																sketch(LightPart(width*depth, seed)),
																num_events(0),
																bitmask(bitmask)
{
}

void DynamicSketch::Node::update(uint32_t item, int amount)
{
	uint8_t* item_key = (uint8_t*)&item;
	sketch.insert(item_key, amount);
	num_events += amount;
}

void DynamicSketch::printInfo(int packet_index) const
{
	std::cout << "{\"index\":" << packet_index << ",\"distribution\":[";
	uint32_t dist_bucket_width = UINT32_MAX / distribution.size();
	for (int i = 0; i < distribution.size(); i++)
	{
		std::cout << "{\"min_key\":" << dist_bucket_width * i << ",\"max_key\":" << dist_bucket_width * (i+1) << ",\"load\":" << distribution[i] << "}";
	}
	std::cout << "]}," << std::endl;
}
