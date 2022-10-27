#include "NaiveSketch.h"

NaiveSketch::NaiveSketch(int width, int depth, int seed)
{
	Node* node = new Node(width, depth, seed);
	nodes.push_back(node);
}

void NaiveSketch::update(uint32_t item, int diff)
{
	Node* min_node = nodes[emptiestNodeIndex()];
	min_node->num_events++;
	CM_Update(min_node->sketch, item, diff);
}

int NaiveSketch::query(uint32_t item)
{
	int res = 0;
	for (auto node : nodes) {
		res += CM_PointEst(node->sketch, item);
	}
	return res;
}

void NaiveSketch::expand()
{
	Node* n = nodes[0];
	Node* k = new Node(n->sketch->width, n->sketch->depth, seed);
	nodes.push_back(k);
}

void NaiveSketch::shrink()
{
	int index_remove = emptiestNodeIndex();
	Node* node = nodes[index_remove];
	nodes.erase(nodes.begin() + index_remove);
	int index_merge = emptiestNodeIndex();
	nodes[index_merge]->num_events += node->num_events;
	CM_Merge(node->sketch, nodes[index_merge]->sketch);
	delete node;
}

int NaiveSketch::sketchCount() const
{
	return nodes.size();
}

int NaiveSketch::emptiestNodeIndex()
{
	int min_node_index = 0;
	for (int i = 1; i < nodes.size(); i++) {
		if (nodes[i]->num_events < nodes[min_node_index]->num_events) min_node_index = i;
	}
	return min_node_index;
}

NaiveSketch::Node::Node(int width, int depth, int seed) :
	sketch(nullptr),
	num_events(0) {
	sketch = CM_Init(width, depth, seed);
}
