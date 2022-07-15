#include "RedirGraph.h"

#include <random>
#include <cassert>
#include <iostream>

RedirGraph::RedirGraph(uint32_t size_initial)
{
	for (uint64_t i = 0; i < size_initial; i++) {
		nodes.push_back(RedirNode());
	}
}

uint32_t RedirGraph::allocNode() {
	uint32_t idx;
	if (unused_node_indice.size() > 0) {
		idx = unused_node_indice.back();
		unused_node_indice.pop_back();
	}
	else {
		idx = size();
		nodes.push_back(RedirNode());
	}
	return idx;
}

uint32_t RedirGraph::size() const
{
	return nodes.size();
}

// create an edge from nodes[i_src] to nodes[i_dst]
void RedirGraph::redirect(uint32_t i_src, uint32_t i_dst)
{
	assert(i_dst < size() && i_src < size() && isRoot(i_src));

	const auto& indice_with_edge_to_i_src = reverse_edges[i_src];
	for (const auto& i : indice_with_edge_to_i_src) {
		if (nodes[i].child0 == i_src) {
			reverse_edges[i_dst].insert(i);
			nodes[i].child0 = i_dst;
		}
		if (nodes[i].child1 == i_src) {
			reverse_edges[i_dst].insert(i);
			nodes[i].child1 = i_dst;
		}
	}
	reverse_edges[i_src].clear();
	unused_node_indice.push_back(i_src);
}

// create two new nodes and edges from nodes[i] to to them, return their indice in i_splitX
void RedirGraph::split(uint32_t i, uint32_t& i_split0, uint32_t& i_split1)
{
	assert(i < size() && isRoot(i));
	nodes[i].child0 = allocNode();
	i_split0 = nodes[i].child0;
	nodes[i].child1 = allocNode();
	i_split1 = nodes[i].child1;

	reverse_edges[i_split0] = std::set<uint32_t>({i});
	reverse_edges[i_split1] = std::set<uint32_t>({i});
}

bool RedirGraph::isRoot(uint32_t i) const
{
	assert(i < size());
	return nodes[i].child0 == UINT32_MAX && nodes[i].child1 == UINT32_MAX;
}

bool RedirGraph::isSplit(uint32_t i) const
{
	assert(i < size());

	return nodes[i].child0 != UINT32_MAX && nodes[i].child1 != UINT32_MAX;
}

uint32_t RedirGraph::getChild(uint32_t i, bool child1)
{
	assert(isSplit(i));

	return child1 ? nodes[i].child1 : nodes[i].child0;
}

void RedirGraph::print()
{
	std::vector<int> V;
	std::vector<std::pair<int, int>> E;
	for (int i = 0; i < nodes.size(); i++) {
		V.push_back(i);
		if (isSplit(i)) {
			E.push_back(std::make_pair(i, nodes[i].child0));
			E.push_back(std::make_pair(i, nodes[i].child1));
		}
		else if (!isRoot(i)) {
			E.push_back(std::make_pair(i, nodes[i].child0));
		}
	}
	for (const auto& v : V) std::cout << "\'" << v << "\', ";
	for (const auto& e : E) std::cout << "(" << e.first << ", " << e.second << "), " << std::endl;
}

// This function uses rand() to generate a 64 bit key
uint64_t RedirGraph::randKey()
{
	uint64_t key = 0;
	int shift = std::log2(RAND_MAX);
	for (int i = 0; i < 64; i += shift) {
		// shift prev rand() up the key
		key = key << shift;
		// write rand() to it's lower bits
		key += rand();
	}
	return key;
}

// None edge marked with UINT32_MAX
RedirGraph::RedirNode::RedirNode() :
	child0(UINT32_MAX),
	child1(UINT32_MAX)
{
}
