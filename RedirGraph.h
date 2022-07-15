#pragma once

#include <cstdint>
#include <vector>
#include <map>
#include <set>

// 128 bit per node, node per split
class RedirGraph
{
	struct RedirNode {
		uint32_t child0, child1;

		RedirNode();
	};
	std::vector<uint32_t> unused_node_indice;
	std::vector<RedirNode> nodes;
	std::map<uint32_t, std::set<uint32_t>> reverse_edges;
public:
	RedirGraph(uint32_t size_initial);

	uint32_t size() const;
	uint32_t allocNode(); // allocate a new node, return it's index

	void redirect(uint32_t i_src, uint32_t i_dst);	
	void split(uint32_t i, uint32_t& i_split0, uint32_t& i_split1);

	bool isRoot(uint32_t i) const;
	bool isSplit(uint32_t i) const;
	uint32_t getChild(uint32_t i, bool child1);

	void print();
private:
	static uint64_t randKey();
};