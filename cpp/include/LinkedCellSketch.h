#pragma once

#include "Dictionary.h"

class LinkedCellSketch : public Dictionary
{
    std::vector<uint32_t> counters;
    unsigned offset;
    unsigned branching_factor;
    unsigned width;
    unsigned depth;
public:
    LinkedCellSketch(int width, int depth, int branching_factor);
    ~LinkedCellSketch();

    void update(uint32_t key, int amount);
    int query(uint32_t key);

    int shrink(int n);
    int expand(int n);
    int undoExpand(int n);
    int compress(int n);
    void print();
    int getMemoryUsage() const;
private:
    uint64_t hash(uint32_t key, uint32_t row_index, uint32_t layer_index) const;
    
    int getLastLayerVectorOffsetFromKey(
        uint32_t key,
        uint16_t row_index
    );

    int getNextLayerVectorOffsetFromKey(
        uint32_t key,
        uint16_t row_index,
        int& layer_index,
        int& layer_begin_counter_index,
        int& last_counter_index,
        int& B_pow_layer_index
    ) const;

    int getVectorOffsetParent(int counter_index) const;

    int rowOffsetToLayerIndex(int row_offset, int& layer_offset) const;
    int vectorOffsetToRowIndex(int vector_offset, int& row_offset) const;
    int rowOffsetToVectorOffset(int row_index, int row_offset) const;

    int getLayerRowOffset(int layer_index) const;
    int getVectorOffsetFirstChild(int vector_offset) const;
};
