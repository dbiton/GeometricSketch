#pragma once

#include "Dictionary.h"
#include "MultiHash.h"

class LinkedCellSketch : public Dictionary
{
    std::vector<uint32_t> counters;
    unsigned offset;
    unsigned branching_factor;
    unsigned width;
    unsigned depth;

    int consumed_bits;
    uint64_t seed;
    uint64_t hash;
    uint64_t key;

    uint64_t subhash_mask;
    int bits_per_subhash;
    uint64_t subhash_modulus;

    uint64_t first_subhash_mask;
    int bits_per_first_subhash;
    uint64_t first_subhash_modulus;
public:
    LinkedCellSketch(int width, int depth, int branching_factor);
    ~LinkedCellSketch();

    void update(uint32_t key, int amount);
    int query(uint32_t key);

    int shrink(int n);
    int expand(int n);
    int undoExpand(int n);
    int compress(int n);

    int getMemoryUsage() const;
private:
    //int hash(uint32_t key, uint16_t row_index, uint16_t layer_index) const;
    
    int getLastLayerVectorOffsetFromKey(
        uint64_t key,
        uint64_t row_index
    );

    int getNextLayerVectorOffsetFromKey(
        uint64_t key,
        uint64_t row_index,
        int& layer_begin_counter_index,
        int& last_counter_index,
        int& B_pow_layer_index
    );

    int getVectorOffsetParent(int counter_index) const;

    int rowOffsetToLayerIndex(int row_offset, int& layer_offset) const;
    int vectorOffsetToRowIndex(int vector_offset, int& row_offset) const;
    int rowOffsetToVectorOffset(int row_index, int row_offset) const;

    int getLayerRowOffset(int layer_index) const;
    int getVectorOffsetFirstChild(int vector_offset) const;

    void initialize(uint64_t _key, uint64_t _seed);
    void setFirstSubHashModulus(uint64_t _first_subhash_modulus);

    void setSubHashModulus(uint64_t _subhash_modulus);
    uint64_t next();
    uint64_t first();
};
