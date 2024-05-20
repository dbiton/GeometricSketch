#pragma once

#include "Dictionary.h"
#include "MultiHash.h"

class GeometricSketchMH : public Dictionary
{
public:
    GeometricSketchMH(int width, int depth, int branching_factor);
    ~GeometricSketchMH();

    void update(uint32_t key, int amount);
    int query(uint32_t key);

    int shrink(int n);
    int expand(int n);
    int undoExpand(int n);
    int compress(int n);
    uint64_t getMemoryUsage() const;


    // below should be private, but it isn't to allow testing
    // private:
    uint64_t hash(uint32_t key, uint32_t row_id, uint32_t layer_id) const;

    int getLastVectorIndexFromKey(
        uint32_t key,
        uint16_t row_id
    );

    int getNextVectorIndexFromKey(
        uint32_t key,
        uint16_t row_id,
        int& layer_id,
        int& layer_begin_counter_index,
        int& last_counter_index,
        int& B_pow
    );

    int rowIndexToLayerId(int row_offset, int& layer_offset) const;
    int rowIndexToVectorIndex(int row_id, int row_offset) const;

    int vectorIndexToRowId(int vector_index, int& row_offset) const;

    int getRowIndexOfLayer(int layer_id) const;

    int getVectorIndexOfFirstChild(int vector_index) const;
    int getVectorIndexOfParent(int counter_index) const;


    // fields

    MultiHash multihash;

    std::vector<uint32_t> counters;
    unsigned compressed_counters;
    unsigned branching_factor;
    unsigned width;
    unsigned depth;
};