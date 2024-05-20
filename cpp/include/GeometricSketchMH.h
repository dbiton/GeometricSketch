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
    inline uint64_t hash(uint32_t key, uint32_t row_id, uint32_t layer_id) const;

    inline int getLastVectorIndexFromKey(
        uint32_t key,
        uint32_t row_id
    );

    inline int getNextVectorIndexFromKey(
        uint32_t key,
        uint32_t row_id,
        int& prev_layer_id,
        int& prev_layer_row_index,
        int& prev_counter_row_index,
        int& prev_B_pow
    );

    inline int rowIndexToLayerId(int row_offset, int& layer_offset) const;
    inline int rowIndexToVectorIndex(int row_id, int row_offset) const;

    inline int vectorIndexToRowId(int vector_index, int& row_offset) const;

    inline int getRowIndexOfLayer(int layer_id) const;

    inline int getVectorIndexOfFirstChild(int vector_index) const;

    inline int getVectorIndexOfParent(int counter_index) const;


    // fields
    MultiHash multihash;
    std::vector<uint32_t> counters;
    int compressed_counters;
    int branching_factor;
    int width;
    int depth;
};
