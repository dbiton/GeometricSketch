#pragma once

#include "Dictionary.h"

class LinkedCellSketch : public Dictionary
{
    std::vector<std::vector<uint32_t>*> rows;
    int offset;
    unsigned branching_factor;
    unsigned width;
public:
    LinkedCellSketch(int width, int depth, int branching_factor);
    ~LinkedCellSketch();

    void update(uint32_t key, int amount);
    int query(uint32_t key);

    int shrink(int n);
    int expand(int n);
    int undoExpand(int n);
    int compress(int n);

    int getSize() const;
    int getMemoryUsage() const;
private:
    int hash(uint32_t key, uint16_t row_index, uint16_t layer_index) const;

    int getNextLayerCounterIndexFromKey(
        uint32_t key,
        uint16_t row_index,
        int& layer_index,
        int& layer_begin_counter_index,
        int& last_counter_index,
        int& B_pow_layer_index
    ) const;
    size_t getLayerIndexOfCounterIndex(int counter_index) const;
    void getAllLayersCounterIndiceFromKey(uint32_t key, uint16_t row_index, std::vector<int>& counter_indice) const;
    int getCounterIndexFromChildIndice(const std::vector<int>& child_indice) const;
    int getLastLayerCounterIndexFromKey(uint32_t key, uint16_t row_index) const;
    int getLayerFirstCounterIndex(int layer_index) const;
    int getCounterParentIndex(int counter_index) const;
    int getCounterFirstChildIndex(int counter_index) const;

    void printRows() const;
};
