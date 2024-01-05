#pragma once

#include "Dictionary.h"

class LinkedCellSketch : public Dictionary
{
    std::vector<std::vector<uint32_t>> rows;
    int offset;
    unsigned branching_factor;
    unsigned width;
public:
    LinkedCellSketch(int width, int depth, int branching_factor);
    ~LinkedCellSketch();

    void update(uint32_t key, int amount);
    int query(uint32_t key);

    void expand(int n);
    int undoExpand(int n);
    int compress(int n);

    void expand();
	void shrink();

    int getSize() const;
    int getMemoryUsage() const;
private:
    int hash(uint32_t key, uint16_t row_index, uint16_t layer_index) const;
    int getCounterIndexOfKey(uint32_t key, uint16_t row_index, std::vector<int>* indice=nullptr);
    
    std::pair<int,int> counterIndexToLayerIndex(int counter_index) const;
    int getParentIndex(uint32_t key) const;
    int getFirstChildIndex(uint32_t key) const;
    
    int layerIndexToRowIndex(int layer, int layer_index);
    int appendRow(int depth, int cell_count);
    void printRows() const;
};
