#pragma once

#include "Dictionary.h"

class LinkedCellSketch : public Dictionary
{
    struct Cell {
        uint32_t index;
        uint32_t value;
        // index - row for expanded
        
        Cell() : index(0), value(0) {};
    };

    typedef std::vector<Cell> Row;
    
    std::vector<std::vector<Row>> counters;
    
    int size;
    int split_counter_width;
public:
    LinkedCellSketch(int width, int depth, int split_counter_width);
    ~LinkedCellSketch();

    void update(uint32_t key, int amount);
    int query(uint32_t key);

    void expand();
	void shrink();

    int getSize() const;
    int getMemoryUsage() const;
private:
    void getLeaves(int depth, std::vector<int> leaves_indice) const;
    int getIndexInRow(int depth, uint32_t key, int row_index) const;
    int appendRow(int depth, int cell_count);
    void printRows() const;
};
