#include "LinkedCellSketch.h"

LinkedCellSketch::LinkedCellSketch(int width, int depth, int branching_factor) : Dictionary(),
                                                                                 branching_factor(branching_factor),
                                                                                 offset(0),
                                                                                 width(width)
{
    for (int i = 0; i < depth; i++)
    {
        rows.push_back(new std::vector<uint32_t>(width, 0));
    }
}

LinkedCellSketch::~LinkedCellSketch()
{
    for (auto &row : rows)
    {
        delete row;
    }
}

void LinkedCellSketch::update(uint32_t key, int amount)
{
    int R = rows[0]->size();
    int B = (int)branching_factor;
    int W = (int)width;
    int O = offset;
    for (int row_index = 0; row_index < rows.size(); row_index++)
    {
        auto& row = *rows[row_index];
        uint32_t current_estimate = 0;
        int prev_layer_index = 0;
        int prev_layer_begin_counter_index = 0;
        int prev_counter_index = hash(key, row_index, 0) % W;
        int counter_index = prev_counter_index;
        int prev_B_pow_layer_index = 1;
        while (counter_index < R + O) {
            if (counter_index >= O) {
                long actual_index = (long)counter_index - O;
                row[actual_index] += amount;
            }
            counter_index = getNextLayerCounterIndexFromKey(key, row_index, prev_layer_index,
                prev_layer_begin_counter_index, prev_counter_index, prev_B_pow_layer_index);
        }
    }
}

int LinkedCellSketch::query(uint32_t key)
{
    int R = rows[0]->size();
    int B = (int)branching_factor;
    int W = (int)width;
    int O = offset;
    uint32_t estimate = UINT32_MAX;
    for (int row_index = 0; row_index < rows.size(); row_index++)
    {
        auto& row = *rows[row_index];
        uint32_t current_estimate = 0;
        int prev_layer_index = 0;
        int prev_layer_begin_counter_index = 0;
        int prev_counter_index = hash(key, row_index, 0) % W;
        int prev_B_pow_layer_index = 1;
        int counter_index = prev_counter_index;
        while (counter_index < R + O){
            if (counter_index >= O) {
                long actual_index = (long)counter_index - O;
                current_estimate += row[actual_index];
            }
            counter_index = getNextLayerCounterIndexFromKey(key, row_index, prev_layer_index,
                prev_layer_begin_counter_index, prev_counter_index, prev_B_pow_layer_index);
        }
        estimate = std::min(estimate, current_estimate);
    }
    return estimate;
    /*
    std::vector<uint32_t> estimates;
    for (int row_index = 0; row_index < rows.size(); row_index++)
    {
        auto& row = *rows[row_index];
        std::vector<int> indice;
        getAllLayersCounterIndiceFromKey(key, row_index, indice);
        uint32_t current_estimate = 0;
        for (int i = 0; i<indice.size(); i++)
        {
            while (estimates.size() <= i) {
                estimates.push_back(INT32_MAX);
            }
            int counter_index = indice[i];
            estimates[i] = min(row[counter_index - offset], estimates[i]);
        }
    }
    int result = 0;
    for (const auto& e : estimates) result += e;
        
    return result;
    */
}

int LinkedCellSketch::undoExpand(int n)
{
    int counter_undo = 0;
    for (auto &row_ptr : rows)
    {
        auto &row = *row_ptr;
        counter_undo = 0;
        for (int i_child = row.size() - 1; i_child >= row.size() - n; i_child--)
        {
            int i_parent = getCounterParentIndex(i_child);
            if (i_parent == -1 || i_parent - offset < 0)
            {
                break;
            }
            long actual_index_parent = (long)i_parent - offset;
            long actual_index_child = (long)i_child - offset;
            row[actual_index_parent] += row[actual_index_child];
            counter_undo++;
        }
        row.resize(row.size() - counter_undo);
    }
    return counter_undo;
}

int LinkedCellSketch::compress(int n)
{
    size_t R = rows[0]->size();
    size_t B = branching_factor;
    size_t O = offset;
    size_t W = width;
    int compress_counter = 0;
    for (auto &row_ptr : rows)
    {
        auto &row = *row_ptr;
        compress_counter = 0;
        for (int counter_index_parent = offset; counter_index_parent < offset+n; counter_index_parent++)
        {
            size_t counter_index_first_child = getCounterFirstChildIndex(counter_index_parent);
            // can't compress counter if some of it's children are missing
            if (counter_index_first_child + branching_factor - offset - 1 >= row.size()){
                break;
            }
            compress_counter++;
            for (int index_child = 0; index_child < branching_factor; index_child++)
            {
                size_t counter_index_child = counter_index_first_child + index_child;
                row[counter_index_child - O ] += row[counter_index_parent - O];
            }
        }
        row.erase(row.begin(), row.begin() + compress_counter);
    }
    offset += compress_counter;
    return compress_counter;
}

int LinkedCellSketch::expand(int n)
{
    for (auto &row_ptr : rows)
    {
        auto &row = *row_ptr;
        row.resize(row.size() + n, 0);
    }
    return n;
}

int LinkedCellSketch::shrink(int n)
{
    return undoExpand(n);
}

int LinkedCellSketch::getSize() const
{
    return sizeof(unsigned) * 2 + sizeof(uint32_t) * rows.size() * rows[0]->size();
}

int LinkedCellSketch::getMemoryUsage() const
{
    // offset, vector length for each row, vector position, vector length for rows and vector position
    return rows.size() * rows[0]->size() * sizeof(uint32_t);
}

void LinkedCellSketch::printRows() const
{
    size_t R = rows[0]->size();
    size_t B = branching_factor;
    size_t W = width;
    for (int depth = 0; depth < rows.size(); depth++)
    {
        std::cout << std::endl << "row index:" << depth;
        for (int layer_index = 0; getLayerFirstCounterIndex(layer_index) < R+offset; layer_index++)
        {
            std::cout << std::endl << "layer index:" << layer_index << ">";
            int layer_length = W * std::pow(B, layer_index);
            for (int layer_offset = 0; layer_offset < layer_length; layer_offset++)
            {
                size_t counter_index = W*(pow(B, layer_index)-1)/(B-1) + layer_offset;
                int counter_value = 0;
                if (counter_index < R + offset && counter_index >= offset){
                    counter_value = (*rows[depth])[counter_index - offset];
                }
                std::cout << "," << std::to_string(counter_value);
            }
        }
    }
}

int LinkedCellSketch::hash(uint32_t key, uint16_t row_index, uint16_t layer_index) const
{
    uint32_t seed = ((uint32_t)row_index << 16) + (uint32_t)layer_index;
    auto h = murmurhash((int *)&key, seed);
    h &= 0x3FF; // USE LOWEST 10 BITS
    // use 16 bit hash and replace % with something else
    return layer_index == 0 ? h % width : h % branching_factor;
}

size_t LinkedCellSketch::getLayerIndexOfCounterIndex(int counter_index) const
{
    int B = branching_factor;
    int C = counter_index;
    int layer_width = width;
    size_t layer = 0;
    while (C >= layer_width) {
        C -= layer_width;
        layer_width *= B;
        layer++;
    }
    return layer;
}

int LinkedCellSketch::getLayerFirstCounterIndex(int layer_index) const
{
    int L = layer_index;
    int B = (int)branching_factor;
    int W = (int)width;
    int B_raised_L = 1;
    // better than pow for our range of values - checked myself
    for (int i = 0; i < L; i++) B_raised_L *= B;
    return W * (1 - B_raised_L) / (1 - B);
}

int LinkedCellSketch::getCounterIndexFromChildIndice(const std::vector<int> &child_indice) const
{
    int L = child_indice.size() - 1;
    int B = (int)branching_factor;

    int row_index = getLayerFirstCounterIndex(L);
    int B_raised = 1;
    for (int i = L; i >= 0; i--)
    {
        row_index += child_indice[i] * B_raised;
        B_raised *= B;
    }
    return row_index;
}


int LinkedCellSketch::getLastLayerCounterIndexFromKey(uint32_t key, uint16_t row_index) const
{
    std::vector<int> child_indice;
    int R = rows[0]->size();
    size_t max_layer_index = getLayerIndexOfCounterIndex(R+offset);
    child_indice.reserve(max_layer_index+1);
    for (int layer_index = 0; layer_index <= max_layer_index; layer_index++)
    {
        child_indice.push_back(hash(key, row_index, layer_index));
    }
    int last_layer_counter_index = getCounterIndexFromChildIndice(child_indice);
    if (last_layer_counter_index - offset >= R)
    {
        child_indice.pop_back();
        int counter_index = getCounterIndexFromChildIndice(child_indice);
        return counter_index;
    }

    return last_layer_counter_index;
}


int LinkedCellSketch::getNextLayerCounterIndexFromKey(
    uint32_t key, 
    uint16_t row_index, 
    int& prev_layer_index, 
    int& prev_layer_begin_counter_index, 
    int& prev_counter_index,
    int& prev_B_pow_layer_index
) const{
    int B = (int)branching_factor;
    int W = (int)width;
    int layer_begin_counter_index = prev_layer_begin_counter_index + W * prev_B_pow_layer_index;
    int h = hash(key, row_index, prev_layer_index+1);
    int counter_index = (prev_counter_index - prev_layer_begin_counter_index) * B + h + layer_begin_counter_index;
    prev_layer_index += 1;
    prev_layer_begin_counter_index = layer_begin_counter_index;
    prev_counter_index = counter_index;
    prev_B_pow_layer_index *= B;
    return counter_index;
}

void LinkedCellSketch::getAllLayersCounterIndiceFromKey(uint32_t key, uint16_t row_index, std::vector<int> &counter_indice) const
{
    counter_indice.clear();
    std::vector<int> child_indice;
    int R = rows[0]->size();
    size_t max_layer_index = getLayerIndexOfCounterIndex(R+offset);
    child_indice.reserve(1 + max_layer_index);
    counter_indice.reserve(1 + max_layer_index);
    for (int layer_index = 0; layer_index < max_layer_index; layer_index++)
    {
        int h = hash(key, row_index, layer_index);
        child_indice.push_back(h);
        int counter_index = getCounterIndexFromChildIndice(child_indice);
        if (counter_index - offset >= 0){
            counter_indice.push_back(counter_index);
        }
    }
    child_indice.push_back(hash(key, row_index, max_layer_index));
    int counter_index = getCounterIndexFromChildIndice(child_indice);
    if (counter_index - offset < R)
    {
        counter_indice.push_back(counter_index);
    }
}

int LinkedCellSketch::getCounterParentIndex(int counter_index) const
{
    int B = (int)branching_factor;
    int W = (int)width;
    int layer_index = getLayerIndexOfCounterIndex(counter_index);
    if (layer_index <= 0){
        return -1;
    }
    int layer_begin = getLayerFirstCounterIndex(layer_index);
    int layer_offset = counter_index - layer_begin;
    int parent_layer_index = layer_index - 1;
    int parent_layer_offset = layer_offset / B;
    int parent_layer_length = W * pow(B, parent_layer_index);
    int parent_layer_begin = layer_begin - parent_layer_length;
    return parent_layer_begin + parent_layer_offset;
}

int LinkedCellSketch::getCounterFirstChildIndex(int counter_index) const
{
    int B = (int)branching_factor;
    int W = (int)width;
    int layer_index = getLayerIndexOfCounterIndex(counter_index);
    int layer_begin = getLayerFirstCounterIndex(layer_index);
    int layer_offset = counter_index - layer_begin;
    int layer_length = W * pow(B, layer_index);
    int child_layer_offset = layer_offset * B;
    int child_layer_begin = layer_begin + layer_length;
    return child_layer_begin + child_layer_offset;
}
