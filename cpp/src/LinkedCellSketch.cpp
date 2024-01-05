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
    for (int row_index = 0; row_index < rows.size(); row_index++)
    {
        auto &row = *rows[row_index];
        int counter_key_index = getLastLayerCounterIndexFromKey(key, row_index);
        row[counter_key_index - offset] += amount;
        if (counter_key_index - offset < 0 || counter_key_index - offset >= row.size()){
            int x = 3;
        }
    }
}

int LinkedCellSketch::query(uint32_t key)
{
    uint32_t estimate = UINT32_MAX;
    for (int row_index = 0; row_index < rows.size(); row_index++)
    {
        auto &row = *rows[row_index];
        std::vector<int> indice;

        getAllLayersCounterIndiceFromKey(key, row_index, indice);
        uint32_t current_estimate = 0;
        for (const auto& counter_index : indice)
        {
            if (counter_index - offset < 0 || counter_index - offset >= row.size()){
                int x = 3;
            }
            current_estimate += row[counter_index - offset];
        }
        estimate = std::min(estimate, current_estimate);
    }
    return (int)estimate;
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
            if (i_parent - offset < 0)
            {
                break;
            }
            row[i_parent - offset] += row[i_child - offset];
            counter_undo++;
        }
        row.erase(row.end() - counter_undo);
    }
    return counter_undo;
}

int LinkedCellSketch::compress(int n)
{
    int compress_counter = 0;
    for (auto &row_ptr : rows)
    {
        auto &row = *row_ptr;
        compress_counter = 0;
        for (int counter_index_parent = offset; counter_index_parent < offset+n; counter_index_parent++)
        {
            int counter_index_first_child = getCounterFirstChildIndex(counter_index_parent);
            // can't compress counter if some of it's children are missing
            if (counter_index_first_child + branching_factor - offset - 1 >= row.size()){
                break;
            }
            compress_counter++;
            for (int index_child = 0; index_child < branching_factor; index_child++)
            {
                int counter_index_child = counter_index_first_child + index_child;
                row[counter_index_child - offset ] += row[counter_index_parent - offset];
            }
        }
        row.erase(row.begin(), row.begin() + compress_counter);
    }
    offset += compress_counter;
    return compress_counter;
}

void LinkedCellSketch::expand(int n)
{
    for (auto &row_ptr : rows)
    {
        auto &row = *row_ptr;
        row.resize(row.size() + n, 0);
    }
}

void LinkedCellSketch::shrink(int n)
{
    compress(n);
}

int LinkedCellSketch::getSize() const
{
    return sizeof(unsigned) * 2 + sizeof(uint32_t) * rows.size() * rows[0]->size();
}

int LinkedCellSketch::getMemoryUsage() const
{
    // offset, vector length for each row, vector position, vector length for rows and vector position
    return sizeof(unsigned) * 3 + rows.size() * ((rows[0]->size() + 2) * sizeof(uint32_t));
}

void LinkedCellSketch::printRows() const
{
    int R = rows[0]->size();
    int B = (int)branching_factor;
    int W = (int)width;
    for (int depth = 0; depth < rows.size(); depth++)
    {
        std::cout << "row index:" << depth << std::endl;
        bool stop = false;
        int offset = 0;
        assert(false); // doesnt consider offset
        for (int layer_index = 0; !stop; layer_index++)
        {
            std::cout << "layer index:" << layer_index << std::endl;
            int layer_length = W * std::pow(B, layer_index);
            for (int layer_offset = 0; layer_offset < layer_length; layer_offset++)
            {
                int counter_index = offset + layer_offset;
                if (counter_index >= R){
                    stop = true;
                    break;
                }
                std::cout << "layer offset:" << layer_offset << ",counter index:";
                std::cout << counter_index << ",value:" << (*rows[depth])[counter_index] << std::endl;
            }
            offset += layer_length;
        }
    }
}

int LinkedCellSketch::hash(uint32_t key, uint16_t row_index, uint16_t layer_index) const
{
    uint32_t seed = ((uint32_t)row_index << 16) + (uint32_t)layer_index;
    auto h = murmurhash((int *)&key, seed);
    return layer_index == 0 ? h % width : h % branching_factor;
}

int LinkedCellSketch::getLayerIndexOfCounterIndex(int counter_index) const
{
    float B = branching_factor;
    float C = counter_index;
    float W = width;
    // if counter_index == 0, we return num_layers=-1 instead of 0 due to something with log... this handles the edge case
    if (counter_index < W){
        return 0;
    }
    int num_layers = std::ceil(log2(C / W * (B - 1) + 1) / log2(B));
    return num_layers - 1;
}

int LinkedCellSketch::getLayerFirstCounterIndex(int layer_index) const
{
    int L = layer_index;
    int B = (int)branching_factor;
    int W = (int)width;
    return W * (1 - pow(B, L)) / (1 - B);
}

int LinkedCellSketch::getCounterIndexFromChildIndice(const std::vector<int> &child_indice) const
{
    int L = child_indice.size() - 1;
    int B = (int)branching_factor;

    int row_index = getLayerFirstCounterIndex(L);
    for (int i = 0; i <= L; i++)
    {
        row_index += child_indice[i] * pow(B, L - i);
    }
    return row_index;
}

int LinkedCellSketch::getLastLayerCounterIndexFromKey(uint32_t key, uint16_t row_index) const
{
    std::vector<int> child_indice;
    int R = rows[0]->size();
    int max_layer_index = getLayerIndexOfCounterIndex(R+offset);
    for (int layer_index = 0; layer_index <= max_layer_index; layer_index++)
    {
        child_indice.push_back(hash(key, row_index, layer_index));
    }
    int last_layer_counter_index = getCounterIndexFromChildIndice(child_indice);
    if (last_layer_counter_index - offset >= R)
    {
        child_indice.pop_back();
        int counter_index = getCounterIndexFromChildIndice(child_indice);
        if (counter_index - offset < 0 || counter_index - offset >= R){
            int x = 3;
        }
        return counter_index;
    }

    return last_layer_counter_index;
}

void LinkedCellSketch::getAllLayersCounterIndiceFromKey(uint32_t key, uint16_t row_index, std::vector<int> &counter_indice) const
{
    counter_indice.clear();
    std::vector<int> child_indice;
    int R = rows[0]->size();
    int max_layer_index = getLayerIndexOfCounterIndex(R+offset);
    for (int layer_index = 0; layer_index < max_layer_index; layer_index++)
    {
        child_indice.push_back(hash(key, row_index, layer_index));
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
    assert(layer_index > 0);
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
