#include "LinkedCellSketch.h"

LinkedCellSketch::LinkedCellSketch(int width, int depth, int branching_factor) : Dictionary(),
                                                                                 branching_factor(branching_factor),
                                                                                 offset(0),
                                                                                 width(width)
{
    for (int i = 0; i < depth; i++)
    {
        rows.push_back(std::vector<uint32_t>(width, 0));
    }
}

LinkedCellSketch::~LinkedCellSketch()
{
}

void LinkedCellSketch::update(uint32_t key, int amount)
{
    for (int row_index = 0; row_index < rows.size(); row_index++)
    {
        auto &row = rows[row_index];
        int counter_key_index = getCounterIndexOfKey(key, row_index);
        row[counter_key_index - offset] += amount;
    }
}

int LinkedCellSketch::query(uint32_t key)
{
    uint32_t estimate = UINT32_MAX;
    for (int row_index = 0; row_index < rows.size(); row_index++)
    {
        auto &row = rows[row_index];
        std::vector<int> indice;
        getCounterIndexOfKey(key, row_index, &indice);
        uint32_t current_estimate = 0;
        for (auto index : indice)
        {
            current_estimate += row[index];
        }
        estimate = std::min(estimate, current_estimate);
    }
    return (int)estimate;
}

int LinkedCellSketch::undoExpand(int n)
{
    int counter_undo;
    for (auto &row : rows)
    {
        counter_undo = 0;
        for (int i_child = row.size() - 1; i_child >= row.size() - n; i_child--)
        {
            int i_parent = getParentIndex(i_child);
            if (i_parent - offset < 0) {
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
    int compress_counter;
    for (auto &row : rows)
    {
        compress_counter = 0;
        for (int i_parent = 0; i_parent < n; i_parent++)
        {
            for (int i_child = getFirstChildIndex(i_parent); i_child < branching_factor; i_child++){
                if (i_child - offset > row.size()) {
                    break;
                }
                row[i_child - offset] += row[i_parent - offset];
                compress_counter++;
            }
        }
        row.erase(row.begin(), row.begin() + compress_counter);
    }
    return compress_counter;
}

void LinkedCellSketch::expand(int n)
{
    for (auto &row : rows)
    {
        row.resize(row.size() + n, 0);
    }
}

void LinkedCellSketch::expand()
{
    expand(width);
}

int LinkedCellSketch::getSize() const
{
    return sizeof(unsigned) * 2 + sizeof(uint32_t) * rows.size() * rows[0].size();
}

void LinkedCellSketch::shrink()
{
    undoExpand(width);
}

int LinkedCellSketch::getMemoryUsage() const
{
    // offset, vector length for each row, vector position, vector length for rows and vector position
    return sizeof(unsigned) * 3 + rows.size() * ((rows[0].size() + 2) * sizeof(uint32_t));
}

void LinkedCellSketch::printRows() const
{
    for (int depth = 0; depth < rows.size(); depth++)
    {
        /*std::cout << "depth:" << depth << ", ";
        const auto &row = rows[depth];
        const auto &layers = row.layers;
        const auto &bitmap = row.bitmap;
        for (int layer_index = 0; layer_index < layers.size(); layer_index++)
        {
            const auto &layer = layers[layer_index];
            std::cout << "layer:" << layer_index << std::endl;
            for (int i = 0; i < layer.size(); i++)
            {
                std::cout << "index:" << i << ",value:" << layer[i] << std::endl;
            }
        }*/
    }
}

int LinkedCellSketch::hash(uint32_t key, uint16_t row_index, uint16_t layer_index) const
{
    uint32_t seed = ((uint32_t)row_index << 16) + (uint32_t)layer_index;
    auto h = murmurhash((int *)&key, seed);
    return layer_index == 0 ? h % width : h % branching_factor;
}

int LinkedCellSketch::getCounterIndexOfKey(uint32_t key, uint16_t row_index, std::vector<int> *indice)
{
    int counter_index = hash(key, row_index, 0);
    if (indice){
        indice->push_back(counter_index);
    }
    int first_layer_offset = width;
    int prev_layer_index = counter_index;
    int layer_factor = branching_factor;
    int added_layers_offset = 0;
    for (int layer=1; ;layer++){
        int child_index = hash(key, row_index, layer);
        int layer_index = prev_layer_index * branching_factor + child_index;
        int next_counter_index = layer_index + first_layer_offset + added_layers_offset;
        if (next_counter_index > offset + rows[0].size()){
            break;
        }
        counter_index = next_counter_index;
        if (indice){
            indice->push_back(counter_index);
        }
        prev_layer_index = layer_index;
        added_layers_offset += layer_factor * width;
        layer_factor *= branching_factor;
    }
    return counter_index;
}

std::pair<int, int> LinkedCellSketch::counterIndexToLayerIndex(int counter_index) const{
    // counter_index = width*(branching_factor^layer-1)/(branching_factor-1) + layer_index * branching_factor ^ layer
    int bf_pow_layer = counter_index * (branching_factor - 1) / width +1;
    int layer = std::floor(log2(bf_pow_layer) / log2(branching_factor));
    int layer_index = std::round((counter_index - width*(bf_pow_layer-1)/(branching_factor-1)) / bf_pow_layer);
    return std::make_pair(layer, layer_index);
}

int LinkedCellSketch::getParentIndex(uint32_t key) const{
    auto p = counterIndexToLayerIndex(key);
    int layer = p.first;
    int layer_index = p.second;
    assert(layer > 0);
    int bf_pow_prev_layer = pow(branching_factor, layer - 1);
    int prev_layer_begin = width * (bf_pow_prev_layer - 1) / (branching_factor - 1);
    return prev_layer_begin + bf_pow_prev_layer * layer_index;
}

int LinkedCellSketch::getFirstChildIndex(uint32_t key) const{
    auto p = counterIndexToLayerIndex(key);
    int layer = p.first;
    int layer_index = p.second;
    int bf_pow_layer = pow(branching_factor, layer);
    int next_layer_begin = width * (bf_pow_layer - 1) / (branching_factor - 1);
    return next_layer_begin + layer_index * bf_pow_layer;
}
