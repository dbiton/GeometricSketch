#include "LinkedCellSketch.h"
#include "xxhash.h"

LinkedCellSketch::LinkedCellSketch(int width, int depth, int branching_factor) : 
    Dictionary(),
    width(width),
    depth(depth),
    branching_factor(branching_factor),
    offset(0),
    counters(width * depth, 0)
{
    /*
        Counter Layer Index - row_index, layer_index, layer_offset
        Counter Row Index - row_index, row_offset
        Counter Canonical Index - vector offset
    */
}

LinkedCellSketch::~LinkedCellSketch()
{
}

void LinkedCellSketch::update(uint32_t key, int amount)
{
    for (int row_index = 0; row_index < depth; row_index++)
    {
        int vector_offset = getLastLayerVectorOffsetFromKey(key, row_index);
        long actual_index = (long)vector_offset - offset;
        counters[actual_index] += amount;
    }
}

int LinkedCellSketch::getLastLayerVectorOffsetFromKey(
    uint32_t key,
    uint16_t row_index
) {
    int prev_layer_index = 0;
    int prev_layer_begin_counter_index = 0;
    int prev_row_offset = hash(key, row_index, 0) % width;
    int vector_offset = rowOffsetToVectorOffset(row_index, prev_row_offset);
    int prev_B_pow_layer_index = 1;
    int prev_vector_offset = vector_offset;
    while (vector_offset < counters.size() + offset) {
        prev_vector_offset = vector_offset;
        vector_offset = getNextLayerVectorOffsetFromKey(key, row_index, prev_layer_index,
            prev_layer_begin_counter_index, prev_row_offset, prev_B_pow_layer_index);
    }
    return prev_vector_offset;
}

int LinkedCellSketch::getNextLayerVectorOffsetFromKey(
    uint32_t key, 
    uint16_t row_index, 
    int& prev_layer_index, 
    int& prev_layer_row_offset,
    int& prev_counter_row_offset,
    int& prev_B_pow_layer_index
) const{
    int B = (int)branching_factor;
    int W = (int)width;
    int layer_begin_counter_index = prev_layer_row_offset + W * prev_B_pow_layer_index;
    int h = hash(key, row_index, prev_layer_index + 1) % B;
    int counter_index = (prev_counter_row_offset - prev_layer_row_offset) * B + h + layer_begin_counter_index;
    prev_layer_index += 1;
    prev_layer_row_offset = layer_begin_counter_index;
    prev_counter_row_offset = counter_index;
    prev_B_pow_layer_index *= B;
    int vector_offset = rowOffsetToVectorOffset(row_index, counter_index);
    return vector_offset;
}

int LinkedCellSketch::query(uint32_t key)
{
    int O = offset;
    uint32_t estimate = UINT32_MAX;
    for (int row_index = 0; row_index < depth; row_index++)
    {
        uint32_t current_estimate = 0;
        int prev_layer_index = 0;
        int prev_layer_begin_counter_index = 0;
        int prev_row_offset = hash(key, row_index, 0) % width;
        int prev_B_pow_layer_index = 1;
        int vector_offset = rowOffsetToVectorOffset(row_index, prev_row_offset);
        while (vector_offset < counters.size() + offset){
            if (vector_offset >= O) {
                long actual_index = (long)vector_offset - O;
                current_estimate += counters[actual_index];
            }
            vector_offset = getNextLayerVectorOffsetFromKey(key, row_index, prev_layer_index,
                prev_layer_begin_counter_index, prev_row_offset, prev_B_pow_layer_index);
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

// this and compress can be improved by not using getVectorOffsetParent each time
// and instead going from parent to parent but it would make the function uglier and
// I am lazy
int LinkedCellSketch::undoExpand(int n)
{
    int counter_undo = 0;
    // start from the last counter
    //std::cout << "undoExpand:" << std::endl;
    for (int i_child = counters.size()-1; i_child >= counters.size() - n; i_child--)
    {
        long actual_index_child = i_child - offset;
        //std::cout << "child:" << i_child << std::endl;
        int i_parent = getVectorOffsetParent(actual_index_child);
        if (i_parent == -1 || i_parent - offset < 0)
        {
            break;
        }
        //std::cout << "parent:" << i_parent << std::endl;
        long actual_index_parent = (long)i_parent - offset;
        counters[actual_index_parent] += counters[actual_index_child];
        counter_undo++;
    }
    counters.resize(counters.size() - counter_undo);
    return counter_undo;
}

/*
int compress_counter = 0;
    int counter_index_parent = offset;
    int counter_index_first_child = getVectorOffsetFirstChild(counter_index_parent);
    while (counter_index_parent < offset + n) {
        //std::cout << "parent:" << counter_index_parent << std::endl;
        //std::cout << "first child:" << counter_index_first_child << std::endl;
        int counter_index_last_child = counter_index_first_child + (branching_factor - 1) * depth;
        //std::cout << "last child:" << counter_index_last_child << std::endl;
        if (counter_index_last_child >= offset + counters.size()) {
            break;
        }
        compress_counter++;
        long counter_index_parent_actual = counter_index_parent - offset;
        for (int index_child = counter_index_first_child; index_child <= counter_index_last_child; index_child += depth)
        {
            long counter_index_child_actual = index_child - offset;
            counters[counter_index_child_actual] += counters[counter_index_parent_actual];
        }
        counter_index_parent++;
        int child_incr_amount = counter_index_parent % depth == 0 ? depth + 1 : 1;
        counter_index_first_child+=child_incr_amount;
    }
    counters.erase(counters.begin(), counters.begin() + compress_counter);
    offset += compress_counter;
    return compress_counter;
*/

// compress takes into account offset
int LinkedCellSketch::compress(int n)
{
    int compress_counter = 0;
    int counter_index_parent = offset;
    int counter_index_first_child = getVectorOffsetFirstChild(counter_index_parent);
    while (counter_index_parent < offset + n) {
        //std::cout << "parent:" << counter_index_parent << std::endl;
        //std::cout << "first child:" << counter_index_first_child << std::endl;
        int counter_index_last_child = counter_index_first_child + (branching_factor - 1) * depth;
        //std::cout << "last child:" << counter_index_last_child << std::endl;
        if (counter_index_last_child >= offset + counters.size()) {
            break;
        }
        compress_counter++;
        long counter_index_parent_actual = counter_index_parent - offset;
        for (int index_child = counter_index_first_child; index_child <= counter_index_last_child; index_child += depth)
        {
            long counter_index_child_actual = index_child - offset;
            counters[counter_index_child_actual] += counters[counter_index_parent_actual];
        }
        counter_index_parent++;
        int child_incr_amount = counter_index_parent % depth == 0 ? depth + 1 : 1;
        counter_index_first_child+=child_incr_amount;
    }
    counters.erase(counters.begin(), counters.begin() + compress_counter);
    offset += compress_counter;
    return compress_counter;
}

int LinkedCellSketch::expand(int n)
{
    counters.resize(counters.size() + n, 0);
    return n;
}

int LinkedCellSketch::shrink(int n)
{
    return undoExpand(n);
}

int LinkedCellSketch::getMemoryUsage() const
{
    // offset, vector length for each row, vector position, vector length for rows and vector position
    return counters.size() * sizeof(uint32_t) + sizeof(unsigned) * 4;
}

/*
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
}*/

int LinkedCellSketch::hash(uint32_t key, uint16_t row_index, uint16_t layer_index) const
{
    uint32_t seed = ((uint32_t)row_index << 16) + (uint32_t)layer_index;
    return XXH32(&key, sizeof(key), seed);
}

int LinkedCellSketch::rowOffsetToLayerIndex(int row_offset, int& layer_offset) const
{
    int B = branching_factor;
    layer_offset = row_offset;
    int layer_width = width;
    int layer = 0;
    while (layer_offset >= layer_width) {
        layer_offset -= layer_width;
        layer_width *= B;
        layer++;
    }
    return layer;
}

int LinkedCellSketch::getVectorOffsetParent(int vector_offset) const
{
    int B = (int)branching_factor;
    int W = (int)width;
    int counter_row_offset, counter_layer_offset;
    int row_index = vectorOffsetToRowIndex(vector_offset, counter_row_offset);
    int counter_layer_index = rowOffsetToLayerIndex(counter_row_offset, counter_layer_offset);
    if (counter_layer_index <= 0){
        return -1;
    }
    int parent_layer_offset = counter_layer_offset / B;
    int parent_row_offset = parent_layer_offset + getLayerRowOffset(counter_layer_index - 1);
    return rowOffsetToVectorOffset(row_index, parent_row_offset);
}

int LinkedCellSketch::rowOffsetToVectorOffset(int row_index, int row_offset) const {
    return row_offset * depth + row_index;
}

int LinkedCellSketch::getVectorOffsetFirstChild(int vector_offset) const
{
    int B = (int)branching_factor;
    int W = (int)width;
    int counter_row_offset, counter_layer_offset;
    int row_index = vectorOffsetToRowIndex(vector_offset, counter_row_offset);
    int counter_layer_index = rowOffsetToLayerIndex(counter_row_offset, counter_layer_offset);
    int child_layer_offset = counter_layer_offset * B;
    int child_row_offset = child_layer_offset + getLayerRowOffset(counter_layer_index + 1);
    return rowOffsetToVectorOffset(row_index, child_row_offset);
}

int LinkedCellSketch::vectorOffsetToRowIndex(int vector_offset, int& row_offset) const {
    row_offset = vector_offset / depth;
    int row_index = vector_offset % depth;
    return row_index;
}

int LinkedCellSketch::getLayerRowOffset(int layer_index) const
{
    int L = layer_index;
    int B = (int)branching_factor;
    int W = (int)width;
    int B_raised_L = 1;
    // better than pow for our range of values - checked myself
    for (int i = 0; i < L; i++) B_raised_L *= B;
    return W * (1 - B_raised_L) / (1 - B);
}