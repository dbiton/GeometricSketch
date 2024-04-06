#include "GeometricSketch.h"
#include "xxhash.h"

GeometricSketch::GeometricSketch(int width, int depth, int branching_factor) : 
    Dictionary(),
    width(width),
    depth(depth),
    branching_factor(branching_factor),
    offset(0),
    counters(width * depth, 0)
{
}

GeometricSketch::~GeometricSketch()
{
}

void GeometricSketch::update(uint32_t key, int amount)
{
    for (unsigned row_index = 0; row_index < depth; row_index++)
    {
        int vector_offset = getLastLayerVectorOffsetFromKey(key, row_index);
        long actual_index = (long)vector_offset - offset;
        counters[actual_index] += amount;
    }
}

int GeometricSketch::getLastLayerVectorOffsetFromKey(
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

int GeometricSketch::getNextLayerVectorOffsetFromKey(
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

int GeometricSketch::query(uint32_t key)
{
    int O = offset;
    uint32_t estimate = UINT32_MAX;
    for (uint32_t row_index = 0; row_index < depth; row_index++)
    {
        uint32_t current_estimate = 0;
        int prev_layer_index = 0;
        int prev_layer_begin_counter_index = 0;
        int prev_row_offset = hash(key, row_index, 0UL) % width;
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
}

int GeometricSketch::undoExpand(int n)
{
    int counter_undo = 0;
    for (int i_child = (int)counters.size() - 1; i_child >= (int)counters.size() - n; i_child--)
    {
        long actual_index_child = i_child - offset;
        int i_parent = getVectorOffsetParent(actual_index_child);
        if (i_parent == -1 || i_parent - offset < 0)
        {
            break;
        }
        long actual_index_parent = (long)i_parent - offset;
        counters[actual_index_parent] += counters[actual_index_child];
        counter_undo++;
    }
    counters.resize(counters.size() - counter_undo);

    return counter_undo;
}

int GeometricSketch::compress(int n)
{
    int compress_counter = 0;
    size_t counter_index_parent = offset;
    int counter_index_first_child = getVectorOffsetFirstChild(counter_index_parent);
    while (counter_index_parent < (size_t)n + offset) {
        int counter_index_last_child = counter_index_first_child + (branching_factor - 1) * depth;
        if (counter_index_last_child >= offset + counters.size()) {
            break;
        }
        long counter_index_parent_actual = counter_index_parent - offset;
        for (int index_child = counter_index_first_child; index_child <= counter_index_last_child; index_child += depth)
        {
            long counter_index_child_actual = index_child - offset;
            counters[counter_index_child_actual] += counters[counter_index_parent_actual];
        }
        compress_counter++;
        counter_index_parent++;
        counter_index_first_child = (counter_index_parent % depth == 0) ? counter_index_last_child + 1 : counter_index_first_child + 1;
    }
    counters.erase(counters.begin(), counters.begin() + compress_counter);
    offset += compress_counter;
    return compress_counter;
}

void GeometricSketch::print()
{
    for (auto counter : counters) {
        std::cout << counter << std::endl;
    }
}

int GeometricSketch::expand(int n)
{
    counters.resize(counters.size() + n, 0);
    return n;
}

int GeometricSketch::shrink(int n)
{
    return undoExpand(n);
}

int GeometricSketch::getMemoryUsage() const
{
    return (int)counters.size() * sizeof(uint32_t) + sizeof(unsigned) * 4;
}

uint64_t GeometricSketch::hash(uint32_t key, uint32_t row_index, uint32_t layer_index) const
{
    uint64_t seed = ((uint64_t)layer_index << 32) | row_index;
    return XXH64(&key, sizeof(key), seed);
}

int GeometricSketch::rowOffsetToLayerIndex(int row_offset, int& layer_offset) const
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

int GeometricSketch::getVectorOffsetParent(int vector_offset) const
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

int GeometricSketch::rowOffsetToVectorOffset(int row_index, int row_offset) const {
    return row_offset * depth + row_index;
}

int GeometricSketch::getVectorOffsetFirstChild(int vector_offset) const
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

int GeometricSketch::vectorOffsetToRowIndex(int vector_offset, int& row_offset) const {
    row_offset = vector_offset / depth;
    int row_index = vector_offset % depth;
    return row_index;
}

int GeometricSketch::getLayerRowOffset(int layer_index) const
{
    int L = layer_index;
    int B = (int)branching_factor;
    int W = (int)width;
    int B_raised_L = 1;
    // better than pow for our range of values - checked myself
    for (int i = 0; i < L; i++) B_raised_L *= B;
    return W * (1 - B_raised_L) / (1 - B);
}
