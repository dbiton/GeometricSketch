#include "GeometricSketchMH.h"
#include "xxhash.h"
#include "doctest.h"

GeometricSketchMH::GeometricSketchMH(int width, int depth, int branching_factor) :
    Dictionary(),
    width(width),
    depth(depth),
    branching_factor(branching_factor),
    compressed_counters(0),
    counters(width* depth, 0)
{
    multihash.setFirstSubHashModulus(width);
    multihash.setSubHashModulus(branching_factor);
}

GeometricSketchMH::~GeometricSketchMH()
{
}

void GeometricSketchMH::update(uint32_t key, int amount)
{
    for (uint32_t row_id = 0; row_id < depth; row_id++)
    {
        multihash.initialize(key, row_id);
        long vector_index = getLastVectorIndexFromKey(key, row_id);
        long actual_index = vector_index - compressed_counters;
        counters[actual_index] += amount;
    }
}

int GeometricSketchMH::getLastVectorIndexFromKey(
    uint32_t key,
    uint32_t row_id
) {
    int prev_layer_id = 0;
    int prev_layer_begin_counter_index = 0;
    int prev_row_index = multihash.first();
    int vector_index = rowIndexToVectorIndex(row_id, prev_row_index);
    int prev_B_pow = 1;
    const int max_vector_index = counters.size() + compressed_counters - 1;
    int prev_vector_index = vector_index;
    for (
        ;
        vector_index != -1;
        vector_index = getNextVectorIndexFromKey(key, row_id,
            prev_layer_id, prev_layer_begin_counter_index, prev_row_index, prev_B_pow)
        ) {
        prev_vector_index = vector_index;
    }
    return prev_vector_index;
}

int GeometricSketchMH::getNextVectorIndexFromKey(
    uint32_t key,
    uint32_t row_id,
    int& prev_layer_id,
    int& prev_layer_row_index,
    int& prev_counter_row_index,
    int& prev_B_pow
) {
    const int B = (int)branching_factor;
    const int W = (int)width;
    const int layer_begin_counter_index = prev_layer_row_index + W * prev_B_pow;
    const int counter_index_first_child = (prev_counter_row_index - prev_layer_row_index) * B + layer_begin_counter_index;
    const int max_vector_index = counters.size() + compressed_counters;
    if (rowIndexToVectorIndex(row_id, counter_index_first_child) >= max_vector_index) {
        return -1;
    }
    const int h = multihash.next();
    const int counter_index = counter_index_first_child + h;
    prev_layer_id += 1;
    prev_layer_row_index = layer_begin_counter_index;
    prev_counter_row_index = counter_index;
    prev_B_pow *= B;
    const int vector_index = rowIndexToVectorIndex(row_id, counter_index);
    return (vector_index >= max_vector_index) ? -1 : vector_index;
}

int GeometricSketchMH::query(uint32_t key)
{
    int O = compressed_counters;
    uint32_t estimate = UINT32_MAX;
    for (uint32_t row_id = 0; row_id < depth; row_id++)
    {
        multihash.initialize(key, row_id);
        uint32_t current_estimate = 0;
        int prev_layer_id = 0;
        int prev_layer_begin_counter_index = 0;
        int prev_row_index = hash(key, row_id, 0UL) % width;
        int prev_B_pow = 1;
        long vector_index = rowIndexToVectorIndex(row_id, prev_row_index);
        for (
            ;
            vector_index != -1;
            vector_index = getNextVectorIndexFromKey(key, row_id,
                prev_layer_id, prev_layer_begin_counter_index, prev_row_index, prev_B_pow)
            ) {
            if (vector_index >= O) {
                long actual_index = vector_index - O;
                current_estimate += counters[actual_index];
            }
        }
        estimate = std::min(estimate, current_estimate);
    }
    return estimate;
}

int GeometricSketchMH::undoExpand(int n)
{
    int counter_undo = 0;
    for (int i_child = (int)counters.size() - 1; i_child >= (int)counters.size() - n; i_child--)
    {
        long actual_index_child = i_child - compressed_counters;
        long i_parent = getVectorIndexOfParent(actual_index_child);
        if (i_parent == -1 || i_parent - (int)compressed_counters < 0)
        {
            break;
        }
        long actual_index_parent = i_parent - compressed_counters;
        counters[actual_index_parent] += counters[actual_index_child];
        counter_undo++;
    }
    counters.resize(counters.size() - counter_undo);
    return counter_undo;
}

int GeometricSketchMH::compress(int n)
{
    int compress_counter = 0;
    size_t counter_index_parent = compressed_counters;
    int counter_index_first_child = getVectorIndexOfFirstChild(counter_index_parent);
    while (counter_index_parent < (size_t)n + compressed_counters) {
        int counter_index_last_child = counter_index_first_child + (branching_factor - 1) * depth;
        if (counter_index_last_child >= compressed_counters + counters.size()) {
            break;
        }
        long counter_index_parent_actual = counter_index_parent - compressed_counters;
        for (int index_child = counter_index_first_child; index_child <= counter_index_last_child; index_child += depth)
        {
            long counter_index_child_actual = index_child - compressed_counters;
            counters[counter_index_child_actual] += counters[counter_index_parent_actual];
        }
        compress_counter++;
        counter_index_parent++;
        counter_index_first_child = (counter_index_parent % depth == 0) ? counter_index_last_child + 1 : counter_index_first_child + 1;
    }
    counters.erase(counters.begin(), counters.begin() + compress_counter);
    compressed_counters += compress_counter;
    return compress_counter;
}

int GeometricSketchMH::expand(int n)
{
    counters.resize(counters.size() + n, 0);
    return n;
}

int GeometricSketchMH::shrink(int n)
{
    return undoExpand(n);
}

uint64_t GeometricSketchMH::getMemoryUsage() const
{
    return counters.size() * sizeof(uint32_t) + sizeof(unsigned) * 4;
}

uint64_t GeometricSketchMH::hash(uint32_t key, uint32_t row_id, uint32_t layer_id) const
{
    uint64_t seed = ((uint64_t)layer_id << 32) | row_id;
    return XXH64(&key, sizeof(key), seed);
}

int GeometricSketchMH::rowIndexToLayerId(int row_index, int& layer_index) const
{
    const int B = branching_factor;
    layer_index = row_index;
    int layer_width = width;
    int layer_id = 0;
    while (layer_index >= layer_width) {
        layer_index -= layer_width;
        layer_width *= B;
        layer_id++;
    }
    return layer_id;
}

int GeometricSketchMH::getVectorIndexOfParent(int vector_index) const
{
    const int B = (int)branching_factor;
    const int W = (int)width;
    int counter_row_index, counter_layer_index;
    const int row_id = vectorIndexToRowId(vector_index, counter_row_index);
    const int counter_layer_id = rowIndexToLayerId(counter_row_index, counter_layer_index);
    if (counter_layer_id <= 0) {
        return -1;
    }
    const int parent_layer_index = counter_layer_index / B;
    const int parent_row_index = parent_layer_index + getRowIndexOfLayer(counter_layer_id - 1);
    return rowIndexToVectorIndex(row_id, parent_row_index);
}

int GeometricSketchMH::rowIndexToVectorIndex(int row_id, int row_index) const {
    return row_index * depth + row_id;
}

int GeometricSketchMH::getVectorIndexOfFirstChild(int vector_index) const
{
    const int B = (int)branching_factor;
    const int W = (int)width;
    int counter_row_index, counter_layer_index;
    const int row_id = vectorIndexToRowId(vector_index, counter_row_index);
    const int counter_layer_id = rowIndexToLayerId(counter_row_index, counter_layer_index);
    const int child_layer_index = counter_layer_index * B;
    const int child_row_index = child_layer_index + getRowIndexOfLayer(counter_layer_id + 1);
    return rowIndexToVectorIndex(row_id, child_row_index);
}

int GeometricSketchMH::vectorIndexToRowId(int vector_index, int& row_index) const {
    row_index = vector_index / depth;
    const int row_id = vector_index % depth;
    return row_id;
}

int GeometricSketchMH::getRowIndexOfLayer(int layer_id) const
{
    const int L = layer_id;
    const int B = (int)branching_factor;
    const int W = (int)width;
    int B_raised_L = 1;
    // better than pow for our range of values - checked myself
    for (int i = 0; i < L; i++) B_raised_L *= B;
    return W * (1 - B_raised_L) / (1 - B);
}