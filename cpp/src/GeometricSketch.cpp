#include "GeometricSketch.h"
#include "xxhash.h"
#include "doctest.h"

GeometricSketch::GeometricSketch(int width, int depth, int branching_factor) : 
    Dictionary(),
    width(width),
    depth(depth),
    branching_factor(branching_factor),
    compressed_counters(0),
    counters(width * depth, 0)
{
}

GeometricSketch::~GeometricSketch()
{
}

void GeometricSketch::update(uint32_t key, int amount)
{
    for (uint32_t row_id = 0; row_id < depth; row_id++)
    {
        long vector_index = getLastVectorIndexFromKey(key, row_id);
        long actual_index = vector_index - compressed_counters;
        counters[actual_index] += amount;
    }
}

int GeometricSketch::getLastVectorIndexFromKey(
    uint32_t key,
    uint32_t row_id
) const {
    int prev_layer_id = 0;
    int prev_layer_begin_counter_index = 0;
    int prev_row_index = hash(key, row_id, 0) % width;
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

int GeometricSketch::getNextVectorIndexFromKey(
    uint32_t key, 
    uint32_t row_id, 
    int& prev_layer_id, 
    int& prev_layer_row_index,
    int& prev_counter_row_index,
    int& prev_B_pow
) const{
    const int B = (int)branching_factor;
    const int W = (int)width;
    const int layer_begin_counter_index = prev_layer_row_index + W * prev_B_pow;
    const int counter_index_first_child = (prev_counter_row_index - prev_layer_row_index) * B + layer_begin_counter_index;
    if (rowIndexToVectorIndex(row_id, counter_index_first_child) >= counters.size() + compressed_counters) {
        return -1;
    }
    const int h = hash(key, row_id, prev_layer_id + 1) % B;
    const int counter_index = counter_index_first_child + h;
    prev_layer_id += 1;
    prev_layer_row_index = layer_begin_counter_index;
    prev_counter_row_index = counter_index;
    prev_B_pow *= B;
    const int vector_index = rowIndexToVectorIndex(row_id, counter_index);
    if (vector_index >= counters.size() + compressed_counters) {
        return -1;
    }
    return vector_index;
}

int GeometricSketch::query(uint32_t key)
{
    int O = compressed_counters;
    uint32_t estimate = UINT32_MAX;
    for (uint32_t row_id = 0; row_id < depth; row_id++)
    {
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

int GeometricSketch::undoExpand(int n)
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

int GeometricSketch::compress(int n)
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

int GeometricSketch::expand(int n)
{
    counters.resize(counters.size() + n, 0);
    return n;
}

int GeometricSketch::shrink(int n)
{
    return undoExpand(n);
}

uint64_t GeometricSketch::getMemoryUsage() const
{
    return counters.size() * sizeof(uint32_t) + sizeof(unsigned) * 4;
}

uint64_t GeometricSketch::hash(uint32_t key, uint32_t row_id, uint32_t layer_id) const
{
    uint64_t seed = ((uint64_t)layer_id << 32) | row_id;
    return XXH64(&key, sizeof(key), seed);
}

int GeometricSketch::rowIndexToLayerId(int row_index, int& layer_index) const
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

int GeometricSketch::getVectorIndexOfParent(int vector_index) const
{
    const int B = (int)branching_factor;
    const int W = (int)width;
    int counter_row_index, counter_layer_index;
    const int row_id = vectorIndexToRowId(vector_index, counter_row_index);
    const int counter_layer_id = rowIndexToLayerId(counter_row_index, counter_layer_index);
    if (counter_layer_id <= 0){
        return -1;
    }
    const int parent_layer_index = counter_layer_index / B;
    const int parent_row_index = parent_layer_index + getRowIndexOfLayer(counter_layer_id - 1);
    return rowIndexToVectorIndex(row_id, parent_row_index);
}

int GeometricSketch::rowIndexToVectorIndex(int row_id, int row_index) const {
    return row_index * depth + row_id;
}

int GeometricSketch::getVectorIndexOfFirstChild(int vector_index) const
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

int GeometricSketch::vectorIndexToRowId(int vector_index, int& row_index) const {
    row_index = vector_index / depth;
    const int row_id = vector_index % depth;
    return row_id;
}

int GeometricSketch::getRowIndexOfLayer(int layer_id) const
{
    const int L = layer_id;
    const int B = (int)branching_factor;
    const int W = (int)width;
    int B_raised_L = 1;
    // better than pow for our range of values - checked myself
    for (int i = 0; i < L; i++) B_raised_L *= B;
    return W * (1 - B_raised_L) / (1 - B);
}


TEST_SUITE("GeometricSketch Helper Methods") {
    constexpr auto W = 10;
    constexpr auto D = 2;
    constexpr auto B = 2;
    GeometricSketch gs(W, D, B);

    /* The row index (index inside row) of the first counter of layer l in row r */
    TEST_CASE("getRowIndexOfLayer") {
        REQUIRE(gs.getRowIndexOfLayer(0) == 0);
        REQUIRE(gs.getRowIndexOfLayer(1) == W);
        REQUIRE(gs.getRowIndexOfLayer(2) == W + W * B);
    }

    /* The row id of the row the i-th counter in counters belongs to */
    TEST_CASE("vectorIndexToRowId") {
        int row_index;
        REQUIRE(gs.vectorIndexToRowId(0, row_index) == 0);
        REQUIRE(row_index == 0);
        REQUIRE(gs.vectorIndexToRowId(1 + D, row_index) == 1);
        REQUIRE(row_index == 1);
        REQUIRE(gs.vectorIndexToRowId(W * D * B, row_index) == 0);
        REQUIRE(row_index == W * B);
    }

    /* Returns the first child of the i-th counter in counters */
    TEST_CASE("getVectorIndexOfFirstChild") {
        REQUIRE(gs.getVectorIndexOfFirstChild(0) == W * D);
        REQUIRE(gs.getVectorIndexOfFirstChild(D - 1) == W * D + D - 1);
        REQUIRE(gs.getVectorIndexOfFirstChild(W * D) == W * D * (1 + B));
    }

    /* Given the row id and the offset of a counter in it (row index), return the vector index of the counter*/
    TEST_CASE("rowIndexToVectorIndex") {
        REQUIRE(gs.rowIndexToVectorIndex(0, 0) == 0);
        REQUIRE(gs.rowIndexToVectorIndex(0, 1) == D);
        REQUIRE(gs.rowIndexToVectorIndex(1, 0) == 1);
        REQUIRE(gs.rowIndexToVectorIndex(D - 1, W * (1 + B)) == W * (1 + B) * D + D - 1);
    }

    /* Given index of a counter in a row, return its layer index and layer id*/
    TEST_CASE("rowIndexToLayerIndex") {
        int layer_index;
        REQUIRE(gs.rowIndexToLayerId(0, layer_index) == 0);
        REQUIRE(layer_index == 0);
        REQUIRE(gs.rowIndexToLayerId(W, layer_index) == 1);
        REQUIRE(layer_index == 0);
        REQUIRE(gs.rowIndexToLayerId(W * (1 + B) + W, layer_index) == 2);
        REQUIRE(layer_index == W);
    }

    TEST_CASE("getVectorIndexOfParent") {
        REQUIRE(gs.getVectorIndexOfParent(W * D) == 0);
        REQUIRE(gs.getVectorIndexOfParent(W * D + 1) == 1);
        REQUIRE(gs.getVectorIndexOfParent(W * D * (1 + B) + D) == W * D);
    }

    TEST_CASE("getNextVectorIndexFromKey") {
        constexpr int KEY = 0;
        constexpr int ROW_ID = 0;

        uint64_t hash_0 = gs.hash(KEY, ROW_ID, 0) % W;
        uint64_t hash_1 = gs.hash(KEY, ROW_ID, 1) % B;
        uint64_t hash_2 = gs.hash(KEY, ROW_ID, 2) % B;
        uint64_t hash_3 = gs.hash(KEY, ROW_ID, 3) % B;

        int expected_vector_index_1 = gs.rowIndexToVectorIndex(ROW_ID, W + hash_0 * B + hash_1);
        int expected_row_index_2 = W * (1 + B) + hash_0 * B * B + hash_1 * B + hash_2;
        int expected_vector_index_2 = gs.rowIndexToVectorIndex(ROW_ID, expected_row_index_2);
        int expected_row_index_3 = W * (1 + B + B * B) + hash_0 * B * B * B + hash_1 * B * B + hash_2 * B + hash_3;
        int expected_vector_index_3 = gs.rowIndexToVectorIndex(ROW_ID, expected_row_index_3);

        int prev_layer_id = 0;
        int prev_layer_begin_counter_index = 0;
        int prev_row_index = gs.hash(KEY, ROW_ID, 0) % W;
        int prev_B_pow = 1;

        // so get vector index doesn't return -1
        int large_expand = 10000;
        gs.expand(large_expand);

        int actual_vector_index_1 = gs.getNextVectorIndexFromKey(KEY, ROW_ID, prev_layer_id,
            prev_layer_begin_counter_index, prev_row_index, prev_B_pow);
        int actual_vector_index_2 = gs.getNextVectorIndexFromKey(KEY, ROW_ID, prev_layer_id,
            prev_layer_begin_counter_index, prev_row_index, prev_B_pow);
        int actual_vector_index_3 = gs.getNextVectorIndexFromKey(KEY, ROW_ID, prev_layer_id,
            prev_layer_begin_counter_index, prev_row_index, prev_B_pow);

        gs.undoExpand(large_expand);

        REQUIRE(actual_vector_index_1 == expected_vector_index_1);
        REQUIRE(actual_vector_index_2 == expected_vector_index_2);
        REQUIRE(actual_vector_index_3 == expected_vector_index_3);
    }
}

TEST_SUITE("GeometricSketch Core Methods") {
    constexpr auto W = 374;
    constexpr auto D = 5;
    constexpr auto B = 3;

    TEST_CASE("initialize") {
        GeometricSketch gs(W, D, B);
        REQUIRE(gs.compressed_counters == 0);
        REQUIRE(gs.branching_factor == B);
        REQUIRE(gs.width == W);
        REQUIRE(gs.depth == D);
        REQUIRE(gs.counters == std::vector<uint32_t>(W*D, 0));
    }

    TEST_CASE("query") {
        GeometricSketch gs(W, D, B);

        constexpr auto QUERY_COUNT = 5708;
        int MODIFY_SIZE = 10;

        std::vector<int> actual_results(QUERY_COUNT, 0);

        for (int i = 0; i < QUERY_COUNT; i++) {
            gs.update(i, 1);
            actual_results[i] = gs.query(i);
            gs.update(i, -1);

            int action = XXH64(&i, sizeof(i), 0) % 3;
            switch (action) {
                case 0:
                    gs.compress(MODIFY_SIZE);
                    break;
                case 1:
                    gs.expand(MODIFY_SIZE);
                    break;
                case 2:
                    gs.undoExpand(MODIFY_SIZE);
                    break;
            }
        }

        std::vector<int> expected_results(QUERY_COUNT, 1);
        REQUIRE(actual_results == expected_results);
    }

    TEST_CASE("undoExpand") {
        constexpr auto EXPAND_SIZE = 521;
        constexpr auto UPDATE_COUNT_0 = 957;
        constexpr auto UPDATE_COUNT_1 = 589;
        constexpr auto UPDATE_COUNT_2 = 587;

        uint64_t key = 0;
        GeometricSketch regular_gs(W, D, B);
        GeometricSketch undone_gs(W, D, B);

        for (int i = 0; i < UPDATE_COUNT_0; i++) {
            key = XXH64(&key, sizeof(key), 0);
            regular_gs.update(key, 1);
            undone_gs.update(key, 1);
        }

        undone_gs.expand(EXPAND_SIZE);

        for (int i = 0; i < UPDATE_COUNT_1; i++) {
            key = XXH64(&key, sizeof(key), 0);
            regular_gs.update(key, 1);
            undone_gs.update(key, 1);
        }

        undone_gs.undoExpand(EXPAND_SIZE);

        for (int i = 0; i < UPDATE_COUNT_2; i++) {
            key = XXH64(&key, sizeof(key), 0);
            regular_gs.update(key, 1);
            undone_gs.update(key, 1);
        }

        REQUIRE(regular_gs.compressed_counters == undone_gs.compressed_counters);
        REQUIRE(regular_gs.branching_factor == undone_gs.branching_factor);
        REQUIRE(regular_gs.width == undone_gs.width);
        REQUIRE(regular_gs.depth == undone_gs.depth);
        REQUIRE(regular_gs.counters == undone_gs.counters);
    }

    TEST_CASE("compress") {
        constexpr auto EXPAND_SIZE = 4252;
        constexpr auto UPDATE_COUNT = 6682;

        uint64_t key;
        GeometricSketch gs(W, D, B);
        std::vector<int> expected_queries(UPDATE_COUNT, 0);
        std::vector<int> actual_queries(UPDATE_COUNT, 0);


        gs.expand(EXPAND_SIZE);

        key = 0;
        for (int i = 0; i < UPDATE_COUNT; i++) {
            key = XXH64(&key, sizeof(key), 0);
            gs.update(key, 1);
        }

        key = 0;
        for (int i = 0; i < UPDATE_COUNT; i++) {
            expected_queries[i] = gs.query(key);
        }

        gs.compress(EXPAND_SIZE);

        key = 0;
        for (int i = 0; i < UPDATE_COUNT; i++) {
            actual_queries[i] = gs.query(key);
        }

        REQUIRE(actual_queries == expected_queries);
    }
}