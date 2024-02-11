#include <cstdint>
#include "Murmurhash.h"

class GeomtericSketchHasher{
    uint16_t row_index;
    uint32_t prev_layer_index;
    uint32_t prev_layer_begin_counter_index;
    uint32_t prev_counter_index;
    uint32_t prev_B_pow_layer_index;

    const uint32_t B;
    const uint32_t W;
    const uint32_t K;
    const uint32_t R;
    const uint32_t O;
public:
    GeomtericSketchHasher(uint32_t _B, uint32_t _W, uint32_t _K, uint32_t _R, uint32_t _O) : 
        B(_B),
        W(_W),
        K(_K),
        R(_R),
        O(_O)
    {
        prev_layer_index = 0;
        prev_layer_begin_counter_index = 0;
        prev_counter_index = hash(K, row_index, 0);
        prev_B_pow_layer_index = 1;
    }

    uint32_t next(){
        auto i = generate();
        // replace loop with calculation of first layer
        while (i < O){
            i = generate();
        } 
        return i > O + R ? -1 : i;
    }
private:
    uint32_t generate(){
        int layer_begin_counter_index = prev_layer_begin_counter_index + W * prev_B_pow_layer_index;
        int h = hash(K, row_index, prev_layer_index+1);
        int counter_index = (prev_counter_index - prev_layer_begin_counter_index) * B + h + layer_begin_counter_index;
        prev_layer_index += 1;
        prev_layer_begin_counter_index = layer_begin_counter_index;
        prev_counter_index = counter_index;
        prev_B_pow_layer_index *= B;
        return counter_index;
    }
    
    int hash(uint32_t key, uint16_t row_index, uint16_t layer_index) const
    {
        uint32_t seed = ((uint32_t)row_index << 16) + (uint32_t)layer_index;
        auto h = murmurhash((int *)&key, seed);
        // use 16 bit hash and replace % with something else
        return layer_index == 0 ? h % W : h % B;
    }

};