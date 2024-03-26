#pragma once

#include <cmath>
#include <cstdint>
#include "xxhash.h"

class MultiHash{
    int consumed_bits;
    uint64_t seed;
    uint64_t hash;
    uint64_t key;

    uint64_t subhash_mask;
    int bits_per_subhash;
    uint64_t subhash_modulus;

    uint64_t first_subhash_mask;
    int bits_per_first_subhash;
    uint64_t first_subhash_modulus;
public:
    MultiHash();

    void setSubHashModulus(uint64_t _subhash_modulus);

    void setFirstSubHashModulus(uint64_t _first_subhash_modulus);

    void initialize(uint64_t _key, uint64_t _seed);

    uint64_t first();

    uint64_t next();
};
