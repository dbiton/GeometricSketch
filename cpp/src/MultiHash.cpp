#pragma once

#include "MultiHash.h"

MultiHash::MultiHash() :
    consumed_bits(-1),
    seed(-1),
    hash(-1),
    key(-1),
    subhash_mask(-1),
    bits_per_subhash(-1),
    subhash_modulus(-1),
    first_subhash_mask(-1),
    bits_per_first_subhash(-1),
    first_subhash_modulus(-1)
{
}

void MultiHash::setSubHashModulus(uint64_t _subhash_modulus) {
    bits_per_subhash = std::ceil(std::log2(_subhash_modulus));
    subhash_modulus = _subhash_modulus;
    subhash_mask = (1ULL << bits_per_subhash) - 1ULL;
}

void MultiHash::setFirstSubHashModulus(uint64_t _first_subhash_modulus) {
    bits_per_first_subhash =  std::ceil(std::log2(_first_subhash_modulus));
    first_subhash_modulus = _first_subhash_modulus;
    first_subhash_mask = (1ULL << bits_per_first_subhash) - 1ULL;
}

void MultiHash::initialize(uint64_t _key, uint64_t _seed) {
    consumed_bits = 128; // to make sure hash is called on next
    key = _key;
    seed = _seed;
}

uint64_t MultiHash::first() {
    hash = XXH64(&key, sizeof(uint64_t), seed++);
    uint64_t result = (hash & first_subhash_mask) % first_subhash_modulus;
    hash = hash >> bits_per_first_subhash;
    consumed_bits = bits_per_first_subhash;
    return result;
}

uint64_t MultiHash::next() {
    if (consumed_bits + bits_per_subhash >= 64) {
        hash = XXH64(&key, sizeof(uint64_t), seed++);
        consumed_bits = 0;
    }
    uint64_t result = (hash & subhash_mask) % subhash_modulus;
    hash = hash >> bits_per_subhash;
    consumed_bits += bits_per_subhash;
    return result;
}
