#pragma once
#include "param.h"

struct HeavyPart
{
    Bucket* buckets;
    int bucket_num;
public:
    HeavyPart(int bucket_num);
    ~HeavyPart();

    void clear();

    int insert(uint8_t *key, uint8_t *swap_key, uint32_t &swap_val, uint32_t f = 1);
    int quick_insert(uint8_t *key, uint32_t f = 1);

    int query(uint8_t *key);

    int get_memory_usage();
    int get_bucket_num();
private:
    int CalculateFP(uint8_t *key, uint32_t &fp);
};
