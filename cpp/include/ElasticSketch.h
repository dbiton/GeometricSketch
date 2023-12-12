#ifndef _ELASTIC_SKETCH_H_
#define _ELASTIC_SKETCH_H_

#include "HeavyPart.h"
#include "LightPart.h"


class ElasticSketch
{
    const int heavy_mem;
    const int light_mem;
    int bucket_num;

    HeavyPart heavy_part;
    LightPart light_part;

public:
    ElasticSketch(int bucket_num, int tot_memory_in_bytes, int seed):
        bucket_num(bucket_num),
        heavy_mem(bucket_num * COUNTER_PER_BUCKET * 8),
        light_mem(tot_memory_in_bytes - heavy_mem),
        heavy_part(bucket_num),
        light_part(light_mem, seed)
    {
    }

    ~ElasticSketch(){}
    void clear();

    int get_memory_usage() const{
        return heavy_mem + light_mem;
    }

    void insert(uint8_t *key, int f = 1);
    void quick_insert(uint8_t *key, int f = 1);

    int query(uint8_t *key);
    int query_compressed_part(uint8_t *key, uint8_t *compress_part, int compress_counter_num);

    int get_compress_width(int ratio) { return light_part.get_compress_width(ratio);}
    void compress(int ratio, uint8_t *dst) {    light_part.compress(ratio, dst); }
    void compress_self(int ratio){ light_part.compress_self(ratio); }

    int get_bucket_num() { return heavy_part.get_bucket_num(); }
    double get_bandwidth(int compress_ratio) ;

    void get_heavy_hitters(int threshold, vector<pair<string, int>> & results);
    int get_cardinality();
    double get_entropy();
    void get_distribution(vector<double> &dist);

    void *operator new(size_t sz);
    void operator delete(void *p);
};

#endif // _ELASTIC_SKETCH_H_
