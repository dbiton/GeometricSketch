#pragma once

#include <stdint.h>
#include <map>
#include <unordered_map>

#include "countmin.h"
#include "CountSketch.h"
#include "ElasticSketch.h"

class Dictionary
{
public:
    Dictionary();
    virtual ~Dictionary();

    virtual void update(uint32_t key, int amount) = 0;
    virtual int query(uint32_t key) = 0;

    virtual void expand(int bytes) = 0;
    virtual void shrink(int bytes) = 0;
    virtual int getSize() const = 0;
    virtual int getMemoryUsage() const = 0;
};

class ElasticDictionary: public Dictionary
{
    void* elastic_sketch;
    const int bucket_num;
    const int total_memory_in_bytes;
public:
    ElasticDictionary(int bucket_num, int total_memory_in_bytes, int seed);
    ~ElasticDictionary();

    void update(uint32_t key, int amount);
    int query(uint32_t key);

    void expand(int bytes);
    void shrink(int ratio);
    int getSize() const;
    int getMemoryUsage() const;
};

class CountMinDictionary : public Dictionary
{
    CM_type *count_min;

public:
    CountMinDictionary(int width, int depth, int seed);
    ~CountMinDictionary();

    void update(uint32_t key, int amount);
    int query(uint32_t key);

    void expand(int bytes);
    void shrink(int bytes);
    int getSize() const;
    int getMemoryUsage() const; // minimum
};
