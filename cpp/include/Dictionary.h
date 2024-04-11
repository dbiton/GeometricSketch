#pragma once

#include <stdint.h>
#include <map>
#include <unordered_map>

#include "countmin.h"

class Dictionary
{
public:
    Dictionary();
    virtual ~Dictionary();

    virtual void update(uint32_t key, int amount) = 0;
    virtual int query(uint32_t key) = 0;

    virtual int compress(int bytes);
    virtual int expand(int bytes);
    virtual int shrink(int bytes);
    virtual int getSize() const;
    virtual uint64_t getMemoryUsage() const;
};

class CountMinDictionary : public Dictionary
{
    CM_type *count_min;

public:
    CountMinDictionary(int width, int depth, int seed);
    ~CountMinDictionary();

    void update(uint32_t key, int amount);
    int query(uint32_t key);

    uint64_t getMemoryUsage() const; // minimum
};
