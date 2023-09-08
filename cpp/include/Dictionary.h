#pragma once

#include <stdint.h>
#include <map>
#include <unordered_map>

#include "countmin.h"
#include "CountSketch.h"

class Dictionary
{
public:
    Dictionary();

    virtual void update(uint32_t key, int amount);
    virtual int query(uint32_t key);

    virtual void expand();
    virtual void shrink();
    virtual int getSize() const;
    virtual int getMemoryUsage() const;
};

class CountSketchDictionary : public Dictionary
{
    CountSketch count_sketch;

public:
    CountSketchDictionary(double width, double depth);

    void update(uint32_t key, int amount);
    int query(uint32_t key);

    void expand();
    void shrink();
    int getSize() const;
    int getMemoryUsage() const; // minimum
};

class CountMinDictionary : public Dictionary
{
    CM_type *count_min;

public:
    CountMinDictionary(double width, double depth);

    void update(uint32_t key, int amount);
    int query(uint32_t key);

    void expand();
    void shrink();
    int getSize() const;
    int getMemoryUsage() const; // minimum
};

class UnorderedMapDictionary : public Dictionary
{
    std::unordered_map<uint32_t, int> unorderd_map;

public:
    UnorderedMapDictionary();

    void update(uint32_t key, int amount);
    int query(uint32_t key);

    void expand();
    void shrink();
    int getSize() const;
    int getMemoryUsage() const; // minimum
};

class MapDictionary : public Dictionary
{
    std::map<uint32_t, int> map;

public:
    MapDictionary();

    void update(uint32_t key, int amount);
    int query(uint32_t key);

    void expand();
    void shrink();
    int getSize() const;
    int getMemoryUsage() const; // minimum
};