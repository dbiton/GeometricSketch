#include "Dictionary.h"

Dictionary::Dictionary() {}

Dictionary::~Dictionary() {}

UnorderedMapDictionary::UnorderedMapDictionary() : Dictionary()
{
}

UnorderedMapDictionary::~UnorderedMapDictionary() {}

void UnorderedMapDictionary::update(uint32_t key, int amount)
{
    this->unordered_map[key] += amount;
}

int UnorderedMapDictionary::query(uint32_t key)
{
    return this->unordered_map[key];
}

void UnorderedMapDictionary::expand()
{
    throw std::runtime_error("UnorderedMapDictionary::expand - should not be used.");
}

void UnorderedMapDictionary::shrink()
{
    throw std::runtime_error("UnorderedMapDictionary::shrink - should not be used.");
}

int UnorderedMapDictionary::getSize() const
{
    throw std::runtime_error("UnorderedMapDictionary::getSize - should not be used.");
}

int UnorderedMapDictionary::getMemoryUsage() const
{
    int int_usage = sizeof(uint32_t);
    int row_usage = int_usage * 2;
    return this->unordered_map.size() * row_usage;
}

MapDictionary::MapDictionary() : Dictionary()
{
}

MapDictionary::~MapDictionary() {}


void MapDictionary::update(uint32_t key, int amount)
{
    this->map[key] += amount;
}
int MapDictionary::query(uint32_t key)
{
    return this->map[key];
}

void MapDictionary::expand()
{
    throw std::runtime_error("MapDictionary::expand - should not be used.");
}

void MapDictionary::shrink()
{
    throw std::runtime_error("MapDictionary::shrink - should not be used.");
}
int MapDictionary::getSize() const
{
    throw std::runtime_error("MapDictionary::getSize - should not be used.");
}

int MapDictionary::getMemoryUsage() const
{
    int pointer_usage = 4; // assume 32 bit computer
    int int_usage = sizeof(uint32_t);
    int node_usage = pointer_usage + int_usage * 2;
    return this->map.size() * node_usage;
}

CountSketchDictionary::CountSketchDictionary(int width, int depth) : Dictionary(), count_sketch((unsigned int)width, (unsigned int)depth)
{
}

CountSketchDictionary::~CountSketchDictionary() {}

void CountSketchDictionary::update(uint32_t key, int amount)
{
    if (amount != 1)
    {
        throw std::runtime_error("CountSketchDictionary::update - amount must be 1.");
    }
    count_sketch.addInt(key);
}
int CountSketchDictionary::query(uint32_t key)
{
    return this->count_sketch.getIntFrequency(key);
}

void CountSketchDictionary::expand()
{
    throw std::runtime_error("CountSketchDictionary::expand - should not be used.");
}

void CountSketchDictionary::shrink()
{
    throw std::runtime_error("CountSketchDictionary::shrink - should not be used.");
}
int CountSketchDictionary::getSize() const
{
    throw std::runtime_error("CountSketchDictionary::getSize - should not be used.");
}

int CountSketchDictionary::getMemoryUsage() const
{
    // TODO: implement method
    throw std::runtime_error("CountSketchDictionary::getMemoryUsage - not implemented yet.");
}

CountMinDictionary::CountMinDictionary(int width, int depth, int seed) : Dictionary()
{
    this->count_min = CM_Init(width, depth, seed);
}

CountMinDictionary::~CountMinDictionary() {}

void CountMinDictionary::update(uint32_t key, int amount)
{

    CM_Update(this->count_min, key, amount);
}
int CountMinDictionary::query(uint32_t key)
{
    return CM_PointEst(this->count_min, key);
}

void CountMinDictionary::expand()
{
    throw std::runtime_error("CountMinDictionary::expand - should not be used.");
}

void CountMinDictionary::shrink()
{
    throw std::runtime_error("CountMinDictionary::shrink - should not be used.");
}
int CountMinDictionary::getSize() const
{
    throw std::runtime_error("CountMinDictionary::getSize - should not be used.");
}

int CountMinDictionary::getMemoryUsage() const
{
    return CM_Size(this->count_min);
}
