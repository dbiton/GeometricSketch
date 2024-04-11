#include "Dictionary.h"

Dictionary::Dictionary() {}

int Dictionary::compress(int bytes){
    throw std::runtime_error("Dictionary::compress - not implemented.");
}

int Dictionary::expand(int bytes)
{
    throw std::runtime_error("Dictionary::expand - not implemented.");
}

int Dictionary::shrink(int bytes)
{
    throw std::runtime_error("Dictionary::shrink - not implemented.");
}

int Dictionary::getSize() const
{
    throw std::runtime_error("Dictionary::getSize - not implemented.");
}

uint64_t Dictionary::getMemoryUsage() const {
    throw std::runtime_error("Dictionary::getMemoryUsage - not implemented.");
}

Dictionary::~Dictionary() {}

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

uint64_t CountMinDictionary::getMemoryUsage() const
{
    return CM_Size(this->count_min);
}