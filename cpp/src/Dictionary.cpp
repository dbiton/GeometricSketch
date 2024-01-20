#include "Dictionary.h"
#include "param.h"

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

int Dictionary::getMemoryUsage() const {
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

int CountMinDictionary::getMemoryUsage() const
{
    return CM_Size(this->count_min);
}

ElasticDictionary::ElasticDictionary(const int _bucket_num, const int _total_memory_in_bytes, int seed) : bucket_num(_bucket_num),
                                                                                    total_memory_in_bytes(_total_memory_in_bytes),
                                                                                    elastic_sketch(nullptr)
{
    elastic_sketch = new ElasticSketch(bucket_num, total_memory_in_bytes, seed);
}

ElasticDictionary::~ElasticDictionary() {}

void ElasticDictionary::update(uint32_t key, int amount)
{
    auto p = static_cast<ElasticSketch*>(elastic_sketch);
    p->insert((uint8_t *)&key, amount);
}
int ElasticDictionary::query(uint32_t key)
{
    auto p = static_cast<ElasticSketch*>(elastic_sketch);
    return p->query((uint8_t *)&key);
}


int ElasticDictionary::shrink(int ratio)
{
    auto p = static_cast<ElasticSketch*>(elastic_sketch);
    p->compress_self(ratio);
    return ratio;
}

int ElasticDictionary::getMemoryUsage() const {
    return total_memory_in_bytes;
}
