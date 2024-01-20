#include "LightPart.h"
#include <cassert>


LightPart::LightPart(int init_mem_in_bytes, int seed):
    counter_num(init_mem_in_bytes),
	counters((uint8_t*)malloc(counter_num * sizeof(uint8_t)))
{
    this->clear();
    bobhash = new BOBHash32(seed % MAX_PRIME32);
}


LightPart::~LightPart()
{
    delete bobhash;
    delete counters;
}


void LightPart::clear()
{
    memset(counters, 0, counter_num);
    memset(mice_dist, 0, sizeof(int) * 256);
}

void LightPart::merge(const LightPart& o){
    if (counter_num != o.counter_num){
        throw std::runtime_error("mismatching light parts!");
    }
    for (int i=0; i<counter_num; i++){
        int old_val = (int)counters[i];
        int new_val = old_val + (int)o.counters[i];
        new_val = new_val < 255 ? new_val : 255;
        counters[i] = (uint8_t)new_val;
    }
}

void LightPart::insert(uint8_t *key, int f)
{
    uint32_t hash_val = (uint32_t)bobhash->run((const char*)key, KEY_LENGTH_4);
    uint32_t pos = hash_val % (uint32_t)counter_num;

    /* insert */
    int old_val = (int)counters[pos];
    int new_val = (int)counters[pos] + f;

    new_val = new_val < 255 ? new_val : 255;
    counters[pos] = (uint8_t)new_val;

    mice_dist[old_val]--;
    mice_dist[new_val]++;
}


void LightPart::swap_insert(uint8_t *key, int f)
{
    uint32_t hash_val = (uint32_t)bobhash->run((const char*)key, KEY_LENGTH_4);
    uint32_t pos = hash_val % (uint32_t)counter_num;

    /* swap_insert */
    f = f < 255 ? f : 255;
    if (counters[pos] < f) 
    {
        int old_val = (int)counters[pos];
        counters[pos] = (uint8_t)f;
        int new_val = (int)counters[pos];

        mice_dist[old_val]--;
        mice_dist[new_val]++;
    }
}


int LightPart::query(uint8_t *key) 
{
    uint32_t hash_val = (uint32_t)bobhash->run((const char*)key, KEY_LENGTH_4);
    uint32_t pos = hash_val % (uint32_t)counter_num;

    return (int)counters[pos];
}

void LightPart::compress_self(int ratio)
{
    int width = get_compress_width(ratio);
    auto new_counters = ((uint8_t*)malloc(width * sizeof(uint8_t)));
    compress(ratio, new_counters);
    delete counters;
    counters = new_counters;
    counter_num = width;
}

void LightPart::compress(int ratio, uint8_t *dst) 
{
    int width = get_compress_width(ratio);

    for (int i = 0; i < width && i < counter_num; ++i) 
    {
        uint8_t max_val = 0;
        for (int j = i; j < counter_num; j += width)
                max_val = counters[j] > max_val ? counters[j] : max_val;
        dst[i] = max_val;
    }
}


int LightPart::query_compressed_part(uint8_t *key, uint8_t *compress_part, int compress_counter_num) 
{
    uint32_t hash_val = (uint32_t)bobhash->run((const char *)key, KEY_LENGTH_4);
    uint32_t pos = (hash_val % (uint32_t)counter_num) % compress_counter_num;

    return (int)compress_part[pos];
}


int LightPart::get_compress_width(int ratio)
{
    return (counter_num / ratio);
}


int LightPart::get_compress_memory(int ratio){
    return (uint32_t)(counter_num / ratio);
}


int LightPart::get_memory_usage()
{
    return counter_num;
}


int LightPart::get_cardinality() 
{
    int mice_card = 0;
    for (int i = 1; i < 256; i++)
        mice_card += mice_dist[i];

    double rate = (counter_num - mice_card) / (double)counter_num;
    return counter_num * log(1 / rate);
}


void LightPart::get_entropy(int &tot, double &entr)
{
    for (int i = 1; i < 256; i++) 
    {
        tot += mice_dist[i] * i;
        entr += mice_dist[i] * i * log2(i);
    }
}


void LightPart::get_distribution(vector<double> &dist) 
{
    assert(false);
    /*uint32_t tmp_counters[counter_num];
    for (int i = 0; i < counter_num; i++)
        tmp_counters[i] = counters[i];

    em_fsd_algo = new EMFSD();
    em_fsd_algo->set_counters(counter_num, tmp_counters);

    em_fsd_algo->next_epoch();
    em_fsd_algo->next_epoch();
    em_fsd_algo->next_epoch();
    em_fsd_algo->next_epoch();
    em_fsd_algo->next_epoch();
    em_fsd_algo->next_epoch();
    em_fsd_algo->next_epoch();
    em_fsd_algo->next_epoch();
    em_fsd_algo->next_epoch();
    em_fsd_algo->next_epoch();

    dist = em_fsd_algo->ns;*/
}


