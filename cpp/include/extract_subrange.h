#include <algorithm>
#include <assert.h>

// return begin and end of bucket subrange such that the total sum
// of the buckets is as close to B/2 from above, and the range size is minimal
// assumes bucket width is constant
std::pair<int,int> extract_subrange(const std::vector<int>& buckets){
    int total = 0;
    for (auto v : buckets) total += v;
    int target = total / 2;
    
    auto best_range = std::make_pair(0, buckets.size());
    int best_sum = total;

    for (int i=0; i<buckets.size(); i++){
        int current_sum = 0;
        for (int j=i; j<buckets.size(); j++){
            current_sum += buckets[j];
            if (current_sum >= target){
                int current_delta = current_sum - target;
                int best_delta = best_sum - target;
                int current_range_length = j-i+1;
                int best_range_length = best_range.second - best_range.first;
                if (current_delta < best_delta || 
                    (current_delta == best_delta && current_range_length < best_range_length)){
                        best_range = std::make_pair(i, j+1);
                        best_sum = current_sum;
                }
                break;
            }
        }
    }

    return best_range;
}

void test_extract_subrange(){
    std::pair<int,int> p;
    p = extract_subrange(std::vector<int>({0,0,10,0}));
    assert(p == std::make_pair(2,3));
    p = extract_subrange(std::vector<int>({1,5,100,5,1}));
    assert(p == std::make_pair(2,3));
    p = extract_subrange(std::vector<int>({0,0,0,0,0}));
    assert(p == std::make_pair(0,1));
    p = extract_subrange(std::vector<int>({1,1,1,1}));
    assert(p == std::make_pair(0,2));
    p = extract_subrange(std::vector<int>({2,0,0,1,1}));
    assert(p == std::make_pair(0,1));
    p = extract_subrange(std::vector<int>({1,1,0,2,0,0}));
    assert(p == std::make_pair(3,4));
}
