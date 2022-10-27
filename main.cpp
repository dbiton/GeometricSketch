 #include <unordered_map>
#include <time.h>
#include <chrono>

#include "DynamicSketch.h"
#include "NaiveSketch.h"
#include "ElasticSketch.h"

constexpr float epsilon = 0.1;
constexpr float delta = 0.1;
constexpr int SEED = 0x1337C0D3;
constexpr int NUM_PACKETS = 1024*1024*32;
const int CM_WIDTH = ceil(exp(1) / epsilon);
const int CM_DEPTH = ceil(log(1 / delta));

typedef int (*QueryFunc)(uint32_t);

template<int N, int B>
double meanAbsError(ElasticSketch<N,B>& elastic_sketch, std::unordered_map<uint32_t, int>& counts) {
	double mae = 0;
	for (auto p : counts) {
		auto packet = p.first;
		mae += std::abs(p.second - elastic_sketch.query((uint8_t*)&packet));
	}
	mae /= counts.size();
	return mae;
}

double meanAbsError(DynamicSketch& dynamic_sketch, std::unordered_map<uint32_t, int>& counts) {
	double mae = 0;
	for (auto p : counts) {
		mae += std::abs(p.second - dynamic_sketch.query(p.first));
	}
	mae /= counts.size();
	return mae;
}

double parseCapture(std::string capture_path, std::vector<uint32_t>* src_addrs) {
	double start, end, time_diff;
	start = clock();
	std::ifstream myfile(capture_path);
	if (!myfile.is_open()) exit(1);
	uint32_t ipaddr;
	while (myfile >> ipaddr) {
		if (src_addrs->size() == NUM_PACKETS) break;
		src_addrs->push_back(ipaddr);
	}
	end = clock();
	time_diff = (end - start) / CLOCKS_PER_SEC;
	return time_diff;
}

double calc_counts(const std::vector<uint32_t>& indice, std::unordered_map<uint32_t, int>& counts) {
	double start, end, time_diff;
	CM_type* count_min = CM_Init(CM_WIDTH, CM_DEPTH, SEED);
	start = clock();
	for (const auto& index : indice) {
		if (counts.find(index) == counts.end()) {
			counts[index] = 1;
		}
		else {
			counts[index]++;
		}
	}
	end = clock();
	time_diff = (end - start) / CLOCKS_PER_SEC;
	return time_diff;
}

double test_unordered_map_time(const std::vector<uint32_t>& indice, std::unordered_map<uint32_t, uint32_t>& m) {
	auto t0 = std::chrono::high_resolution_clock::now();
	for (auto i : indice) {
		auto it = m.find(i);
		if (it == m.end()) m[i] = 0;
		else (*it).second++;
	}
	auto t1 = std::chrono::high_resolution_clock::now();
	std::chrono::duration<double, std::milli> dt = t1 - t0;
	return dt.count() / 1000;
}

int minimum_binary_tree_size(int elements_count) {
	return (sizeof(void*) * 2 + sizeof(uint32_t)) * elements_count;
}

int minimum_hash_map_size(int elements_count, float load_factor) {
	return sizeof(uint32_t) * elements_count / load_factor;
}

double test_map_time(const std::vector<uint32_t>& indice, std::map<uint32_t, uint32_t>& m) {
	auto t0 = std::chrono::high_resolution_clock::now();
	for (auto i : indice) {
		auto it = m.find(i);
		if (it == m.end()) m[i] = 0;
		else (*it).second++;
	}
	auto t1 = std::chrono::high_resolution_clock::now();
	std::chrono::duration<double, std::milli> dt = t1 - t0;
	return dt.count() / 1000;
}

double test_memory_usage(const std::vector<uint32_t>& indice) {
	constexpr int size_factor = 1;
	auto t0 = std::chrono::high_resolution_clock::now();
	std::map<uint32_t, uint32_t> m;

	std::ofstream log("memory_usage.dat", std::ios::trunc);

	DynamicSketch sketch_hash(CM_WIDTH*128, CM_DEPTH, SEED);
	DynamicSketch sketch_tree(CM_WIDTH*128, CM_DEPTH, SEED);

	constexpr int updates_per_log = 1024 * 4;

	for (int i = 0; i < indice.size(); i++) {
		auto packet = indice[i];

		// sized a tenth of the size of a tree/hash
		int size_hash = minimum_hash_map_size(m.size(), 0.75);
		int size_tree = minimum_binary_tree_size(m.size());
		while (sketch_hash.byteSize() * size_factor < size_hash) {
			sketch_hash.expand();
		}
		while (sketch_tree.byteSize() * size_factor < size_tree) {
			sketch_tree.expand();
		}

		if (i % updates_per_log == 0 && i > 1) {
			double mse_hash = 0;
			for (auto p : m) {
				int actual = p.second;
				int estimate = sketch_hash.query(p.first);
				mse_hash += (actual - estimate) * (actual - estimate);
			}
			mse_hash /= m.size();

			double mse_tree = 0;
			for (auto p : m) {
				mse_tree += std::pow((float)p.second - sketch_tree.query(p.first), 2);
			}
			mse_tree /= m.size();

			log << size_hash << " " << size_tree << " " << sketch_hash.byteSize() << " "
				<< sketch_tree.byteSize() << " " << mse_hash << " " << mse_tree << " " << i << std::endl;
		}
		m[packet]++;
		sketch_hash.update(packet, 1);
		sketch_tree.update(packet, 1);
	}
	log.close();
	auto t1 = std::chrono::high_resolution_clock::now();
	std::chrono::duration<double, std::milli> dt = t1 - t0;
	return dt.count() / 1000;
}

int test_naive(const std::vector<uint32_t>& indice) {
	auto start = clock();
	constexpr int expand_count = 64;
	const int updates_per_expand = indice.size() / expand_count;
	std::ofstream log("naive.dat", std::ios::trunc);

	std::unordered_map<uint32_t, int> counts;
	
	DynamicSketch dynamic_sketch(CM_WIDTH, CM_DEPTH, SEED);
	NaiveSketch naive_sketch(CM_WIDTH, CM_DEPTH, SEED);
	for (int i = 1; i < indice.size(); i++) {
		auto packet = indice[i];
		if (i % updates_per_expand == 0 && i > 1) {
			double mae = 0;
			for (auto p : counts) {
				mae += std::abs(p.second - dynamic_sketch.query(p.first));
			}
			mae /= counts.size();
			log << dynamic_sketch.sketchCount() << " " << mae << std::endl;

			mae = 0;
			for (auto p : counts) {
				mae += std::abs(p.second - naive_sketch.query(p.first));
			}
			mae /= counts.size();
			log << naive_sketch.sketchCount() << " " << mae << std::endl;

			naive_sketch.expand();
			
			dynamic_sketch.expand();
		}
		dynamic_sketch.update(packet, 1);
		naive_sketch.update(packet, 1);
		counts[packet]++;
	}
	log << std::endl;

	return (clock() - start) / CLOCKS_PER_SEC;
}

int test_expand_accuracy(const std::vector<uint32_t>& indice) {
	auto start = clock();
	const std::vector<int> sketch_counts = { 2, 4, 8 };
	constexpr int expand_factor = 8;
	constexpr int updates_per_log = 1024*512;
	std::ofstream log("expand_accuracy.dat", std::ios::trunc);

	CM_type *count_min_small, *count_min_large;
	count_min_small = CM_Init(CM_WIDTH, CM_DEPTH, SEED);
	count_min_large = CM_Init(CM_WIDTH * expand_factor, CM_DEPTH, SEED);
	std::unordered_map<int, int> counts;
	
	log << expand_factor << " ";
	for (auto sketch_count : sketch_counts) log << sketch_count << " ";
	log << std::endl;

	// graph best and worst cases
	for (int i = 0; i < indice.size(); i++) {
		if (i % updates_per_log == 0 && i > 1) {
			double mse = 0;
			for (auto p : counts) {
				int actual = p.second;
				int estimate = CM_PointEst(count_min_small, p.first);
				mse += std::pow(actual-estimate, 2);
			}
			mse /= counts.size();
			log << i << " " << mse << " ";
		}
		int packet = indice[i];
		CM_Update(count_min_small, packet, 1);
		counts[packet]++;
	}
	log << std::endl;
	
	counts.clear();
	for (int i = 0; i < indice.size(); i++) {
		if (i % updates_per_log == 0 && i > 1) {
			double mse = 0;
			for (auto p : counts) {
				mse += std::pow(p.second - CM_PointEst(count_min_large, p.first), 2);
			}
			mse /= counts.size();
			log << i << " " << mse << " ";
		}
		int packet = indice[i];
		CM_Update(count_min_large, packet, 1);
		counts[packet]++;
	}
	log << std::endl;

	// graph expansions
	for (auto sketch_count : sketch_counts) {
		counts.clear();
		const int sketch_width = CM_WIDTH * expand_factor / sketch_count;
		const int updates_per_expand = indice.size() / sketch_count;
		DynamicSketch count_split(sketch_width, CM_DEPTH, SEED);
		for (int i = 1; i < indice.size(); i++) {
			if (i % updates_per_log == 0 && i > 1) {
				double mse = 0;
				for (auto p : counts) {
					mse += std::pow(p.second - count_split.query(p.first), 2);
				}
				mse /= counts.size();
				log << i << " " << mse << " ";
			}
			// no expand on first iteration
			if (i % updates_per_expand == 0 && i > 1) {
				count_split.expand();
			}
			int packet = indice[i];
			count_split.update(packet, 1);
			counts[packet]++;
		}
		log << std::endl;
	}
	return (clock() - start)/CLOCKS_PER_SEC;
}

int test_operations_latency(const std::vector<uint32_t>& indice) {
	constexpr int max_sketch_count = 64;
	constexpr int aggregate_size = 128;
	constexpr unsigned count_logs = 1024;

	std::unordered_map<uint32_t, int> hash;
	std::map<uint32_t, int> tree;
	DynamicSketch dynamic_sketch(CM_WIDTH, CM_DEPTH, SEED);
	auto t0 = std::chrono::high_resolution_clock::now();
	auto t1 = std::chrono::high_resolution_clock::now();
	uint64_t ns;
	bool expand_phase = true;

	auto start = clock();
	std::ofstream log("latency.dat", std::ios::trunc);
	log << max_sketch_count << " " << aggregate_size << std::endl;
	for (int i = 0; i < count_logs; i++) {
		std::vector<uint32_t> chunk;
		for (int j = 0; j < aggregate_size; j++) chunk.push_back(indice[i*aggregate_size+j]);

		log << dynamic_sketch.sketchCount() << " ";

		// tree update time
		t0 = std::chrono::high_resolution_clock::now();
		for (auto packet : chunk) tree[packet]++;
		t1 = std::chrono::high_resolution_clock::now();
		ns = std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count();
		log << ns << " ";

		// tree query time
		t0 = std::chrono::high_resolution_clock::now();
		int res = 0;
		for (auto packet : chunk) res += tree[packet];
		t1 = std::chrono::high_resolution_clock::now();
		ns = std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count();
		log << ns << " " << res << " ";

		// hash update time
		t0 = std::chrono::high_resolution_clock::now();
		for (auto packet : chunk) hash[packet]++;
		t1 = std::chrono::high_resolution_clock::now();
		ns = std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count();
		log << ns << " ";

		// hash query time
		t0 = std::chrono::high_resolution_clock::now();
		for (auto packet : chunk) res += hash[packet];
		t1 = std::chrono::high_resolution_clock::now();
		ns = std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count();
		log << ns << " " << res << " ";

		// update time
		t0 = std::chrono::high_resolution_clock::now();
		for (auto packet : chunk) dynamic_sketch.update(packet, 1);
		t1 = std::chrono::high_resolution_clock::now();
		ns = std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count();
		log << ns << " ";
		
		// query time
		t0 = std::chrono::high_resolution_clock::now();
		for (auto packet : chunk) res += dynamic_sketch.query(packet);
		t1 = std::chrono::high_resolution_clock::now();
		ns = std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count();
		log << ns << " " << res << " ";
		
		if (dynamic_sketch.sketchCount() == max_sketch_count) expand_phase = false;
		else if (dynamic_sketch.sketchCount() == 1) expand_phase = true;
		if (expand_phase) {
			// expand time
			t0 = std::chrono::high_resolution_clock::now();
			dynamic_sketch.expand();
			t1 = std::chrono::high_resolution_clock::now();
			ns = std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count();
			log << "E " << ns << std::endl;
		}
		else {
			// shrink time
			t0 = std::chrono::high_resolution_clock::now();
			dynamic_sketch.shrink();
			t1 = std::chrono::high_resolution_clock::now();
			ns = std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count();
			log << "S " << ns << std::endl;
		}
	}
	log.close();
	auto end = clock();
	return (end - start) / CLOCKS_PER_SEC;
}


int test_independent_runtime(const std::vector<uint32_t>& indice) {
	constexpr int max_sketch_count = 64;
	constexpr int cycle_count = 4;

	auto t0 = std::chrono::high_resolution_clock::now();
	auto t1 = std::chrono::high_resolution_clock::now();

	std::chrono::duration<double> seconds;

	auto start = clock();
	std::ofstream log("runtime_independent.dat", std::ios::trunc);
	log << max_sketch_count << std::endl;

	DynamicSketch dynamic_sketch(CM_WIDTH, CM_DEPTH, SEED);
	const int updates_per_modify_size = indice.size() / cycle_count / max_sketch_count / 2;
		
	bool phase_expand = true;
	for (int i = 0; i < indice.size(); i++) {
		auto packet = indice[i];
		dynamic_sketch.update(packet, 1);
		if (i > 0 && i % updates_per_modify_size == 0) {
			// query test
			t0 = std::chrono::high_resolution_clock::now();
			int sum = 0;
			for (int i = 0; i < indice.size(); i++) {
				auto packet = indice[i];
				sum += dynamic_sketch.query(packet);
			}
			t1 = std::chrono::high_resolution_clock::now();
			seconds = t1 - t0;

			log << dynamic_sketch.sketchCount() << " Q " << (double)indice.size() / seconds.count() << " " << sum << std::endl;

			auto dynamic_sketch_copy = dynamic_sketch;
			// update test
			t0 = std::chrono::high_resolution_clock::now();
			for (int i = 0; i < indice.size(); i++) {
				auto packet = indice[i];
				dynamic_sketch.update(packet, 1);
			}
			t1 = std::chrono::high_resolution_clock::now();
			seconds = t1 - t0;

			log << dynamic_sketch.sketchCount() << " U " << (double)indice.size() / seconds.count() << " " << std::endl;

			dynamic_sketch = dynamic_sketch_copy;
			
			if (dynamic_sketch.sketchCount() == 1) {
				phase_expand = true;
			}
			else if (dynamic_sketch.sketchCount() == max_sketch_count) {
				phase_expand = false;
			}
			if (phase_expand) dynamic_sketch.expand();
			else dynamic_sketch.shrink();
		}
	}

	log.close();
	auto end = clock();
	return (end - start) / CLOCKS_PER_SEC;
}


int test_elastic_sketch(const std::vector<uint32_t>& indice) {
	auto start = clock();

	constexpr int num_shrinks = 8;
	constexpr int chunk_size = 1024;
	constexpr int num_buckets = 1;	// ask about this part
	const int sketch_width = CM_WIDTH;
	constexpr int updates_per_shrink = NUM_PACKETS / num_shrinks;

	ElasticSketch<1, num_shrinks* chunk_size> elastic;
	DynamicSketch dynamic(sketch_width, chunk_size / sketch_width, SEED);
	std::unordered_map<uint32_t, int> counts;
	for (int i = 0; i < num_shrinks; i++) dynamic.expand();	// kinda hoaxy


	std::ofstream log("elastic.dat", std::ios::trunc);
	for (int i = 0; i < indice.size(); i++) {
		auto packet = indice[i];
		dynamic.update(packet, 1);
		elastic.insert((uint8_t*)&packet);
		counts[packet]++;

		if (i % updates_per_shrink == 0 && i > 0) {
			double ratio = (dynamic.sketchCount() - 1) / dynamic.sketchCount();
			std::cout << dynamic.sketchCount() << " " << meanAbsError(dynamic, counts) << " " << meanAbsError(elastic, counts) << std::endl;
			std::cout << dynamic.byteSize() << " " << "mem usage elastic" << std::endl;
			dynamic.shrink();
			elastic.compress(ratio, (uint8_t*)&elastic);
		}
	}
	return clock() - start;
}


int test_error_slope(const std::vector<uint32_t>& indice) {
	constexpr int max_sketch_count = 3;
	constexpr int updates_per_log = 1024 * 128;
	int updates_per_expand = indice.size() / max_sketch_count / 2;

	std::unordered_map<uint32_t, int> counts;
	DynamicSketch dynamic_sketch(CM_WIDTH, CM_DEPTH, SEED);

	auto start = clock();
	std::ofstream log("error_slope.dat", std::ios::trunc);
	
	bool phase_expand = true;

	for (int i = 0; i < indice.size(); i++) {
		auto packet = indice[i]; 
		dynamic_sketch.update(packet, 1);
		counts[packet]++;
		if (i % updates_per_expand == 0 && i > 0) {
			std::cout << dynamic_sketch.sketchCount() << std::endl;
			if (phase_expand) {
				dynamic_sketch.expand();
				if (dynamic_sketch.sketchCount() == max_sketch_count) {
					phase_expand = false;
				}
			}
			else dynamic_sketch.shrink();
		}
		if (i % updates_per_log == 0 && i > 0) {
			double mae = 0;
			for (auto p : counts) {
				mae += std::abs(p.second - dynamic_sketch.query(p.first));
			}
			mae /= counts.size();
			log << dynamic_sketch.sketchCount() << " " << i << " " << mae << std::endl;
		}
	}
	log << std::endl;

	for (int j = 1; j <= max_sketch_count; j++) {
		CM_type* count_min = CM_Init(CM_WIDTH * j, CM_DEPTH, SEED);
		counts.clear();
		for (int i = 0; i < indice.size(); i++) {
			auto packet = indice[i];
			CM_Update(count_min, packet, 1);
			counts[packet]++;
			if (i % updates_per_log == 0 && i > 0) {
				double mae = 0;
				for (auto p : counts) {
					mae += std::abs(p.second - CM_PointEst(count_min, p.first));
				}
				mae /= counts.size();
				log << mae << std::endl;
			}
		}
		log << std::endl;
	}

	log.close();
	auto end = clock();
	return (end - start) / CLOCKS_PER_SEC;
}

int test_amplitude_frequency(const std::vector<uint32_t>& indice) {
	constexpr int iteration_count = 16;
	constexpr int logs_per_trace = 1024;
	auto start = clock();
	std::ofstream log("runtime.dat", std::ios::trunc);

	const std::vector<int> expand_frequency = { 1, 2, 4 };
	const std::vector<int> expand_amplitude = { 1, 2, 4 };

	double dt;

	// multiple iterations to make sure
	for (int iter = 0; iter < iteration_count; iter++) {
		std::map<uint32_t, uint32_t> counts;
		dt = test_map_time(indice, counts);
		log << dt << " ";

		std::unordered_map<uint32_t, uint32_t> counts_unordered;
		dt = test_unordered_map_time(indice, counts_unordered);
		log << dt << std::endl;

		// graph expansions
		for (auto frequency : expand_frequency) for (auto amplitude : expand_amplitude) {
			auto t0 = std::chrono::high_resolution_clock::now();

			std::unordered_map<int, int> counts;

			bool dir = 1;
			const int updates_per_log = indice.size() / logs_per_trace;
			const int updates_per_modify = indice.size() / frequency / 2;
			const int sketch_width = CM_WIDTH / amplitude;

			DynamicSketch count_split(sketch_width, CM_DEPTH, SEED);
			for (int i = 1; i < indice.size(); i++) {
				// no expand on first iteration
				if (i % updates_per_modify == 0 && i > 1) {
					int n = count_split.sketchCount();
					if (n == 1) dir = 1;
					else if (n == amplitude) dir = 0;
					if (dir == 1) count_split.expand();
					else count_split.shrink();
				}
				int packet = indice[i];
				count_split.update(packet, 1);
				counts[packet]++;
			}

			auto t1 = std::chrono::high_resolution_clock::now();
			std::chrono::duration<double, std::milli> dt = t1 - t0;
			double dur = dt.count() / 1000;
			log << frequency << " " << amplitude << " " << dur << std::endl;
		}
	}
	log.close();
	return (clock() - start) / CLOCKS_PER_SEC;
}

int test_modify_size(const std::vector<uint32_t>& indice) {
	auto start = clock();
	
	std::ofstream log("modify_size.dat", std::ios::trunc);

	const std::vector<int> expand_frequency = { 2, 4, 8 };
	const std::vector<int> expand_amplitude = { 2, 4, 8 };
	const int updates_per_log = 1024 * 512;

	// graph expansions
	for (auto frequency: expand_frequency) for (auto amplitude : expand_amplitude) {
		log << frequency << " " << amplitude << std::endl;
		
		std::unordered_map<int, int> counts;

		bool dir = 1;
		const int updates_per_modify = indice.size() / frequency / 2;
		const int sketch_width = CM_WIDTH / amplitude;
		
		DynamicSketch count_split(sketch_width, CM_DEPTH, SEED);
		for (int i = 1; i < indice.size(); i++) {
			if (i % updates_per_log == 0 && i > 1) {
				double mse = 0;
				for (auto p : counts) {
					mse += std::pow(p.second - count_split.query(p.first), 2);
				}
				mse /= counts.size();
				log << i << " " << mse << " ";
			}
			// no expand on first iteration
			if (i % updates_per_modify == 0 && i > 1) {
				int n = count_split.sketchCount();
				if (n == 1) dir = 1;
				else if (n == amplitude) dir = 0;
				if (dir == 1) count_split.expand();
				else count_split.shrink();
			}
			int packet = indice[i];
			count_split.update(packet, 1);
			counts[packet]++;
		}
		log << std::endl;
	}
	log.close();
	return (clock() - start) / CLOCKS_PER_SEC;
}

int main(int argc, char* argv[])
{
	std::ofstream file_res;

	double dt;
	std::vector<uint32_t> indice;
	indice.reserve(indice.size());

	printf("parsing pcap file...\n");
	dt = parseCapture("C:/capture.txt", &indice);
	printf("%f seconds elapsed\n", dt);
	
	printf("testing naive...\n");
	dt = test_naive(indice);
	printf("%f seconds elapsed\n", dt);

	printf("testing amplitude frequency...\n");
	dt = test_amplitude_frequency(indice);
	printf("%f seconds elapsed\n", dt);

	printf("testing error slope...\n");
	dt = test_error_slope(indice);
	printf("%f seconds elapsed\n", dt);

	printf("testing memory usage...\n");
	dt = test_memory_usage(indice);
	printf("%f seconds elapsed\n", dt);

	printf("testing expand accuracy...\n");
	dt = test_expand_accuracy(indice);
	printf("%f seconds elapsed\n", dt);

	printf("testing operations' latency...\n");
	dt = test_operations_latency(indice);
	printf("%f seconds elapsed\n", dt);

	printf("testing independent runtime...\n");
	dt = test_independent_runtime(indice);
	printf("%f seconds elapsed\n", dt);

	printf("testing modify size...\n");
	dt = test_modify_size(indice);
	printf("%f seconds elapsed\n", dt);

	file_res.close();
	return 1;

}