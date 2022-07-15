#include <unordered_map>
#include <time.h>
#include <chrono>

#include "CountMinSketch.h"
#include "DynamicSketch.h"
#include "countminexpanding.h"
#include "countmin.h"
#include "VectorSketch.h"

constexpr float epsilon = 0.1;
constexpr float delta = 0.1;
constexpr int SEED = 0x1337C0D3;
constexpr int NUM_PACKETS = 1024 * 1024 * 32;
const int CM_WIDTH = ceil(exp(1) / epsilon);
const int CM_DEPTH = ceil(log(1 / delta));


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

double test_count_min(const std::vector<uint32_t>& indice, const std::unordered_map<uint32_t, int>& counts, double& mse, double& dt_query, double& dt_update) {
	CM_type* count_min = CM_Init(CM_WIDTH, CM_DEPTH, SEED);

	auto t0 = std::chrono::high_resolution_clock::now();
	for (const auto& index : indice) {
		CM_Update(count_min, index, 1);
	}
	auto t1 = std::chrono::high_resolution_clock::now();
	mse = 0;
	for (const auto& count : counts) {
		uint32_t index = count.first;
		int index_count = count.second;
		mse += std::pow(CM_PointEst(count_min, index) - index_count, 2);
	}
	mse /= counts.size();

	auto t2 = std::chrono::high_resolution_clock::now();
	std::chrono::duration<double, std::milli> dt0 = t1 - t0;
	std::chrono::duration<double, std::milli> dt1 = t2 - t1;
	dt_update = dt0.count() / indice.size();
	dt_query = dt1.count() / counts.size();
	delete count_min;
	return (dt0.count() + dt1.count()) / 1000;
}

void test_count_ring_accuracy(const std::vector<uint32_t>& indice) {
	std::ofstream file;
	file.open("once2.dat");
	//for (int incr_per_expand = 1000; incr_per_expand <= 1000; incr_per_expand *= 2) {
		DynamicSketch count_min_ring(CM_WIDTH, CM_DEPTH, SEED);
		std::unordered_map<uint32_t, int> counts;

		int counter = 256*4;
		file << 0;
		for (const auto& index : indice) {
			count_min_ring.update(index, 1);
			if (counts.find(index) == counts.end()) {
				counts[index] = 1;
			}
			else {
				counts[index]++;
			}

			counter--;
			if (counter == 0) {
				counter = 256 * 4;
				int size = count_min_ring.sketchCount();
				count_min_ring.expand();
				std::cout << size << std::endl;

				double mse = 0;
				for (const auto& count : counts) {
					uint32_t index = count.first;
					int index_count = count.second;
					mse += std::pow(count_min_ring.query(index) - index_count, 2);
				}
				mse /= counts.size();

				file << "," << mse;
			}
		}
	//}
	file.close();
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

int minimum_binary_tree_size(int value_size, int elements_count) {
	return (sizeof(void*) * 2 + value_size) * elements_count;
}

int minimum_hash_map_size(int value_size, int elements_count, float load_factor) {
	return value_size * elements_count / load_factor;
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
	constexpr int size_factor = 10;
	auto t0 = std::chrono::high_resolution_clock::now();
	std::map<uint32_t, uint32_t> m;

	std::ofstream log("memory_usage.dat", std::ios::trunc);

	DynamicSketch sketch_hash(CM_WIDTH, CM_DEPTH, SEED);
	DynamicSketch sketch_tree(CM_WIDTH, CM_DEPTH, SEED);

	constexpr int updates_per_log = 1024 * 512;

	for (int i = 0; i < NUM_PACKETS; i++) {
		auto packet = indice[i];

		// sized a tenth of the size of a tree/hash
		int size_hash = minimum_hash_map_size(sizeof(uint32_t), m.size(), 0.75);
		int size_tree = minimum_binary_tree_size(sizeof(uint32_t), m.size());
		while (sketch_hash.byteSize() * size_factor < size_hash) {
			sketch_hash.expand();
		}
		while (sketch_tree.byteSize() * size_factor < size_tree) {
			sketch_tree.expand();
		}

		if (i % updates_per_log == 0 && i > 1) {
			double mse_hash = 0;
			for (auto p : m) {
				mse_hash += std::pow((float)p.second - sketch_hash.query(p.first), 2);
			}
			mse_hash /= i;

			double mse_tree = 0;
			for (auto p : m) {
				mse_tree += std::pow((float)p.second - sketch_tree.query(p.first), 2);
			}
			mse_tree /= i;

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

int test_expand_improvement_bound(const std::vector<uint32_t>& indice) {
	auto start = clock();
	constexpr int expand_count = 512;
	const int updates_per_expand = indice.size() / expand_count;
	std::ofstream log("expand_improvement_bound.dat", std::ios::trunc);

	std::unordered_map<uint32_t, int> counts;
	
	DynamicSketch dynamic_sketch(CM_WIDTH, CM_DEPTH, SEED);
	for (int i = 1; i < indice.size(); i++) {
		auto packet = indice[i];
		if (i % updates_per_expand == 0 && i > 1) {
			double mae = 0;
			for (auto p : counts) {
				mae += std::abs(p.second - dynamic_sketch.query(p.first));
			}
			mae /= i;
			log << dynamic_sketch.sketchCount() << " " << mae << std::endl;
			dynamic_sketch.expand();
		}
		dynamic_sketch.update(packet, 1);
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
	for (int i = 0; i < NUM_PACKETS; i++) {
		if (i % updates_per_log == 0 && i > 1) {
			double mse = 0;
			for (auto p : counts) {
				mse += std::pow(p.second - CM_PointEst(count_min_small, p.first), 2);
			}
			mse /= i;
			log << i << " " << mse << " ";
		}
		int packet = indice[i];
		CM_Update(count_min_small, packet, 1);
		counts[packet]++;
	}
	log << std::endl;
	
	counts.clear();
	for (int i = 0; i < NUM_PACKETS; i++) {
		if (i % updates_per_log == 0 && i > 1) {
			double mse = 0;
			for (auto p : counts) {
				mse += std::pow(p.second - CM_PointEst(count_min_large, p.first), 2);
			}
			mse /= i;
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
		const int updates_per_expand = NUM_PACKETS / sketch_count;
		DynamicSketch count_split(sketch_width, CM_DEPTH, SEED);
		for (int i = 1; i < NUM_PACKETS; i++) {
			if (i % updates_per_log == 0 && i > 1) {
				double mse = 0;
				for (auto p : counts) {
					mse += std::pow(p.second - count_split.query(p.first), 2);
				}
				mse /= i;
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
	constexpr int count_logs = 1024*1024;


	DynamicSketch dynamic_sketch(CM_WIDTH, CM_DEPTH, SEED);
	std::chrono::steady_clock::time_point t0, t1;
	uint64_t ns;
	bool expand_phase = true;

	auto start = clock();
	std::ofstream log("latency.dat", std::ios::trunc);
	log << max_sketch_count << std::endl;
	for (int i = 0; i < min(count_logs, indice.size()) ; i++) {
		auto packet = indice[i];

		// update time
		t0 = std::chrono::high_resolution_clock::now();
		dynamic_sketch.update(packet, 1);
		t1 = std::chrono::high_resolution_clock::now();
		ns = std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count();
		log << dynamic_sketch.sketchCount() << " " << ns << " ";
		
		// query time
		t0 = std::chrono::high_resolution_clock::now();
		dynamic_sketch.query(packet);
		t1 = std::chrono::high_resolution_clock::now();
		ns = std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count();
		log << ns << " ";
		
		if (dynamic_sketch.sketchCount() == max_sketch_count) expand_phase = false;
		else if (dynamic_sketch.sketchCount() == 1) expand_phase = true;
		if (expand_phase) {
			// expand time
			t0 = std::chrono::high_resolution_clock::now();
			dynamic_sketch.expand();
			t1 = std::chrono::high_resolution_clock::now();
			ns = std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count();
			log << "0 " << ns << std::endl;
		}
		else {
			// shrink time
			t0 = std::chrono::high_resolution_clock::now();
			dynamic_sketch.shrink();
			t1 = std::chrono::high_resolution_clock::now();
			ns = std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count();
			log << "1 " << ns << std::endl;
		}
	}
	log.close();
	auto end = clock();
	return (end - start) / CLOCKS_PER_SEC;
}


int test_independent_runtime(const std::vector<uint32_t>& indice) {
	constexpr int max_sketch_count = 64;
	constexpr int cycle_count = 4;

	std::chrono::steady_clock::time_point t0, t1;
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
			
			std::cout << dynamic_sketch.sketchCount() << std::endl;
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

int test_error_slope(const std::vector<uint32_t>& indice) {
	constexpr int max_sketch_count = 3;
	constexpr int updates_per_log = 1024 * 128;
	int updates_per_expand = indice.size() / (max_sketch_count) / (2-1);

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
			if (phase_expand) {
				dynamic_sketch.expand();
				if (dynamic_sketch.sketchCount() == max_sketch_count) phase_expand = false;
			}
			else dynamic_sketch.shrink();
		}
		if (i % updates_per_log == 0 && i > 0) {
			double mae = 0;
			for (auto p : counts) {
				mae += std::abs(p.second - dynamic_sketch.query(p.first));
			}
			mae /= i;
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
				mae /= i;
				log << mae << std::endl;
			}
		}
		log << std::endl;
	}

	log.close();
	auto end = clock();
	return (end - start) / CLOCKS_PER_SEC;
}

int test_runtime(const std::vector<uint32_t>& indice) {
	constexpr int iteration_count = 16;
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
			const int updates_per_log = 1024 * 512;
			const int updates_per_modify = NUM_PACKETS / frequency / 2;
			const int sketch_width = CM_WIDTH / amplitude;

			DynamicSketch count_split(sketch_width, CM_DEPTH, SEED);
			for (int i = 1; i < NUM_PACKETS; i++) {
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
		const int updates_per_modify = NUM_PACKETS / frequency / 2;
		const int sketch_width = CM_WIDTH / amplitude;
		
		DynamicSketch count_split(sketch_width, CM_DEPTH, SEED);
		for (int i = 1; i < NUM_PACKETS; i++) {
			if (i % updates_per_log == 0 && i > 1) {
				double mse = 0;
				for (auto p : counts) {
					mse += std::pow(p.second - count_split.query(p.first), 2);
				}
				mse /= i;
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
	indice.reserve(NUM_PACKETS);

	printf("parsing pcap file...\n");
	dt = parseCapture("C://capture.txt", &indice);
	printf("%f sea conds elapsed\n", dt);

	printf("testing operations' latency...\n");
	dt = test_operations_latency(indice);
	printf("%f seconds elapsed\n", dt);

	printf("testing independent runtime...\n");
	dt = test_independent_runtime(indice);
	printf("%f seconds elapsed\n", dt);

	printf("testing expand improvement bound...\n");
	dt = test_expand_improvement_bound(indice);
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

	printf("testing modify size...\n");
	dt = test_modify_size(indice);
	printf("%f seconds elapsed\n", dt);

	printf("testing runtime...\n");
	dt = test_runtime(indice);
	printf("%f seconds elapsed\n", dt);

	file_res.close();
	return 1;

}