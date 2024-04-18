#define DOCTEST_CONFIG_IMPLEMENT
#include "doctest.h"

#include <unordered_map>
#include <time.h>
#include <chrono>
#include <iostream>
#include <string>

#include "DynamicSketch.h"
#include "GeometricSketch.h"
#include "GeometricSketchMH.h"
#include "Dictionary.h"

#include "MultiHash.h"

typedef std::chrono::high_resolution_clock chrono_clock;
typedef std::chrono::duration<double, std::milli> duration;

constexpr double epsilon = 0.01;
constexpr double delta = 0.01;
constexpr int SEED = 0x1337C0D3;

int DIST_BUCKET_COUNT = 32;
int CM_WIDTH = 272;
int CM_DEPTH = 5;
int BUCKET_COUNT = 32;
int BRANCHING_FACTOR = 2;
bool DCMS_USE_SAME_SEED = false;

// --file path-- type dynamic-- repeat[command num][modify_size_random | modify_size_cyclic | expand | shrink | log_memory_usage | log_size | log_time][times]-- time
// --packets [num] [packet0] ...

struct ActionTimer
{
	long packets_per_action;
	std::string action_name;
	long argument;
	bool is_repeat;

	ActionTimer(std::string _action_name, long _packets_per_action, bool _is_repeat, long _argument) : action_name(_action_name),
																					  packets_per_action(_packets_per_action),
																					  is_repeat(_is_repeat),
																					argument(_argument){}
};

void loadPacketsFromArgs(const char *argv[], int &i, int packet_num, std::vector<uint32_t> *packets, int max_packets = -1)
{
	packets->clear();
	uint32_t ipaddr;
	for (int i = 0; i < packet_num; i++)
	{
		if (i != max_packets)
		{
			ipaddr = static_cast<uint32_t>(std::stoi(argv[++i]));
			packets->push_back(ipaddr);
		}
	}
}

void loadPacketsFromFile(std::string path, std::vector<uint32_t> *packets, int max_packets = -1)
{
	packets->clear();
	std::ifstream myfile(path);
	if (!myfile.is_open())
	{
		throw std::invalid_argument("--file=" + path + "?");
	}
	uint32_t ipaddr;
	while (myfile >> ipaddr)
	{
		if (packets->size() == max_packets)
			break;
		packets->push_back(ipaddr);
	}
}

Dictionary *createDictionary(std::string type)
{
	if (type == "countmin")
	{
		return new CountMinDictionary(CM_WIDTH, CM_DEPTH, SEED);
	}
	else if (type == "geometric")
	{
		return new GeometricSketch(CM_WIDTH, CM_DEPTH, BRANCHING_FACTOR);
	}
	else if (type == "multihash")
	{
		return new GeometricSketchMH(CM_WIDTH, CM_DEPTH, BRANCHING_FACTOR);
	}
    else if (type == "dynamic"){
        return new DynamicSketch(CM_WIDTH, CM_DEPTH, DCMS_USE_SAME_SEED);
    }
	else
	{
		throw std::invalid_argument("--type=" + type + "?");
	}
}

double calculateAverageAbsoluteError(Dictionary *dictionary, const std::vector<uint32_t> &packets, int packet_index)
{
	if (packet_index == 0)
		return 0;
	std::unordered_map<uint32_t, int> hashtable;
	for (int i = 0; i < packet_index; i++)
	{
		hashtable[packets[i]] += 1;
	}

	double delta = 0.0;
	for (auto const &packet_count_pair : hashtable)
	{
		uint32_t packet = packet_count_pair.first;
		int count = packet_count_pair.second;
		int dictionary_estimate = dictionary->query(packet);
		double mae = std::abs(dictionary_estimate - count);
		delta += mae;
	}
	return delta / hashtable.size();
}

void printHeavyHitters(int limit, Dictionary* dictionary, const std::vector<uint32_t>& packets, int packet_index)
{
	std::cout << "{\"index\":" << packet_index << ",\"log_heavy_hitters\":[";
	if (packet_index > 0) {
		std::unordered_map<uint32_t, int> packet_to_count;
		for (int i = 0; i < packet_index; i++)
		{
			packet_to_count[packets[i]] += 1;
		}
		std::multimap<int, uint32_t> count_to_packet;
		for (auto const& packet_count_pair : packet_to_count)
		{
			uint32_t packet = packet_count_pair.first;
			int count = packet_count_pair.second;
			count_to_packet.insert(std::make_pair(-count, packet));
		}

		bool is_last = false;
		int i = 0;
		for (auto const& count_packet_pair : count_to_packet)
		{
			if (i++ >= limit) {
				break;
			}
			uint32_t packet = count_packet_pair.second;
			int count = -count_packet_pair.first;
			int query = dictionary->query(packet);
			std::cout << "{\"id\":" << packet << ",\"count\":" << count << ",\"query\":" << query << "}";
			is_last = (i >= limit) || (i + 1 == count_to_packet.size() - 1);
			if (!is_last) {
				std::cout << ",";
			}
		}
	}
		std::cout << "]}," << std::endl;
}

double calculateAverageRelativeError(Dictionary *dictionary, const std::vector<uint32_t> &packets, int packet_index)
{
	if (packet_index == 0)
	return 0;
	std::unordered_map<uint32_t, int> hashtable;
	for (int i = 0; i < packet_index; i++)
	{
		hashtable[packets[i]] += 1;
	}

	double total_re = 0.0;
	for (auto const &packet_count_pair : hashtable)
	{
		uint32_t packet = packet_count_pair.first;
		int count = packet_count_pair.second;
		int dictionary_estimate = dictionary->query(packet);
		double re = std::abs(dictionary_estimate - count) / count;
		total_re += re;
	}
	return total_re / hashtable.size();
}


double calculateMeanSquaredError(Dictionary *dictionary, const std::vector<uint32_t> &packets, int packet_index)
{
	if (packet_index == 0)
		return 0;
	std::unordered_map<uint32_t, int> hashtable;
	for (int i = 0; i < packet_index; i++)
	{
		hashtable[packets[i]] += 1;
	}

	double delta = 0.0;
	for (auto const &packet_count_pair : hashtable)
	{
		uint32_t packet = packet_count_pair.first;
		int count = packet_count_pair.second;
		int dictionary_estimate = dictionary->query(packet);
		double mse = std::pow(dictionary_estimate - count, 2);
		delta += mse;
	}
	return delta / hashtable.size();
}

std::set<int> uniquePacketsBeforeIndex(const std::vector<uint32_t> &packets, int packet_index)
{
	auto it_first = packets.begin();
	auto it_last = packets.begin() + packet_index;
	return std::set<int>(it_first, it_last);
}

void doPendingActions(Dictionary *dictionary, const std::vector<uint32_t> &packets, std::vector<ActionTimer> &action_timers, int packet_index)
{
	auto it = action_timers.begin();
	while (it != action_timers.end())
	{
        bool deleted = false;
        const ActionTimer& action_timer = *it;
        if (action_timer.packets_per_action == 0 || (packet_index > 0 && packet_index % action_timer.packets_per_action == 0))
		{			
			std::string action_name = action_timer.action_name;
			if (action_name == "expand")
			{
				auto t_start = chrono_clock::now();
				dictionary->expand(action_timer.argument);
				duration duration_expand = chrono_clock::now() - t_start;
				double expand_time = duration_expand.count();
				std::cout << "{\"log_expand_time\":" << expand_time << ",\"index\":" << packet_index << "}," << std::endl;
			}
            else if (action_name == "compress")
            {
				auto t_start = chrono_clock::now();
                dictionary->compress(action_timer.argument);
				duration duration_compress = chrono_clock::now() - t_start;
				double compress_time = duration_compress.count();
				std::cout << "{\"log_compress_time\":" << compress_time << ",\"index\":" << packet_index << "}," << std::endl;
            }
			else if (action_name == "shrink")
			{
				auto t_start = chrono_clock::now();
				dictionary->shrink(action_timer.argument);
				duration duration_shrink = chrono_clock::now() - t_start;
				double shrink_time = duration_shrink.count();
				std::cout << "{\"log_shrink_time\":" << shrink_time << ",\"index\":" << packet_index << "}," << std::endl;
			}
			else if (action_name == "log_memory_usage")
			{
				std::cout << "{\"memory_usage\":" << dictionary->getMemoryUsage() << ",\"index\":" << packet_index << "}," << std::endl;
			}
			else if (action_name == "log_average_relative_error"){
				double error = calculateAverageRelativeError(dictionary, packets, packet_index);
				std::cout << "{\"log_average_relative_error\":" << error << ",\"index\":" << packet_index << "}," << std::endl;
			}
			else if (action_name == "log_heavy_hitters") {
				int limit = action_timer.argument;
				printHeavyHitters(limit, dictionary, packets, packet_index);
			}
			else if (action_name == "log_mean_squared_error")
			{
				double error = calculateMeanSquaredError(dictionary, packets, packet_index);
				std::cout << "{\"log_mean_squared_error\":" << error << ",\"index\":" << packet_index << "}," << std::endl;
			}
			else if (action_name == "log_average_absolute_error")
			{
				double error = calculateAverageAbsoluteError(dictionary, packets, packet_index);
				std::cout << "{\"log_average_absolute_error\":" << error << ",\"index\":" << packet_index << "}," << std::endl;
			}
			else if (action_name == "log_size")
			{
				int size = dictionary->getSize();
				std::cout << "{\"log_size\":" << size << ",\"index\":" << packet_index << "}," << std::endl;
			}
			else if (action_name == "log_update_time")
			{
				auto t_start = chrono_clock::now();
				for (const auto& packet : packets)
				{
					dictionary->update(packet, 1);
					dictionary->update(packet, -1);
				}
				duration duration_update = chrono_clock::now() - t_start;
				double update_time = duration_update.count() / (2.0 * packets.size());
				std::cout << "{\"log_update_time\":" << update_time << ",\"index\":" << packet_index << "}," << std::endl;
			}
			else if (action_name == "log_query_time")
			{
				auto t_start = chrono_clock::now();
				for (const auto& packet : packets)
				{
					dictionary->query(packet);
				}
				duration duration_query = chrono_clock::now() - t_start;
				double query_time = duration_query.count() / ((double)packets.size());
				std::cout << "{\"log_query_time\":" << query_time << ",\"index\":" << packet_index << "}," << std::endl;
			}
			else if (action_name == "log_unique_packet_count")
			{
				auto unique_packets_so_far = uniquePacketsBeforeIndex(packets, packet_index + 1);
				size_t unique_packet_count = unique_packets_so_far.size();
				std::cout << "{\"log_unique_packet_count\":" << unique_packet_count << ",\"index\":" << packet_index << "}," << std::endl;
			}
			else
			{
				throw std::invalid_argument(action_name + "?");
			}
            if (!action_timer.is_repeat){
                deleted = true;
                it = action_timers.erase(it);
            }
        }
        if (!deleted){
            it++;
        }
	}
}

void run(Dictionary *dictionary, const std::vector<uint32_t> &packets, std::vector<ActionTimer> action_timers)
{
	std::cout << "[" << std::endl;
	for (int i = 0; i < packets.size(); i++)
	{
		doPendingActions(dictionary, packets, action_timers, i);
		uint32_t packet = packets[i];
		dictionary->update(packet, 1);
	}
	std::cout << "]" << std::endl;
}

void proccess_input(int argc, const char *argv[])
{
	std::string type = "";
	Dictionary *dictionary = nullptr;
	std::vector<uint32_t> packets;
	std::vector<ActionTimer> action_timers;
	int i = 0;
	while (argv[++i])
	{
		std::string arg = argv[i];

		if (arg == "--file" || arg == "-f")
		{
			std::string path = argv[++i];
			loadPacketsFromFile(path, &packets);
		}
		else if (arg == "--limit_file" || arg == "-l")
		{
			std::string path = argv[++i];
			int max_packets = std::stoi(argv[++i]);
			loadPacketsFromFile(path, &packets, max_packets);
		}
		else if (arg == "--packets" || arg == "-p")
		{
			int packet_num = std::stoi(argv[++i]);
			loadPacketsFromArgs(argv, i, packet_num, &packets);
		}
		else if (arg == "--type" || arg == "-t")
		{
			if (type.length() > 0)
			{
				throw std::invalid_argument("only one --type allowed");
			}
			type = argv[++i];
		}
		else if (arg == "--buckets")
		{
			BUCKET_COUNT = std::stoi(argv[++i]);
		}
		else if (arg == "--branching_factor")
		{
			BRANCHING_FACTOR = std::stoi(argv[++i]);
		}
		else if (arg == "--width")
		{
			CM_WIDTH = std::stoi(argv[++i]);
		}
		else if (arg == "--depth")
		{
			CM_DEPTH = std::stoi(argv[++i]);
		}
		else if (arg == "--dcms_same_seed")
		{
			DCMS_USE_SAME_SEED = std::stoi(argv[++i]);
		}
		else if (arg == "--repeat" || arg == "--once")
		{
			bool is_repeat = "--repeat" == arg;
			std::string action_name = argv[++i];
			int packets_per_action = std::stoi(argv[++i]);
			long argument = 0;
			if (action_name == "expand" || action_name == "shrink" | action_name == "compress" | action_name == "log_compress_time" | action_name == "log_expand_and_shrink_time" | action_name == "log_heavy_hitters")
			{
				argument = std::stol(argv[++i]);
			}
			ActionTimer action_timer = ActionTimer(action_name, packets_per_action, is_repeat, argument);
			action_timers.push_back(action_timer);
		}
		else
		{
			throw std::invalid_argument(arg + "?");
		}
	}
	if (type.length() > 0)
	{
		dictionary = createDictionary(type);
		run(dictionary, packets, action_timers);
		delete dictionary;
	}
	else
	{
		throw std::invalid_argument("--type required");
	}
}

void manual_argument()
{
	std::string cmd = "--limit_file ..\\pcaps\\capture.txt 4000000 --type geometric --width 272 --depth 5 --branching_factor 2 --once log_memory_usage 0 --repeat log_memory_usage 250000 --once log_memory_usage 3999999 --once log_heavy_hitters 3999999 100 --once expand 2720 1849600 --once expand 3701920 2515456000";
	std::vector<const char*> args;
	std::istringstream iss(cmd);

	args.push_back("");
	std::string token;
	while (iss >> token)
	{
		char *arg = new char[token.size() + 1];
		copy(token.begin(), token.end(), arg);
		arg[token.size()] = '\0';
		args.push_back(arg);
	}
	args.push_back(0);

	proccess_input(args.size(), &args[0]);
	// args leaks memory, but its a hack anyways and should exit right after
}

int test() {
	doctest::Context context;
	int result = context.run();
	return result;
}

int main(int argc, const char *argv[])
{
	// test();
    // manual_argument();
    proccess_input(argc, argv);

	/*
	uint64_t sum = 0;
	int key = 0;

	double M_OP = 100;
	int layer_count = 6;

	MultiHash mh;
	mh.setFirstSubHashModulus(16);
	mh.setSubHashModulus(4);
    auto t_start = chrono_clock::now();
	for (int i = 0; i < M_OP * 1000000 / layer_count; i++) {
		mh.initialize(key, i);
		auto vf = mh.first();
		sum += vf;
		for (int j = 0; j < layer_count - 1; j++) {
			auto v = mh.next();
			sum += v;
		}
	}
    std::chrono::duration<float> d = chrono_clock::now() - t_start;
    std::cout << "MultiHash " << M_OP / d.count() << " MOPS" << std::endl;

    t_start = chrono_clock::now();
    for(int i=0; i< M_OP * 1000000; i++){
        auto v = XXH64(&key, sizeof(uint64_t), i);
        sum += v;
    }
    d = chrono_clock::now() - t_start;
    std::cout << "XXH64 " << M_OP / d.count() << " MOPS" << std::endl;


	return sum;
	*/
}
