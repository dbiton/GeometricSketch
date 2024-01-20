#include <unordered_map>
#include <time.h>
#include <chrono>
#include <iostream>
#include <string>

#include "DynamicSketch.h"
#include "NaiveSketch.h"
#include "ElasticSketch.h"
#include "Dictionary.h"
#include "LinkedCellSketch.h"

typedef std::chrono::high_resolution_clock chrono_clock;
typedef std::chrono::duration<double, std::milli> duration;

constexpr float epsilon = 0.01;
constexpr float delta = 0.01;
constexpr int SEED = 0x1337C0D3;

int DIST_BUCKET_COUNT = 32;
int CM_WIDTH = ceil(exp(1) / epsilon);
int CM_DEPTH = ceil(log(1 / delta));
int BUCKET_COUNT = 32;
int MEMORY_USAGE = 1024 * 1024 * 32; // 32MB
int BRANCHING_FACTOR = 2;

duration duration_update = duration::zero();
duration duration_expand = duration::zero();
;
duration duration_shrink = duration::zero();
;

// --file path-- type dynamic-- repeat[command num][modify_size_random | modify_size_cyclic | expand | shrink | log_memory_usage | log_size | log_time][times]-- time
// --packets [num] [packet0] ...

struct ActionTimer
{
	int packets_per_action;
	std::string action_name;
	int argument;
	bool is_repeat;

	ActionTimer(std::string _action_name, int _packets_per_action, bool _is_repeat) : action_name(_action_name),
																					  packets_per_action(_packets_per_action),
																					  is_repeat(_is_repeat) {}
};

void loadPacketsFromArgs(const char *argv[], int &i, int packet_num, std::vector<uint32_t> *packets, int max_packets = -1)
{
	packets->clear();
	uint32_t ipaddr;
	for (int i = 0; i < packet_num; i++)
	{
		if (i != max_packets)
		{
			ipaddr = static_cast<uint32_t>(stoi(argv[++i]));
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
	else if (type == "countsketch")
	{
		return nullptr;
		// return new CountSketchDictionary(CM_WIDTH, CM_DEPTH);
	}
	else if (type == "elastic")
	{
		return new ElasticDictionary(BUCKET_COUNT, BUCKET_COUNT * COUNTER_PER_BUCKET * 8 + CM_WIDTH * CM_DEPTH, SEED);
	}
	else if (type == "cellsketch")
	{
		return new LinkedCellSketch(CM_WIDTH, CM_DEPTH, BRANCHING_FACTOR);
	}
    else if (type == "dynamic"){
        return new DynamicSketch(CM_WIDTH, CM_DEPTH);
    }
	else
	{
		throw std::invalid_argument("--type=" + type + "?");
	}
}

double calculateMeanAbsoluteError(Dictionary *dictionary, const std::vector<uint32_t> &packets, int packet_index)
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
        auto action_timer = *it;
		if (packet_index % action_timer.packets_per_action == 0 && packet_index > 0)
		{			
			std::string action_name = action_timer.action_name;
			if (action_name == "expand")
			{
				dictionary->expand(action_timer.argument);
			}
            else if (action_name == "compress")
            {
                dictionary->compress(action_timer.argument);
            }
			else if (action_name == "shrink")
			{
				dictionary->shrink(action_timer.argument);
			}
			else if (action_name == "log_memory_usage")
			{
				std::cout << "{\"memory_usage\":" << dictionary->getMemoryUsage() << ",\"index\":" << packet_index << "}," << std::endl;
			}
			else if (action_name == "log_average_relative_error"){
				double error = calculateAverageRelativeError(dictionary, packets, packet_index);
				std::cout << "{\"log_average_relative_error\":" << error << ",\"index\":" << packet_index << "}," << std::endl;
			}
			else if (action_name == "log_mean_squared_error")
			{
				double error = calculateMeanSquaredError(dictionary, packets, packet_index);
				std::cout << "{\"log_mean_squared_error\":" << error << ",\"index\":" << packet_index << "}," << std::endl;
			}
			else if (action_name == "log_mean_absolute_error")
			{
				double error = calculateMeanAbsoluteError(dictionary, packets, packet_index);
				std::cout << "{\"log_mean_absolute_error\":" << error << ",\"index\":" << packet_index << "}," << std::endl;
			}
			else if (action_name == "log_size")
			{
				int size = dictionary->getSize();
				std::cout << "{\"log_size\":" << size << ",\"index\":" << packet_index << "}," << std::endl;
			}
			else if (action_name == "log_update_time")
			{
				double update_time = duration_update.count() / ((double)action_timer.packets_per_action);
				std::cout << "{\"log_update_time\":" << update_time << ",\"index\":" << packet_index << "}," << std::endl;
				duration_update = duration::zero();
			}
			else if (action_name == "log_query_time")
			{
				auto unique_packets_so_far = uniquePacketsBeforeIndex(packets, packet_index + 1);
				duration duration_query = duration::zero();
				for (const auto &packet : unique_packets_so_far)
				{
					auto t0 = chrono_clock::now();
					dictionary->query(packet);
					duration_query += chrono_clock::now() - t0;
				}
				double query_time = duration_query.count() / ((double)unique_packets_so_far.size());
				std::cout << "{\"log_query_time\":" << query_time << ",\"index\":" << packet_index << "}," << std::endl;
			}
			else if (action_name == "log_unique_packet_count")
			{
				auto unique_packets_so_far = uniquePacketsBeforeIndex(packets, packet_index + 1);
				int unique_packet_count = unique_packets_so_far.size();
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
		auto t0 = chrono_clock::now();
		dictionary->update(packet, 1);
		duration_update += chrono_clock::now() - t0;
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
			int max_packets = stoi(argv[++i]);
			loadPacketsFromFile(path, &packets, max_packets);
		}
		else if (arg == "--packets" || arg == "-p")
		{
			int packet_num = stoi(argv[++i]);
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
			BUCKET_COUNT = stoi(argv[++i]);
		}
		else if (arg == "--branching_factor")
		{
			BRANCHING_FACTOR = stoi(argv[++i]);
		}
		else if (arg == "--width")
		{
			CM_WIDTH = stoi(argv[++i]);
		}
		else if (arg == "--depth")
		{
			CM_DEPTH = stoi(argv[++i]);
		}
		else if (arg == "--repeat" || arg == "--once")
		{
			bool is_repeat = "--repeat" == arg;
			std::string action_name = argv[++i];
			int packets_per_action = stoi(argv[++i]);
			ActionTimer action_timer = ActionTimer(action_name, packets_per_action, is_repeat);
            if (action_name == "expand" || action_name == "shrink" | action_name == "compress")
			{
				int _argument = stoi(argv[++i]);
				action_timer.argument = _argument;
			}
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
    std::string cmd = "--limit_file /home/dbiton/Desktop/Projects/DynamicSketch/pcaps/capture.txt 1000 --type dynamic --width 500 --depth 5 --repeat log_average_relative_error 62 --repeat log_memory_usage 62 --once expand 100 1000 --once expand 200 2000 --once expand 300 4000 --once expand 400 8000 --once shrink 600 8000 --once shrink 700 4000 --once shrink 800 2000 --once shrink 900 1000";
    std::vector<const char *> args;
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
	/*for (size_t i = 0; i < args.size(); i++){
		 delete[] args[i];
	}*/
}

int main(int argc, const char *argv[])
{
    manual_argument();
    //proccess_input(argc, argv);
}
