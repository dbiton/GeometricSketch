#include <unordered_map>
#include <time.h>
#include <chrono>
#include <iostream>
#include <string>

#include "DynamicSketch.h"
#include "NaiveSketch.h"
#include "ElasticSketch.h"
#include "Dictionary.h"

constexpr float epsilon = 0.1;
constexpr float delta = 0.1;
constexpr int SEED = 0x1337C0D3;
constexpr int NUM_PACKETS = 1024 * 1024 * 32;
const int CM_WIDTH = ceil(exp(1) / epsilon);
const int CM_DEPTH = ceil(log(1 / delta));

// --file path-- type dynamic-- repeat[command num][modify_size_random | modify_size_cyclic | expand | shrink | log_memory_usage | log_size | log_time][times]-- time
// --packets [num] [packet0] ...

struct ActionTimer
{
	int packets_per_action;
	std::string action_name;
	// additional info for modify_size_cyclic and modify_size_random
	int max_size;
	int min_size;

	ActionTimer(std::string _action_name, int _packets_per_action) : action_name(_action_name), packets_per_action(_packets_per_action) {}
	void setSizeLimits(int _max_size, int _min_size)
	{
		max_size = _max_size;
		min_size = _min_size;
	}
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
	if (type == "map")
	{
		return new MapDictionary();
	}
	else if (type == "unordered_map")
	{
		return new UnorderedMapDictionary();
	}
	else if (type == "dynamic")
	{
		return new DynamicSketch(CM_WIDTH, CM_DEPTH, SEED);
	}
	else if (type == "countmin")
	{
		return new CountMinDictionary(CM_WIDTH, CM_DEPTH, SEED);
	}
	else if (type == "countsketch")
	{
		return new CountSketchDictionary(CM_WIDTH, CM_DEPTH);
	}
	else
	{
		throw std::invalid_argument("--type=" + type + "?");
	}
}

double calculateError(Dictionary *dictionary, std::vector<uint32_t> packets, int packet_index)
{
	if (packet_index == 0) return 0;
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
		delta += std::abs(dictionary_estimate - count) ;
	}
	return delta / hashtable.size();
}

void doPendingActions(Dictionary *dictionary, std::vector<uint32_t> packets, std::vector<ActionTimer> action_timers, int packet_index)
{
	for (const auto &action_timer : action_timers)
	{
		if (packet_index % action_timer.packets_per_action == 0)
		{
			std::string action_name = action_timer.action_name;
			if (action_name == "expand")
			{
				dictionary->expand();
			}
			else if (action_name == "shrink")
			{
				dictionary->shrink();
			}
			else if (action_name == "modify_size_random")
			{
				int size = dictionary->getSize();
				if (size == action_timer.max_size)
				{
					dictionary->shrink();
				}
				else if (size == action_timer.min_size)
				{
					dictionary->expand();
				}
				else
				{
					int expand = rand() % 2;
					if (expand)
					{
						dictionary->expand();
					}
					else
					{
						dictionary->shrink();
					}
				}
			}
			else if (action_name == "modify_size_cyclic")
			{
				int size = dictionary->getSize();
			}
			else if (action_name == "log_memory_usage")
			{
				std::cout << "{\"memory_usage\":" << dictionary->getMemoryUsage() <<",\"index\""<< packet_index << "}" << std::endl;
			}
			else if (action_name == "log_error")
			{
				double error = calculateError(dictionary, packets, packet_index);
				std::cout << "{\"log_error\":" << error << ",\"index\":"  << packet_index << "}" << std::endl;
			}
			else if (action_name == "log_size")
			{
				int size = dictionary->getSize();
				std::cout << "{\"log_size\":" << size << ",\"index\":" << packet_index << "}" << std::endl;
			}
			else {
				throw std::invalid_argument(action_name + "?");
			}
		}
	}
}

void run(Dictionary *dictionary, const std::vector<uint32_t>& packets, std::vector<ActionTimer> action_timers)
{
	for (int i = 0; i < packets.size(); i++)
	{
		doPendingActions(dictionary, packets, action_timers, i);
		uint32_t packet = packets[i];
		dictionary->update(packet, 1);
	}
}

void proccess_input(int argc, const char *argv[])
{
	Dictionary* dictionary = nullptr;
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
			int packet_num = stoi(argv[i++]);
			loadPacketsFromArgs(argv, i, packet_num, &packets);
		}
		else if (arg == "--type" || arg == "-t")
		{
			std::string type = argv[++i];
			if (dictionary)
			{
				throw std::invalid_argument("only one --type allowed");
			}
			dictionary = createDictionary(type);
		}
		else if (arg == "--repeat" || arg == "-r")
		{
			int packets_per_action = stoi(argv[++i]);
			std::string action_name = argv[++i];
			ActionTimer action_timer = ActionTimer(action_name, packets_per_action);
			if (action_name == "modify_size_cyclic" || action_name == "modify_size_random")
			{
				int min_size = stoi(argv[++i]);
				int max_size = stoi(argv[++i]);
				action_timer.setSizeLimits(min_size, max_size);
			}
			action_timers.push_back(action_timer);
		}
		else
		{
			throw std::invalid_argument(arg + "?");
		}
	}
	run(dictionary, packets, action_timers);
}


int manual_argument() {
	std::string cmd = "--limit_file C:/Users/USER2/Desktop/projects/DynamicSketch/pcaps/capture.txt 1000000 --type countmin --repeat 1000 log_error";

	std::vector<const char*> args;
	std::istringstream iss(cmd);

	args.push_back("");
	std::string token;
	while (iss >> token) {
		char* arg = new char[token.size() + 1];
		copy(token.begin(), token.end(), arg);
		arg[token.size()] = '\0';
		args.push_back(arg);
	}
	args.push_back(0);

	proccess_input(args.size(), &args[0]);
	for (size_t i = 0; i < args.size(); i++)
		delete[] args[i];
	return 0;
}


int main(int argc, const char* argv[]) {
	//manual_argument();
	proccess_input(argc, argv);
}