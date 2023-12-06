#pragma once

/*
 * Copyright 2015 Gianluca Tiepolo <tiepolo.gian@gmail.com>
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 * CountSketch.h
 *
 *  Created on: Jul 7, 2016
 *      Author: Gianluca Tiepolo <tiepolo.gian@gmail.com>
 */

#ifndef COUNTSKETCH_H_
#define COUNTSKETCH_H_

#include "Murmurhash.h"
#include <vector>
#include <time.h>
#include <stdlib.h>
#include <string>
#include <functional>
#include <algorithm>
#include "util.h"

/*
 * Implementation of the Count Sketch algorithm
 * based on <http://dimacs.rutgers.edu/~graham/pubs/papers/freqvldbj.pdf>
 */
class CountSketch {
	typedef std::vector<int> Matrice;
public:
	/*
	 * Constuctor
	 * params are epsilon (between 0.01 and 1) and gamma (between 0.1 and 1)
	 */
    CountSketch(double epsilon, double gamma, int seed=0)
	{
		// calculate depth and width based on epsilon and gamma
		d = ceil(log(4 / gamma));
		w = ceil(1 / pow(epsilon, 2));
		row_counters = std::vector<int>(d, 0);
		// create Matrix
		//srand(time(NULL)); // seed time to rand so we get some random numbers
        srand(seed); // use 0 seed so that we get the same seeds - allowing merge
		for (unsigned int i = 0; i < d; ++i) {
			C.push_back(Matrice(w)); // create 'w' columns for each 'd' row
			seeds.push_back(rand()); // add random number to first seed function
			sign_seeds.push_back(rand()); // add random number to second seed function
		}
	}

    CountSketch(unsigned int width, unsigned int depth, int seed=0) {
		// calculate depth and width based on epsilon and gamma
		d = depth;
		w = width;
		row_counters = std::vector<int>(d, 0);
		// create Matrix
        srand(seed); // use 0 seed so that we get the same seeds - allowing merge
		for (unsigned int i = 0; i < d; ++i) {
			C.push_back(Matrice(w)); // create 'w' columns for each 'd' row
			seeds.push_back(rand()); // add random number to first seed function
			sign_seeds.push_back(rand()); // add random number to second seed function
		}
	}

	CountSketch* split() {
		CountSketch* sketch = new CountSketch(w, d);
		bool flipflop = false;
		auto v = row_counters;
		for (auto i : sort_indexes(v)) {
			flipflop = !flipflop;
			if (flipflop) {
				sketch->row_counters[i] = row_counters[i];
				row_counters[i] = 0;
				for (unsigned int j = 0; j < w; ++j) {
					sketch->C[i][j] = C[i][j];
					C[i][j] = 0;
				}
			}
		}
		return sketch;
	}

	/*
	*  Merge sketchs assuming same d, w, seeds, sign_seeds
	*/
	void merge(CountSketch& sketch) {
		for (unsigned int i = 0; i < d; ++i) {
			for (unsigned int j = 0; j < w; ++j) {
				C[i][j] += sketch.C[i][j];
			}
			row_counters[i] += sketch.row_counters[i];
		}
	}

	/*
	 * Add an int to the counter
	 */
	void addInt(int item) {
		for (unsigned int i = 0; i < d; ++i) {
			row_counters[i]++;
			// use value from seeds vector to seed the hashing function and create hash
			int p = murmurhash(&item, seeds[i]) % w;
			// use value from second seed vector (-1/+1)
			int sign = murmurhash(&item, sign_seeds[i]) % 2;
			// C = C + cg - update value
			C[i][p] += (sign * 2 - 1) * 1;
		}
	}

	/*
	 * Get total count of additions
	 */
	int getTotalCount() {
		int sum = 0;
		for (int i = 0; i < d; i++) sum += row_counters[i];
		return sum;
	}

	bool isBalanced() {
		auto min_max = std::minmax_element(row_counters.begin(), row_counters.end());
		return (*min_max.first)*2 >= *min_max.second;
	}

	/*
	 * Get the frequency of a string
	 */
	double getStringFrequency(std::string s) {
		int item = hasher(s);
		return getIntFrequency(item);
	}

	/*
	 * Get the frequency of an int
	 */
	double getIntFrequency(int item) {
		int max_counter = *std::max_element(row_counters.begin(), row_counters.end());
		std::vector<double> values;
		for (unsigned int i = 0; i < d; ++i) {
			if (row_counters[i] < max_counter / 2) continue;
			int p = murmurhash(&item, seeds[i]) % w;
			int sign = murmurhash(&item, sign_seeds[i]) % 2;
			values.push_back((sign * 2 - 1) * C[i][p]);
		}
		// return the median (4.3.2 The median trick, "ESTIMATING THE NUMBER OF DISTINCT ELEMENTS", page 18)
		if (values.size() > 1) {
			auto m = values.begin() + values.size() / 2;
			std::nth_element(values.begin(), m, values.end());
			double res = values[values.size() / 2];
			return res;
		}
		else return values[0];
	}

	/*
	 * Virtual destructor
	 */
	virtual ~CountSketch(){};
private:
	// depth and width of the matrix
	unsigned int d, w;
	// matrix
	std::vector<Matrice> C;
	// vector of seeds
	std::vector<int> seeds;
	// 2nd vector of seeds (-1/+1)
	std::vector<int> sign_seeds;
	// string to int hasher
	std::hash<std::string> hasher;
	std::vector<int> row_counters;
};

#endif /* COUNTSKETCH_H_ */
