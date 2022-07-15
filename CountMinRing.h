#pragma once

#include "countmin.h"
#include <map>


class CountMinRing
{
	std::vector<CM_type> sketchs;
public:
	CountMinRing(double epsilon, double gamma);

	void increment(int key);
	int query(int key);

	void expand();
	void shrink();
	int size() const;
private:

};

