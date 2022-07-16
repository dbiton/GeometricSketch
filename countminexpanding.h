#pragma once

#define COUNTMINEXPANDING_h

#include "prng.h"

typedef struct CMChunk_type {
	int width;
	int** counts;
	CMChunk_type* next;
} CMChunk_type;

extern CMChunk_type* CMChunk_Init(int, int);
extern void CMChunk_Destroy(CMChunk_type*);
extern int CMChunk_Size(CMChunk_type*);
extern void CMChunk_Append(CMChunk_type*, CMChunk_type*);
extern bool CMChunk_Update(CMChunk_type*, int, int, int, int);


typedef struct CME_type {
	int64_t count;
	int depth;
	int width;
	unsigned int* hasha, * hashb;
	CMChunk_type* chunks;
} CME_type;

extern CME_type* CME_Init(int, int, int);
extern CME_type* CME_Copy(CME_type*);
extern void CME_Destroy(CME_type*);
extern int CME_Size(CME_type*);
extern bool CME_Expand(CME_type*, unsigned int);

extern void CME_Update(CME_type*, unsigned int, int);
extern int CME_PointEst(CME_type*, unsigned int);