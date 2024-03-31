// Two different structures: 
//   1 -- The basic CM Sketch
//   2 -- The hierarchical CM Sketch: with log n levels, for range sums etc. 

#ifndef COUNTMIN_h
#define COUNTMIN_h

#include "prng.h"

typedef struct CM_type{
  int64_t count;
  uint64_t seed;
  int depth;
  int width;
  int ** counts;
  unsigned int *hasha, *hashb;
} CM_type;

extern CM_type * CM_Init(int, int, int);
extern CM_type * CM_Copy(CM_type *);
extern void CM_Destroy(CM_type *);
extern int CM_Size(CM_type *);

extern int CM_Compatible(CM_type* cm1, CM_type* cm2);
extern void CM_HalveCounts(CM_type*);
extern void CM_Clear(CM_type*);
extern bool CM_Merge(CM_type*, CM_type*);
extern void CM_Update(CM_type *, unsigned int, int); 
extern int CM_PointEst(CM_type *, unsigned int);
extern int CM_PointMed(CM_type *, unsigned int);
extern int64_t CM_InnerProd(CM_type *, CM_type *);
extern int CM_Residue(CM_type *, unsigned int *);
extern int64_t CM_F2Est(CM_type *);

#endif
