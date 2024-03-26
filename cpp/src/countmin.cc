/********************************************************************
Count-Min Sketches

G. Cormode 2003,2004

Updated: 2004-06 Added a floating point sketch and support for 
                 inner product point estimation
Initial version: 2003-12

This work is licensed under the Creative Commons
Attribution-NonCommercial License. To view a copy of this license,
visit http://creativecommons.org/licenses/by-nc/1.0/ or send a letter
to Creative Commons, 559 Nathan Abbott Way, Stanford, California
94305, USA. 
*********************************************************************/

#include <stdlib.h>
#include "prng.h"
#include "countmin.h"
#include "xxhash.h"

#define min(x,y)	((x) < (y) ? (x) : (y))
#define max(x,y)	((x) > (y) ? (x) : (y))

/************************************************************************/
/* Routines to support Count-Min sketches                               */
/************************************************************************/

CM_type* CM_Init(int width, int depth, int seed)
{     // Initialize the sketch based on user-supplied size
    CM_type* cm;

    cm = (CM_type*)malloc(sizeof(CM_type));
    if (cm)
    {
        cm->depth = depth;
        cm->width = width;
        cm->count = 0;
        cm->counts = new std::vector<int>(width * depth, 0);
    }
    return cm;
}

CM_type * CM_Copy(CM_type * cmold)
{     // create a new sketch with the same parameters as an existing one
  CM_type * cm;
  int j;

  if (!cmold) return(NULL);
  cm=(CM_type *) malloc(sizeof(CM_type));
  if (cm)
  {
      cm->depth = cmold->depth;
      cm->width = cmold->width;
      cm->count = 0;
      cm->counts = cmold->counts;
  }
  return cm;
}

void CM_Destroy(CM_type * cm)
{
  if (!cm) return;
  free(cm);  
  cm=NULL;
}

int CM_Size(CM_type * cm)
{
  int counts, admin;
  if (!cm) return 0;
  admin=sizeof(CM_type);
  counts=cm->counts->size() * sizeof(int);
  return(admin + counts);
}


void CM_Clear(CM_type* cm)
{
    int j, k;

    if (!cm) return;
    cm->count = 0;
    // can be switched with memset 
    for (j = 0; j < cm->depth; j++)
        for (k = 0; k < cm->width; k++)
            (*cm->counts)[j*cm->width+k] = 0;
    return;
}

bool CM_Merge(CM_type * cm, CM_type* cm_other)
{
    int j, k;

    if (!cm || !cm_other || !CM_Compatible(cm, cm_other)) return false;
    cm->count += cm_other->count;
    for (j = 0; j < cm->depth; j++)
        for (k = 0; k < cm->width; k++)
            cm->counts->operator[](j* cm->width + k) += cm->counts->operator[](j* cm_other->width + k);
    return true;
}

void CM_Update(CM_type * cm, unsigned int item, int diff)
{
  int j;

  if (!cm) return;
  cm->count+=diff;
  for (j = 0; j < cm->depth; j++) {
      int k = XXH32(&item, sizeof(item), j) % cm->width;
      cm->counts->operator[](j * cm->width + k) += diff;
  }
  // this can be done more efficiently if the width is a power of two
}

int CM_PointEst(CM_type * cm, unsigned int query)
{
  // return an estimate of the count of an item by taking the minimum
  int j, ans;

  if (!cm) return 0;
  int k = XXH32(&query, sizeof(query), 0) % cm->width;
  ans= (*cm->counts)[k];
  for (j = 1; j < cm->depth; j++) {
      k = XXH32(&query, sizeof(query), j) % cm->width;
      ans = min(ans, (*cm->counts)[j * cm->width + k]);
  }
  // this can be done more efficiently if the width is a power of two
  return (ans);
}

int CM_Compatible(CM_type * cm1, CM_type * cm2)
{ // test whether two sketches are comparable (have same parameters)
  int i;
  if (!cm1 || !cm2) return 0;
  if (cm1->width!=cm2->width) return 0;
  if (cm1->depth!=cm2->depth) return 0;
  return 1;
}