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
#include "countminexpanding.h"

/************************************************************************/
/* Routines to support Count-Min sketches                               */
/************************************************************************/

CME_type* CME_Init(int width, int depth, int seed)
{     // Initialize the sketch based on user-supplied size
    CME_type* cm;
    int j;
    prng_type* prng;

    cm = (CME_type*)malloc(sizeof(CME_type));
    prng = prng_Init(-abs(seed), 2);
    // initialize the generator to pick the hash functions

    if (cm && prng)
    {
        cm->depth = depth;
        cm->width = width;
        cm->count = 0;
        cm->hasha = (unsigned int*)calloc(sizeof(unsigned int), cm->depth);
        cm->hashb = (unsigned int*)calloc(sizeof(unsigned int), cm->depth);
        cm->chunks = CMChunk_Init(width, depth);
        if (cm->chunks && cm->hasha && cm->hashb)
        {
            for (j = 0; j < depth; j++)
            {
                cm->hasha[j] = prng_int(prng) & MOD;
                cm->hashb[j] = prng_int(prng) & MOD;
            }
        }
        else cm = NULL;
    }
    return cm;
}

void CME_Destroy(CME_type* cm)
{     // get rid of a sketch and free up the space
    if (!cm) return;
    CMChunk_Destroy(cm->chunks);
    if (cm->hasha) free(cm->hasha); cm->hasha = NULL;
    if (cm->hashb) free(cm->hashb); cm->hashb = NULL;
    free(cm);  cm = NULL;
}

int CME_Size(CME_type* cm)
{ // return the size of the sketch in bytes
    int counts, hashes, admin;
    if (!cm) return 0;
    admin = sizeof(CME_type);
    counts = CMChunk_Size(cm->chunks) * cm->depth;
    hashes = cm->depth * 2 * sizeof(unsigned int);
    return(admin + hashes + counts);
}

bool CME_Expand(CME_type* cm, unsigned int width)
{
    CMChunk_type* chunk = CMChunk_Init(width, cm->depth);
    if (!chunk) {
        return false;
    }
    CMChunk_Append(cm->chunks, chunk);
    cm->width += width;
    return true;
}

void CME_Update(CME_type* cm, unsigned int item, int diff)
{
    if (!cm) return;
    cm->count += diff;

    for (int j = 0; j < cm->depth; j++) {
        int k = hash31(cm->hasha[j], cm->hashb[j], item);
        CMChunk_Update(cm->chunks, j, k, diff, 0);
    }
}

int CME_PointEst(CME_type* cm, unsigned int query)
{
    if (!cm) return 0;

    // calculate hash of query for every depth
    std::vector<int> ks;
    std::vector<int> counts;
    for (int depth = 0; depth < cm->depth; depth++) {
        int k = hash31(cm->hasha[depth], cm->hashb[depth], query);
        ks.push_back(k);
        counts.push_back(0);
    }

    int width = 0;
    int width_prev;
    for (CMChunk_type* chunk = cm->chunks; chunk != NULL; chunk = chunk->next) {
        width_prev = width;
        width += chunk->width;
        for (int depth = 0; depth < cm->depth; depth++) {
            int k = ks[depth] % width;
            if (k >= width_prev) {
                int k_relative = k - width_prev;
                //printf("Q: depth: %d, k: %d, width: %d, width_prev: %d, k_relative: %d\n", depth, k, width, width_prev, k_relative);
                counts[depth] += chunk->counts[depth][k_relative];
            }
        }
    }
    return *std::min_element(counts.begin(), counts.end());
}

/************************************************************************/
/* Routines to support Count-Min Extensions                             */
/************************************************************************/

CMChunk_type* CMChunk_Init(int width, int depth)
{
    CMChunk_type* chunk;
    int j;

    chunk = (CMChunk_type*)malloc(sizeof(CMChunk_type));
    if (!chunk) {
        return NULL;
    }

    chunk->next = NULL;
    chunk->width = width;
    chunk->counts = (int**)calloc(sizeof(int*), depth);
    if (!chunk->counts) {
        free(chunk);
        return NULL;
    }
    chunk->counts[0] = (int*)calloc(sizeof(int), depth * width);
    if (!chunk->counts[0]) {
        free(chunk->counts);
        free(chunk);
        return NULL;
    }
    for (j = 0; j < depth; j++)
    {
        chunk->counts[j] = (int*)chunk->counts[0] + (j * width);
    }
    return chunk;
}

void CMChunk_Destroy(CMChunk_type* root)
{
    if (root == NULL) return;

    if (root->counts[0]) free(root->counts[0]);
    free(root->counts);

    CMChunk_Destroy(root->next);
}

int CMChunk_Size(CMChunk_type* root) {
    if (root == NULL) return 0;
    int counts = root->width * sizeof(int);
    return counts + CMChunk_Size(root->next);
}

void CMChunk_Append(CMChunk_type* root, CMChunk_type* leaf)
{
    CMChunk_type* node = root;
    CMChunk_type* node_next;
    while (1) {
        node_next = node->next;
        if (node_next == NULL) {
            node->next = leaf;
            break;
        }
        else {
            node = node_next;
        }
    }
}

bool CMChunk_Update(CMChunk_type* root, int j, int k, int diff, int width_prev)
{
    if (root != NULL) {
        bool updated = CMChunk_Update(root->next, j, k, diff, width_prev + root->width);
        if (updated) {
            return true;
        }
        else {
            int width = width_prev + root->width;
            int k_modulo = k % width;
            if (k_modulo >= width_prev) {
                int k_relative = k_modulo - width_prev;
                //printf("U: depth: %d, k: %d, width: %d, width_prev: %d, k_relative: %d\n", j, k, width, width_prev, k_relative);
                root->counts[j][k_relative] += diff;
                return true;
            }
            else {
                return false;
            }
        }
    }
    else {
        return false;
    }
}