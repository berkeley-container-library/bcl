#include <cstdlib>
#include <cstdio>
#include "kmer_t.hpp"

#include <vector>

FILE * OpenDebugFile(char * prefix, FILE * pFile, int myrank);
FILE * UFXInitOpen(const char * filename, int * dsize, int64_t * myshare, int nprocs, int myrank, int64_t *nEntries);
int64_t UFXRead(FILE * f, int dsize, char *** kmersarr, int ** counts, char ** lefts, char ** rights, int64_t requestedkmers, int dmin, int reuse, int myrank, std::vector <kmer_pair_> &kmers);
void DeAllocateAll(char *** kmersarr, int ** counts, char ** lefts, char ** rights, int64_t initialread);
