#ifndef _FRIENDS_H_
#define _FRIENDS_H_

#include <iostream>
#include <vector>
#include <string>
#include <algorithm>
#include <fstream>
#include <sstream>
#include "Kmer.hpp"
#include "KmerIterator.hpp"
#include "Deleter.h"

#include <sys/stat.h>

using namespace std;

#ifndef MAX_KMER_SIZE
#define MAX_KMER_SIZE 64
#endif
#define KMERLONGS MAX_KMER_SIZE/32	// 32 = numbits(uint64_t)/2-  with 2 being the number of bits needed per nucleotide

typedef array<uint64_t, KMERLONGS> MERARR;


struct filedata
{
    char filename[256];
    size_t filesize;
};


ostream & operator<<(ostream & os, uint8_t val)
{
    return os << static_cast<int>(val);
}

struct kmerpack	// the pair<MERARR,int> used as value_type in map is not guaranteed to be contiguous in memory
{
	MERARR arr;
	int count;

	bool operator > (const kmerpack & rhs) const
	{ return (arr > rhs.arr); }
	bool operator < (const kmerpack & rhs) const
	{ return (arr < rhs.arr); }
	bool operator == (const kmerpack & rhs) const
	{ return (arr == rhs.arr); }
};


struct ufxpack	// 38bytes for k=51
{
	MERARR arr;	// ~128-bits=16bytes for k=51
	int count;
	char left;
	char right;
	int leftmin;
	int leftmax;
	int rightmin;
	int rightmax;

	bool operator > (const ufxpack & rhs) const
	{ return (arr > rhs.arr); }
	bool operator < (const ufxpack & rhs) const
	{ return (arr < rhs.arr); }
	bool operator == (const ufxpack & rhs) const
	{ return (arr == rhs.arr); }
};

void PackIntoUFX(array<int,4> & leftcnt, array<int,4> & righcnt, int count, ufxpack & pack)
{
	pair<int, char> lsort[4] = {make_pair(leftcnt[0], 'A'), make_pair(leftcnt[1], 'C'), make_pair(leftcnt[2], 'G'), make_pair(leftcnt[3], 'T')};
	pair<int, char> rsort[4] = {make_pair(righcnt[0], 'A'), make_pair(righcnt[1], 'C'), make_pair(righcnt[2], 'G'), make_pair(righcnt[3], 'T')};
	sort(lsort, lsort+4);
	sort(rsort, rsort+4);

	pack.left = lsort[3].second;	// max entry guarenteed to exist
	pack.leftmax = lsort[3].first;
	pack.leftmin = lsort[2].first;

	pack.right = rsort[3].second;
	pack.rightmax = rsort[3].first;
	pack.rightmin = rsort[2].first;

	pack.count = count;
}


struct SNPdata
{
	MERARR karr;
	char extA;
	char extB;

	bool operator > (const SNPdata & rhs) const
	{ return (karr > rhs.karr); }
	bool operator < (const SNPdata & rhs) const
	{ return (karr < rhs.karr); }
	bool operator == (const SNPdata & rhs) const
	{ return (karr == rhs.karr); }
};

#endif
