#include <iostream>
#include <cstring>
#include <string>
#include <sstream>
#include <vector>
#include <algorithm>
#include <sys/types.h>
#include <sys/stat.h>
#include "kmer_t.hpp"
#include <unistd.h>

#include "Friends.h"
#include "KmerMiddle.hpp"
#include "readufx.h"
using namespace std;


#ifndef KMERlength
        #define KMERlength 51
#endif


FILE * OpenDebugFile(char * prefix, FILE * pFile, int myrank)
{
    stringstream ss;
    string rank;
    ss << myrank;
    ss >> rank;
    string ofilename = prefix;
    ofilename += rank;
    pFile  = fopen(ofilename.c_str(), "w");
    return pFile;
}

// input parameters: filename, dsize, nprocs, myrank
// output: myshare, dsize
FILE * UFXInitOpen(const char * filename, int * dsize, int64_t *myshare, int nprocs, int myrank, int64_t *nEntries)
{
	KmerMiddle::set_k(KMERlength);
	Kmer::set_k(KMERlength);
    *dsize = sizeof(ufxpack);

    struct stat st;
    stat(filename, &st);
    int64_t filesize = st.st_size;
    if(myrank  == 0) cout << "Filesize is " << filesize << " bytes" << endl;
    int64_t numentries = filesize / static_cast<int64_t>(*dsize );
   (*nEntries) = numentries;
    if(myrank == 0) cout << "Number of records is " << numentries << endl;

	FILE * f = fopen(filename, "r");

	int64_t perproc = numentries / nprocs;
	int64_t begin = perproc * myrank;
	if(myrank == nprocs-1)
		*myshare = numentries - (nprocs-1)* (perproc);
	else
		*myshare = perproc;

	fseek (f, begin * static_cast<int64_t>(*dsize ), SEEK_SET );
	return f;
}


// inputs: f, dsize, requestedkmers, dmin
// outputs: kmersarr, counts, lefts, rights
// returns: number of k-mers read (can be less than requestedkmers if end of file)
int64_t UFXRead(FILE * f, int dsize, char *** kmersarr, int ** counts,
                char ** lefts, char ** rights, int64_t requestedkmers,
                int dmin, int reuse, int myrank, std::vector <kmer_pair_> &kmers)
{
   int64_t i;

	if(!f){
		cerr << "Problem reading binary input file\n";
		return 1;
	}

	ufxpack * upack = new ufxpack[requestedkmers];
	int64_t totread = fread(upack, dsize, requestedkmers, f);

    if(!reuse)  // OK in the last iteration too because the invariant (totread <= requestedkmers) holds
    {
        // (*kmersarr) is of type char**
        (*kmersarr) = (char**) malloc(sizeof(char*) * totread);
        for (i = 0; i < totread; i++)
            (*kmersarr)[i] = (char*) malloc((KMERlength+1) * sizeof(char)); // extra character for NULL termination

        *counts = (int*) malloc(sizeof(int) * totread);
        *lefts = (char*) malloc(sizeof(char) * totread);
        *rights = (char*) malloc(sizeof(char) * totread);
    }

	for(i=0; i< totread; ++i)
	{
		KmerMiddle kmer(upack[i].arr);

        // from C++11 standard:
        // 21.4.7.1 says that the pointer returned by c_str() must point to a buffer of length size()+1.
        // 21.4.5 says that the last element of this buffer must have a value of charT() -- in other words, the null character
		std::strcpy ((*kmersarr)[i], kmer.toString().c_str()); // (*kmersarr)[i] is of type char*
		(*counts)[i] = upack[i].count;
		if(upack[i].leftmax < dmin)
			(*lefts)[i] = 'X';
		else if(upack[i].leftmin < dmin)
			(*lefts)[i] = upack[i].left;
		else	// both dmin < leftmin < leftmax
			(*lefts)[i] = 'F';

		if(upack[i].rightmax < dmin)
			(*rights)[i] = 'X';
		else if(upack[i].rightmin < dmin)
			(*rights)[i] = upack[i].right;
		else	// both dmin < rightmin < rightmax
			(*rights)[i] = 'F';
    // printf("%s %c%c\n", (*kmersarr)[i], (*lefts)[i], (*rights)[i]);
    // kmers.push_back(kmer_pair_(std::string((*kmersarr)[i]), std::string() + (*lefts)[i] + (*rights)[i]));
    kmer_pair_ pair(std::string((*kmersarr)[i]), std::string() + (*lefts)[i] + (*rights)[i]);

    kmers.push_back(pair);

        // fprintf(stderr, "K-mer is named (%c) %s (%c) with count %d\n", (*lefts)[i], (*kmersarr)[i], (*rights)[i] , (*counts)[i]);
	}
    delete [] upack;
	return totread;
}

void DeAllocateAll(char *** kmersarr, int ** counts, char ** lefts, char ** rights, int64_t initialread)
{
   int64_t i;
    for (i = 0; i < initialread; i++)
        free((*kmersarr)[i]);
    free(*kmersarr);
    free(*counts);
    free(*lefts);
    free(*rights);
}
