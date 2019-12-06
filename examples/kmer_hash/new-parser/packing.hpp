#pragma once

#include <cassert>

#define KMER_LEN 51
#define PACKED_KMER_LEN ((KMER_LEN+3)/4)

bool packedCodeToFourMerCoded = false;
unsigned int packedCodeToFourMer[256];

#define pow4(a) (1<<((a)<<1))

void init_LookupTable()
{
  // Work with 4-mers for the moment to have small lookup tables
  int merLen = 4, i, slot, valInSlot;
  unsigned char mer[4];

  for ( i = 0; i < 256; i++ ) {
    // convert a packedcode to a 4-mer
    int remainder = i;
    int pos = 0;
    for( slot = merLen-1; slot >= 0; slot-- ) {
      valInSlot = remainder / pow4(slot);
      char base;

      if (valInSlot == 0) { base = 'A'; }
        else if( valInSlot == 1 ) { base = 'C'; }
        else if( valInSlot == 2 ) { base = 'G'; }
        else if( valInSlot == 3 ) { base = 'T'; }
        else{ assert( 0 ); }

      mer[pos] = base;
      pos++;
      remainder -= valInSlot * pow4(slot);
    }
    unsigned int *merAsUInt = (unsigned int*) mer;
    packedCodeToFourMer[i] = (unsigned int) (*merAsUInt);
  }
}

unsigned char packFourMer(const char *fourMer)
{
  int retval = 0;
  int code, i;
  int pow = 64;

  for ( i=0; i < 4; i++) {
    char base = fourMer[i];
    switch ( base ) {
      case 'A':
        code = 0;
        break;
      case 'C':
        code = 1;
        break;
      case 'G':
        code = 2;
        break;
      case 'T':
        code = 3;
        break;
      }
      retval += code * pow;
      pow /= 4;
  }
  return ((unsigned char) retval);
}

void packKmer(const char *kmer, unsigned char *packed_kmer) {
  int ind, j = 0;
  int i = 0;

  for ( ; j <= KMER_LEN - 4; i++, j += 4) {
    packed_kmer[i] = packFourMer(kmer + j);
  }

  int remainder = KMER_LEN % 4;
  char blockSeq[5] = "AAAA";
  for (ind = 0; ind < remainder; ind++) {
    blockSeq[ind] = kmer[j + ind];
  }

  packed_kmer[i] = packFourMer(blockSeq);
}


void unpackKmer(const unsigned char packed_kmer[PACKED_KMER_LEN],
  char *kmer) {
  if (!packedCodeToFourMerCoded) {
    packedCodeToFourMerCoded = true;
    init_LookupTable();
  }
  int i = 0, j = 0;
  for( ; i < PACKED_KMER_LEN; i++, j += 4 ) {
    unsigned char block[4];
    *(unsigned int *) block = packedCodeToFourMer[packed_kmer[i]];
    for (int i = 0; i < 4; i++) {
      (kmer + j)[i] = (char) block[i];
    }
  }
}
