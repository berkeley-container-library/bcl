#ifndef BFG_KMER_HPP
#define BFG_KMER_HPP

#ifndef MAX_KMER_SIZE
 #define MAX_KMER_SIZE 64	// ABAB: This code will probably crush if this is not a multiple of 32
#endif

#include <stdio.h>
#include <stdint.h>
#include <cassert>
#include <cstring>
#include <string>
#include <array>

#include "hash.hpp"



/* Short description:
 *  - Store kmer strings by using 2 bits per base instead of 8
 *  - Easily return reverse complements of kmers, e.g. TTGG -> CCAA
 *  - Easily compare kmers
 *  - Provide hash of kmers
 *  - Get last and next kmer, e.g. ACGT -> CGTT or ACGT -> AACGT
 *  */
class Kmer {
 public:

  Kmer();
  Kmer(const Kmer& o);
  explicit Kmer(const char *s);
  void copyDataFrom(uint8_t *  mybytes)	// this is like a shadow constructor (to avoid accidental signature match with the existing constructor)
  {
	  memcpy(longs, mybytes, sizeof(uint64_t) * (MAX_K/32));
  }
  explicit Kmer(const std::array<uint64_t, MAX_KMER_SIZE/32> & arr)
  {
	std::memcpy (longs, arr.data(), sizeof(uint64_t) * (MAX_K/32));
  }

  Kmer& operator=(const Kmer& o);
  void set_deleted();
  bool operator<(const Kmer& o) const;
  bool operator==(const Kmer& o) const;
  bool operator!=(const Kmer& o) const {
    return !(*this == o);
  }

  Kmer(const std::string& str) {
    set_kmer(str.c_str());
  }

  void set_kmer(const char *s);
  uint64_t hash() const;

  Kmer twin() const;
  Kmer rep() const; // ABAB: return the smaller of itself (lexicographically) or its reversed-complement (i.e. twin)
  Kmer getLink(const size_t index) const;
  Kmer forwardBase(const char b) const;
  Kmer backwardBase(const char b) const;
  std::string getBinary() const;
  void toString(char * s) const;
  std::string toString() const;

  void copyDataInto(void * pointer) const
  {
	// void * memcpy ( void * destination, const void * source, size_t num );
	  memcpy(pointer, longs, sizeof(uint64_t) * (MAX_K/32));
  }

  // ABAB: return the raw data packed in an std::array
  // this preserves the lexicographical order on k-mers
  // i.e. A.toString() < B.toString <=> A.getArray() < B.getArray()
  std::array<uint64_t, MAX_KMER_SIZE/32> getArray()
  {
	std::array<uint64_t,MAX_K/32> i64array;
	std::memcpy (i64array.data(),longs,sizeof(uint64_t) * (MAX_K/32));
	return i64array;
  }

  bool equalUpToLastBase(const Kmer & rhs);	// returns true for completely identical k-mers as well as k-mers that only differ at the last base

  // static functions
  static void set_k(unsigned int _k);
  static constexpr size_t numBytes() {
	  return (sizeof(uint64_t) * (MAX_K/32));
  }


  static const unsigned int MAX_K = MAX_KMER_SIZE;
  static unsigned int k;

 private:
  static unsigned int k_bytes;
  static unsigned int k_longs;
  static unsigned int k_modmask; // int?

  // data fields
  union {
    uint8_t bytes[MAX_K/4];
    uint64_t longs[MAX_K/32];
  };

  // Unions are very useful for low-level programming tasks that involve writing to the same memory area
  // but at different portions of the allocated memory space, for instance:
  //		union item {
  //			// The item is 16-bits
  //			short theItem;
  //			// In little-endian lo accesses the low 8-bits -
  //			// hi, the upper 8-bits
  //			struct { char lo; char hi; } portions;
  //		};
  //  item tItem;
  //  tItem.theItem = 0xBEAD;
  //  tItem.portions.lo = 0xEF; // The item now equals 0xBEEF


 // void shiftForward(int shift);
 // void shiftBackward(int shift);
};

struct KmerHash {
  size_t operator()(const Kmer &km) const {
    return km.hash();
  }
};


namespace std {	// to use in unordered_map
  template <>
  struct hash<Kmer>
  {
    uint64_t operator()(const Kmer& k) const
    {
	return k.hash();
    }
  };
}


#endif // BFG_KMER_HPP
