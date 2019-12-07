#pragma once

#include <map>
#include <functional>
#include <upcxx/upcxx.hpp>
#include "kmer_t.hpp"

class HashMap;
namespace std {
  template<> class hash<pkmer_t> {
  public:
    uint64_t operator()(const pkmer_t &kmer) const {
      return kmer.hash();
    }
  };
}

class HashMap {
private:
    // store the local unordered map in a distributed object to access from RPCs
    using dobj_map_t = upcxx::dist_object<std::unordered_map<pkmer_t,kmer_pair>>;
    dobj_map_t local_map;
    size_t my_size;
    size_t total_size;
    // map the key to a target process
    int get_target_rank(const pkmer_t &kmer) {
        return uint64_t(kmer.hash() % total_size) / uint64_t(my_size);
    }
public:
    // initialize the local map
    HashMap(size_t size) : local_map({}) {
        my_size = (size + upcxx::rank_n() - 1) / upcxx::rank_n();
        total_size = my_size * upcxx::rank_n();
    }
    // insert a key-value pair into the hash table
    upcxx::future<> insert(const kmer_pair &kmer) {
        // the RPC returns an empty upcxx::future by default
        return upcxx::rpc(get_target_rank(kmer.kmer),
                [](dobj_map_t &lmap, kmer_pair kmer){
                    lmap->insert({kmer.kmer, kmer});
                }, local_map, kmer);
    }
    
    upcxx::future<kmer_pair> find(const pkmer_t &key_kmer){
        return upcxx::rpc(get_target_rank(key_kmer),
                [](dobj_map_t &lmap, pkmer_t key_kmer)
                    -> kmer_pair {
                    auto elem = lmap->find(key_kmer);
                    if (elem == lmap->end()) {
                        return kmer_pair();
                    } else {
                        return elem->second;
                    }
                }, local_map, key_kmer);
    }

};