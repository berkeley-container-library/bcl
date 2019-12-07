#ifndef ARH_HASH_MAP_HPP
#define ARH_HASH_MAP_HPP

#include "external/libcuckoo/cuckoohash_map.hh"

namespace ARH {

  template <
      typename Key,
      typename Val,
      typename Hash = std::hash<Key>
  >
  class HashMap {
  private:
    using local_map_t = libcuckoo::cuckoohash_map<Key, Val, Hash>;
    std::vector<local_map_t*> map_ptrs;
    size_t my_size;
    size_t total_size;
    // map the key to a target process
    int get_target_proc(const Key &key) {
      return uint64_t(Hash{}(key) % total_size) % (total_size / my_size);
    }
  public:
    // initialize the local map
    HashMap(size_t size) {
      my_size = (size + nprocs() - 1) / nprocs();
      total_size = my_size * nprocs();

      map_ptrs.resize(nprocs());
      if (my_worker_local() == 0) {
        map_ptrs[my_proc()] = new local_map_t();
        map_ptrs[my_proc()]->reserve(my_size);
      }
      for (size_t i = 0; i < nprocs(); ++i) {
        broadcast(map_ptrs[i], i * nworkers_local());
      }
    }

    ~HashMap() {
      ARH::barrier();
      if (my_worker_local() == 0) {
        delete map_ptrs[my_proc()];
      }
    }

    Future<bool> insert(Key key, Val val) {
      size_t target_proc = get_target_proc(key);
      return rpc(target_proc * nworkers_local(),
                        [](local_map_t* lmap, Key key, Val val){
                          lmap->insert(key, val);
                          return true; // hashmap will never be full
                        }, map_ptrs[target_proc], key, val);
    }

    Future<Val> find(Key key){
      size_t target_proc = get_target_proc(key);
      return rpc(target_proc * nworkers_local(),
                        [](local_map_t* lmap, Key key)
                            -> Val {
                            Val out;
                          if (lmap->find(key, out)) {
//                            printf("Node %lu find {%lu, %lu}\n", my_proc(), key, out);
                            return out;
                          } else {
//                            printf("Worker %lu cannot find %lu\n", my_worker(), key.hash());
                            return Val();
                          }
                        }, map_ptrs[target_proc], key);
    }

    void insert_ff(Key key, Val val) {
      size_t target_proc = get_target_proc(key);
      rpc_ff(target_proc * nworkers_local(),
             [](local_map_t* lmap, Key key, Val val){
                 lmap->insert(key, val);
             }, map_ptrs[target_proc], key, val);
    }

    Future<Val> find_agg(Key key){
      size_t target_proc = get_target_proc(key);
      return rpc_agg(target_proc * nworkers_local(),
                 [](local_map_t* lmap, Key key)
                         -> Val {
                     Val out;
                     if (lmap->find(key, out)) {
//                            printf("Node %lu find {%lu, %lu}\n", my_proc(), key, out);
                       return out;
                     } else {
//                            printf("Worker %lu cannot find %lu\n", my_worker(), key.hash());
                       return Val();
                     }
                 }, map_ptrs[target_proc], key);
    }

  };
}

#endif //ARH_HASH_MAP_HPP
