#ifndef ARH_AM_HPP
#define ARH_AM_HPP

namespace ARH {

  extern size_t nprocs(void);
  extern void progress(void);

  alignas(alignof_cacheline) size_t handler_num;
  struct ThreadAcknowledged {
    alignas(alignof_cacheline) std::atomic<size_t> val;
  };
  struct ThreadRequested {
    alignas(alignof_cacheline) std::atomic<size_t> val;
  };
  alignas(alignof_cacheline) std::vector<ThreadAcknowledged> acknowledgeds; // TODO: change this to thread-local
  alignas(alignof_cacheline) std::vector<ThreadRequested> requesteds; // TODO: change this to thread-local

  alignas(alignof_cacheline) gex_AM_Index_t hidx_generic_rpc_ackhandler_;
  alignas(alignof_cacheline) gex_AM_Index_t hidx_generic_rpc_reqhandler_;

  using rpc_t = ARH::rpc_t;
  using rpc_result_t = ARH::rpc_t::rpc_result_t;

  template<typename T>
  struct BufferPack {
    BufferPack(void *buf, size_t nbytes) {
      assert(nbytes % sizeof(T) == 0);
      len = nbytes / sizeof(T);
      pointer = reinterpret_cast<T *>(buf);
    }

    T &operator[](size_t index) {
      assert(index >= 0 && index < len);
      return *(pointer + index);
    }

    [[nodiscard]] size_t size() const {
      return len;
    }

  private:
    T* pointer;
    size_t len;
  };

  void generic_handler_reply_impl_(gex_Token_t token, std::vector<rpc_result_t> &&results) {
    gex_AM_ReplyMedium0(token, hidx_generic_rpc_ackhandler_, results.data(),
        results.size() * sizeof(rpc_result_t), GEX_EVENT_NOW, 0);
  }

  void generic_handler_request_impl_(size_t remote_proc, std::vector<rpc_t> &&rpcs) {
    gex_AM_RequestMedium0(BCL::tm, remote_proc, hidx_generic_rpc_reqhandler_, rpcs.data(),
    rpcs.size() * sizeof(rpc_t), GEX_EVENT_NOW, 0);
  }

  void generic_rpc_ackhandler_(gex_Token_t token, void *buf, size_t nbytes) {
    BufferPack<rpc_result_t> results(buf, nbytes);

    for (size_t i = 0; i < results.size(); ++i) {
      results[i].future_p_->payload = results[i].data_;
      results[i].future_p_->ready = true;
      acknowledgeds[results[i].source_worker_local_].val += 1;
    }
  }

  void generic_rpc_reqhandler_(gex_Token_t token, void *buf, size_t nbytes) {
    BufferPack<rpc_t> rpcs(buf, nbytes);
    std::vector<rpc_result_t> results;

    for (size_t i = 0; i < rpcs.size(); ++i) {
      size_t mContext = get_context();
      set_context((size_t) rpcs[i].target_worker_local_);
      rpc_result_t result = rpcs[i].run();
      set_context(mContext);
      results.push_back(result);
    }

    generic_handler_reply_impl_(token, std::move(results));
  }

  void init_am() {
    handler_num = GEX_AM_INDEX_BASE;

    hidx_generic_rpc_ackhandler_ = handler_num++;
    hidx_generic_rpc_reqhandler_ = handler_num++;

    gex_AM_Entry_t htable[2] = {
        { hidx_generic_rpc_ackhandler_, (gex_AM_Fn_t) generic_rpc_ackhandler_,
          GEX_FLAG_AM_MEDIUM | GEX_FLAG_AM_REPLY,   0 },
        { hidx_generic_rpc_reqhandler_, (gex_AM_Fn_t) generic_rpc_reqhandler_,
          GEX_FLAG_AM_MEDIUM | GEX_FLAG_AM_REQUEST, 0 },
    };

    gex_EP_RegisterHandlers(BCL::ep, htable, sizeof(htable)/sizeof(gex_AM_Entry_t));

    acknowledgeds = std::vector<ThreadAcknowledged>(nworkers_local());
    for (auto& t : acknowledgeds) {
      t.val = 0;
    }
    requesteds = std::vector<ThreadRequested>(nworkers_local());
    for (auto& t : requesteds) {
      t.val = 0;
    }
  }

  void flush_am() {
    while (acknowledgeds[my_worker_local()].val < requesteds[my_worker_local()].val) {
      gasnet_AMPoll();
    }
  }

  void flush_am_nopoll() {
    while (acknowledgeds[my_worker_local()].val < requesteds[my_worker_local()].val) {}
  }

  template<typename T>
  struct Future {

    Future() : data_p(new FutureData()) {}
    Future(Future&& future) = default;
    Future& operator=(Future&& future) = default;


    FutureData* get_p() const {
      return data_p.get();
    }

    T get() const {
      while (!data_p->ready) {
        progress();
      }

      if constexpr(!std::is_void<T>::value) {
        static_assert(sizeof(FutureData::payload_t) >= sizeof(T));
        return *reinterpret_cast<T*>(data_p->payload.data());
      }
    }

    [[nodiscard]] std::future_status check() const {
      if (data_p->ready) {
        return std::future_status::ready;
      } else {
        return std::future_status::timeout;
      }
    }

    ~Future() {
      if (get_p() != NULL) {
        get();
      }
    }

  private:
    std::unique_ptr<FutureData> data_p;
  };

  template <typename Fn, typename... Args>
  Future<std::invoke_result_t<Fn, Args...>>
  rpc(size_t remote_worker, Fn&& fn, Args&&... args) {
    assert(remote_worker < nworkers());

    size_t remote_proc = remote_worker / nworkers_local();
    u_int8_t remote_worker_local = (u_int8_t) remote_worker % nworkers_local();

    Future<std::invoke_result_t<Fn, Args...>> future;
    rpc_t my_rpc(future.get_p(), remote_worker_local);
    my_rpc.load(std::forward<Fn>(fn), std::forward<Args>(args)...);

    std::vector<rpc_t> rpcs;
    rpcs.push_back(std::move(my_rpc));
    generic_handler_request_impl_(remote_proc, std::move(rpcs));
    requesteds[my_worker_local()].val++;

    return std::move(future);
  }
} // end of arh

#endif