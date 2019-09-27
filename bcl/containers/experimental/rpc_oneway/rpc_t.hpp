#ifndef BCL_RPC_T_HPP
#define BCL_RPC_T_HPP

#include <array>

namespace BCL {
    // Get a *position independent* function pointer
    template<typename T>
    std::uintptr_t get_pi_fnptr_(T *fn) {
      return reinterpret_cast<std::uintptr_t>(fn) - reinterpret_cast<std::uintptr_t>(init);
    }

    // Resolve a *position independent* function pointer
    // to a local function pointer.
    template<typename T = char>
    T *resolve_pi_fnptr_(std::uintptr_t fn) {
      return reinterpret_cast<T *>(fn + reinterpret_cast<std::uintptr_t>(init));
    }

    template<typename Fn, typename... Args>
    struct rpc_invoker;

    struct rpc_t {
        static constexpr size_t max_rpc_size = 8; // argument/return value data size
        static constexpr size_t rpc_result_source_rank = std::numeric_limits<size_t>::max();
        static constexpr size_t rpc_kill_nonce = std::numeric_limits<size_t>::max();
        static constexpr size_t rpc_flush_nonce = rpc_kill_nonce-1;

        using rpc_result_t = std::array<char, max_rpc_size>;

        std::size_t rpc_id_;
        bool buffered_;
        std::size_t source_rank_;
        std::uintptr_t fn_;
        std::uintptr_t invoker_;
        std::array<char, max_rpc_size> data_;

        rpc_t() = default;

        rpc_t(std::size_t rpc_id, bool buffered = false, std::size_t source_rank = BCL::rank()) : rpc_id_(rpc_id),
                                                                                                  buffered_(buffered),
                                                                                                  source_rank_(
                                                                                                          source_rank) {}

        template<typename Fn, typename... Args>
        void load(Fn &&fn, Args &&... args) {
          using tuple_t = std::tuple<std::remove_reference_t<Args>...>;
          static_assert(sizeof(tuple_t) <= max_rpc_size, "Max RPC size too small.");

          this->fn_ = get_pi_fnptr_(reinterpret_cast<char *>(+fn));

          invoker_ = get_pi_fnptr_(
                  reinterpret_cast<char *>(&decltype(
                  rpc_invoker<Fn, Args...>())::invoke));

          tuple_t &tuple = *reinterpret_cast<tuple_t *>(data_.data());
          new(&tuple) tuple_t();

          tuple = std::tie(args...);
        }

        template<typename T>
        void load_result(const T &value) {
          static_assert(sizeof(T) <= max_rpc_size, "Max return val size too small");
          *reinterpret_cast<T *>(data_.data()) = value;
        }

        rpc_t run() {
          using invoker_type = rpc_t (*)(BCL::rpc_t &);

          auto invoker = reinterpret_cast<invoker_type>
                  (resolve_pi_fnptr_(invoker_));
          rpc_t rpc_result = invoker(*this);
          return rpc_result;
        }
    };

    template<typename Fn, typename... Args>
    struct rpc_invoker {

        template<std::size_t... I>
        static auto invoke_impl_(rpc_t &rpc_requests, std::index_sequence<I...>) {
          using fn_t = decltype(+std::declval<std::remove_reference_t<Fn>>());
          fn_t fn = reinterpret_cast<fn_t>(resolve_pi_fnptr_(rpc_requests.fn_));
          using tuple_t = std::tuple<std::remove_reference_t<Args>...>;
          tuple_t &args_ptr = *reinterpret_cast<tuple_t *>(rpc_requests.data_.data());

          return std::invoke(std::forward<fn_t>(fn),
                             std::forward<Args>(std::get<I>(args_ptr))...);
        }

        static rpc_t invoke(rpc_t &rpc_requests) {
          rpc_t rpc_result(rpc_requests.rpc_id_, rpc_requests.buffered_, rpc_t::rpc_result_source_rank);
          if constexpr(std::is_void<std::invoke_result_t<Fn, Args...>>::value) {
            invoke_impl_(rpc_requests, std::index_sequence_for<Args...>{});
          } else {
            rpc_result.load_result(invoke_impl_(rpc_requests, std::index_sequence_for<Args...>{}));
          }
          return rpc_result;
        }
    };
}
#endif //BCL_RPC_T_HPP
