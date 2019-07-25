#pragma once

#include <bcl/bcl.hpp>
#include <cstring>

namespace BCL {

extern gex_EP_t ep;

namespace gas {

template <typename T, std::size_t N>
struct mtuple {
  using type = decltype(std::tuple_cat(std::tuple<T>(), typename mtuple<T, N-1>::type()));
};

template <typename T>
struct mtuple<T, 1> {
  using type = std::tuple<T>;
};

template <std::size_t N>
struct gasnet_pack {
  static constexpr std::size_t gex_arg_size = sizeof(gex_AM_Arg_t);
  static constexpr std::size_t nargs = (N + gex_arg_size - 1) / gex_arg_size;

  gasnet_pack() = default;
  gasnet_pack(const gasnet_pack&) = default;

  gasnet_pack(const char data[N]) {
    std::memcpy(buf, data, sizeof(char)*N);
  }

  gasnet_pack(const std::array<gex_AM_Arg_t, nargs>& args) {
    for (size_t i = 0; i < nargs; i++) {
      buf[i] = args[i];
    }
  }

  template <typename... Args>
  gasnet_pack(const std::tuple<Args...>& args) {
    static_assert(sizeof(std::tuple<Args...>) <= N);
    std::memcpy(buf, &args, sizeof(std::tuple<Args...>));
  }

  template <typename... Args>
  std::tuple<Args...>& unpack() {
    return *reinterpret_cast<std::tuple<Args...>*>(buf);
  }

  using gex_tuple_type = typename mtuple<gex_AM_Arg_t, nargs>::type;

  gex_tuple_type& as_gex_tuple() {
    return *reinterpret_cast<gex_tuple_type*>(buf);
  }

  // TODO: send AM fn

  gex_AM_Arg_t buf[nargs];
};

size_t acknowledged = 0;
size_t requested = 0;

size_t handler_num = 0;

void acknowledge_handler_(gex_Token_t token) {
  acknowledged++;
}

template <std::size_t N, std::size_t... I>
gasnet_pack<N> get_packed_repr_impl_(const std::array<gex_AM_Arg_t, 16>& args, std::index_sequence<I...>) {
  return gasnet_pack<N>({args[I]...});
}

template <typename Fn, typename... Args>
void generic_handler_(gex_Token_t token,
                      gex_AM_Arg_t arg0, gex_AM_Arg_t arg1, gex_AM_Arg_t arg2,
                      gex_AM_Arg_t arg3, gex_AM_Arg_t arg4, gex_AM_Arg_t arg5,
                      gex_AM_Arg_t arg6, gex_AM_Arg_t arg7, gex_AM_Arg_t arg8,
                      gex_AM_Arg_t arg9, gex_AM_Arg_t arg10, gex_AM_Arg_t arg11,
                      gex_AM_Arg_t arg12, gex_AM_Arg_t arg13, gex_AM_Arg_t arg14,
                      gex_AM_Arg_t arg15)
{
  using tuple_type = std::tuple<Fn, std::tuple<Args...>>;
  constexpr size_t tuple_size = sizeof(tuple_type);
  constexpr size_t nargs = gasnet_pack<tuple_size>::nargs;
  gasnet_pack<tuple_size> packed_args = get_packed_repr_impl_<tuple_size>({arg0, arg1, arg2,
                                                                           arg3, arg4, arg5,
                                                                           arg6, arg7, arg8,
                                                                           arg9, arg10, arg11,
                                                                           arg12, arg13, arg14,
                                                                           arg15},
                                                                           std::make_index_sequence<nargs>());

  tuple_type& args = packed_args.template unpack<Fn, std::tuple<Args...>>();

  std::apply(std::move(std::get<0>(args)), std::move(std::get<1>(args)));
  gex_AM_ReplyShort(token, GEX_AM_INDEX_BASE, 0);
}

void init_am() {
  handler_num = GEX_AM_INDEX_BASE;

  gex_AM_Entry_t entry;
  entry.gex_index = handler_num++;
  entry.gex_fnptr = (gex_AM_Fn_t) acknowledge_handler_;
  entry.gex_flags = GEX_FLAG_AM_SHORT | GEX_FLAG_AM_REPLY;
  entry.gex_nargs = 0;

  gex_EP_RegisterHandlers(BCL::ep, &entry, 1);
  BCL::barrier();
}

void flush_am() {
  while (acknowledged < requested) {
    gasnet_AMPoll();
  }
}

template <typename Fn, typename... Args>
struct launch_am {
  launch_am(size_t am_id, Fn fn) : am_id_(am_id), fn_(fn) {}

  using args_type = std::tuple<Fn, std::tuple<Args...>>;
  static constexpr size_t args_size = sizeof(args_type);

  template<std::size_t... I>
  void launch_impl_(size_t remote_proc, gasnet_pack<args_size>& pack, std::index_sequence<I...>) {
    auto& args = pack.as_gex_tuple();
    // int rv = gasnetc_AMRequestShortM(BCL::tm, remote_proc, am_id_, 0 GASNETI_THREAD_GET, sizeof...(I), std::get<I>(args)...);
    int rv = gasnetc_AMRequestShortM(BCL::tm, remote_proc, am_id_, 0 GASNETI_THREAD_GET, sizeof...(I), pack.buf[I]...);
  }

  void launch(size_t remote_proc, Args&&... args) {
    args_type args_{fn_, std::tuple<Args...>(args...)};
    gasnet_pack<args_size> pack(args_);

    auto args_tuple = std::make_tuple(args...);

    launch_impl_(remote_proc, pack, std::make_index_sequence<gasnet_pack<args_size>::nargs>());
  }

  size_t am_id_;
  Fn fn_;
};

template <typename Fn, typename... Args>
auto register_am(Fn fn, Args... args) {
  size_t max_args = gex_AM_MaxArgs();

  if (handler_num > 256) {
    throw std::runtime_error("Used up all your GASNet-EX Handlers.");
  }

  gex_AM_Entry_t entry;
  entry.gex_index = handler_num++;
  entry.gex_fnptr = (gex_AM_Fn_t) generic_handler_<Fn, Args...>;
  entry.gex_flags = GEX_FLAG_AM_SHORT | GEX_FLAG_AM_REQUEST;
  entry.gex_nargs = gasnet_pack<sizeof(std::tuple<Fn, std::tuple<Args...>>)>::nargs;

  gex_EP_RegisterHandlers(BCL::ep, &entry, 1);
  BCL::barrier();
  return launch_am<Fn, Args...>(entry.gex_index, fn);
}

} // end gas
} // end BCL
