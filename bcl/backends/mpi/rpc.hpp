#pragma once

#include "backend.hpp"

namespace BCL {
  struct args_ptr {
    char *data = nullptr;
    size_t n = 0;

    args_ptr() {}

    template <typename T>
    args_ptr push(const T &val) {
      data = (char *) realloc(data, n + sizeof(BCL::Container <T, BCL::serialize <T>>));
      new (data + n) BCL::Container <T, BCL::serialize <T>> (val);
      n += sizeof(BCL::Container <T, BCL::serialize <T>>);
      return *this;
    }

    void free() {
      std::free(data);
    }
  };

  args_ptr pack_args(args_ptr &&ptr) {
    return ptr;
  }

  template <typename T>
  args_ptr pack_args(args_ptr &&ptr, const T &arg) {
    return ptr.push(arg);
  }

  template <typename T, typename... Args>
  args_ptr pack_args(args_ptr &&ptr, const T &arg, const Args & ...args) {
    return pack_args(ptr.push(arg), args...);
  }

  struct AbstractContainer {
    char *data = nullptr;

    AbstractContainer(char *data) : data(data) {}
    AbstractContainer() {}

    template <typename T>
    T get() {
      return ((BCL::Container <T, BCL::serialize <T>> *) data)->get();
    }

    void set(char *data) {
      this->data = data;
    }
  };

  template <typename... LArgs>
  struct GetArgs;

  template <typename T, typename... LArgs>
  struct GetArgs <T, LArgs...> {
    static std::vector <AbstractContainer> get_abstract_args(
      std::vector <AbstractContainer> &my_args, char *data) {
      my_args.push_back(AbstractContainer(data));
      return GetArgs <LArgs...>::get_abstract_args(my_args, data + sizeof(BCL::Container <T, BCL::serialize <T>>));
    }
  };

  template <>
  struct GetArgs <> {
    static std::vector <AbstractContainer> get_abstract_args(
      std::vector <AbstractContainer> &my_args, char *data) {
      return my_args;
    }
  };

  template <typename FT> struct Function;

  template <typename RT, typename... Args>
  struct Function <RT (*)(Args...)> {

    template <std::size_t... Ints>
    static BCL::Container <RT, BCL::serialize <RT>> call(void (*fn)(void),
      std::vector <AbstractContainer> &args, std::integer_sequence <std::size_t, Ints...>) {
      RT (*my_fn)(Args...) = reinterpret_cast <RT (*)(Args...)> (fn);
      BCL::Container <RT, BCL::serialize <RT>> rv;
      if constexpr(!std::is_void <RT>::value) {
        rv.set(my_fn(args[Ints].get <typename std::tuple_element <Ints, std::tuple <Args...>>::type>()...));
      } else {
        my_fn(args[Ints].get <typename std::tuple_element <Ints, std::tuple <Args...>>::type>()...);
      }
      return rv;
    }

    static void execute(void (*fn)(void), char *args, uint64_t return_rank) {
      std::vector <AbstractContainer> my_args;
      my_args = GetArgs<Args...>::get_abstract_args (
        my_args, args);
      constexpr std::size_t count = std::tuple_size <std::tuple <Args...>>::value;

      if (my_args.size() != count) {
        throw std::runtime_error("Function::execute(): called with incorrect number of arguments");
      }
      BCL::Container <RT, BCL::serialize <RT>> rv =
        Function <RT (*)(Args...)>::call(fn, my_args, std::make_integer_sequence <std::size_t,
        std::tuple_size <std::tuple <Args...>>::value> ());
      MPI_Send(&rv, sizeof(BCL::Container <RT, BCL::serialize <RT>>), MPI_CHAR,
        return_rank, BCL::signal::rpc_rval, MPI_COMM_WORLD);
    }
  };

  extern void progress_thread();
  // Position-independent FP (with respect to progress_thread)
  template <typename RV, typename... Args>
  uint64_t get_pid_fp(RV (*foo)(Args...)) {
    uint64_t muh_foo = reinterpret_cast <char *> (&BCL::progress_thread) -
      reinterpret_cast <char *> (foo);
    return muh_foo;
  }

  template <typename T>
  T resolve_pid_fp(uint64_t ptr) {
    return reinterpret_cast <T> (reinterpret_cast <char *> (&BCL::progress_thread)
      - reinterpret_cast <char *> (ptr));
  }

  void handle_rpc(int source, uint64_t exec_ptr, uint64_t fn_ptr,
    uint64_t args_size) {
    void (*fn)(void) = resolve_pid_fp <void (*)(void)> (fn_ptr);
    void (*exec_fn)(void (*) (void), char *, uint64_t) =
      resolve_pid_fp <void (*) (void (*) (void), char *, uint64_t)> (exec_ptr);
    char *data = (char *) malloc(args_size);
    MPI_Recv(data, args_size, MPI_CHAR, source, BCL::signal::rpc_args,
      MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    exec_fn(fn, data, source);
    free(data);
  }

  template <typename RT, typename... Args>
  RT rpc(uint64_t dst, RT (*fn) (Args...), const Args & ... args) {
    if (!BCL::progress_running) {
      throw std::runtime_error("BCL: error, attempting to send RPC when progress\
        thread is not enabled");
    }
    args_ptr my_args = pack_args(args_ptr(), args...);
    uint64_t request[4];
    request[0] = BCL::signal::rpc_label;
    request[1] = get_pid_fp(Function <RT (*)(Args...)>::execute);
    request[2] = get_pid_fp(fn);
    request[3] = my_args.n;
    MPI_Send(request, 4, MPI_UNSIGNED_LONG_LONG, dst, BCL::signal::progress_tag,
      MPI_COMM_WORLD);
    MPI_Send(my_args.data, my_args.n, MPI_CHAR, dst, BCL::signal::rpc_args,
      MPI_COMM_WORLD);
    MPI_Status status;
    BCL::Container <RT, BCL::serialize <RT>> rv;
    MPI_Recv(&rv, sizeof(BCL::Container <RT, BCL::serialize <RT>>), MPI_CHAR,
      dst, BCL::signal::rpc_rval, MPI_COMM_WORLD, &status);
    my_args.free();
    fflush(stdout);
    if constexpr(!std::is_void <RT>::value) {
      RT result = rv.get();
      rv.free();
      return result;
    }
  }
}
