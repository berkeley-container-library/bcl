#pragma once

#include <stdexcept>

namespace BCL {

#ifdef BCL_NODEBUG
  #define BCL_DEBUG(stmt)
#else
  #define BCL_DEBUG(stmt) stmt
#endif

class error
{
public:
  error(const std::string& what_arg) : what_arg_(what_arg) {}
  error() {}

  const char* what() const throw() {
    return what_arg_.c_str();
  }

private:
  std::string what_arg_;
};

/*
   XXX: debug_error exceptions are expensive to check
        for at runtime, even if they don't occur.  They
        will only be thrown if the user compiles with
        the `DEBUG` flag---thus the expensive checks
        will normally be compiled away.
*/
class debug_error final : public error
{
public:
  debug_error(const std::string& what_arg) : what_arg_(what_arg) {}

  const char* what() const throw() {
    return what_arg_.c_str();
  }

private:
  std::string what_arg_;
};

} // end BCL
