#ifndef BCL_ARH_HPP
#define BCL_ARH_HPP

#define ARH_THREAD_PIN

#include "bcl/bcl.hpp"
#include <functional>
#include <iostream>
#include <thread>
#include <vector>
#include <unordered_map>
#include <atomic>
#include <sched.h>
#include <pthread.h>
#include <utility>

#include "arh_global.hpp"
#include "arh_threadbarrier.hpp"
#include "arh_tools.hpp"
#include "arh_AverageTimer.hpp"
#include "arh_rpc_t.hpp"
#include "arh_agg_buffer.hpp"
#include "arh_worker_object.hpp"
#include "arh_base.hpp"
#include "arh_collective.hpp"
#include "arh_am.hpp"
#include "arh_am_agg.hpp"

namespace ARH {
}

#endif //BCL_ARH_HPP
