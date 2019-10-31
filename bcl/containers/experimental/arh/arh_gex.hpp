//
// Created by Jiakun Yan on 10/30/19.
//

#ifndef BCL_ARH_GEX_HPP
#define BCL_ARH_GEX_HPP

#include "arh_am.hpp"

namespace ARH {
  extern size_t handler_num;

  int getHandlerIdx() {
    return handler_num;
  }

  int registerHandlers(gex_AM_Entry_t* htable, size_t num) {
    handler_num += num;
    return gex_EP_RegisterHandlers(BCL::ep, htable, num);
  }
}

#endif //BCL_ARH_GEX_HPP
