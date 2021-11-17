// SPDX-FileCopyrightText: 2021 Benjamin Brock
//
// SPDX-License-Identifier: BSD-3-Clause

#pragma once

// This REQUIRES macro design is borrowed from Jared Hoberock's Agency.

#define ___BCL_CONCATENATE_IMPL(x, y) x##y
#define __BCL_CONCATENATE(x, y) ___BCL_CONCATENATE_IMPL(x, y)
#define ___BCL_MAKE_UNIQUE(x) __BCL_CONCATENATE(x, __COUNTER__)
#define __BCL_REQUIRES_IMPL(unique_name, ...) bool unique_name = true, typename std::enable_if<(unique_name and __VA_ARGS__)>::type* = nullptr
#define __BCL_REQUIRES(...) __BCL_REQUIRES_IMPL(___BCL_MAKE_UNIQUE(__deduced_true), __VA_ARGS__)
