// Progress log on rank 0.

#pragma once

#include <fmt/format.h>

#include <mpi.h>

#include <cstdio>
#include <utility>

namespace dcip {

/// Verbosity of the log: 0 quiet, 1 progress, 2 every linear solve.
inline int &log_level() {
  static int level = 1;
  return level;
}

/// Print a formatted line on rank 0 of MPI_COMM_WORLD when the log level reaches level.
template <typename... Args> void log(int level, fmt::format_string<Args...> format, Args &&...args) {
  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  if (rank == 0 && log_level() >= level) {
    fmt::print(format, std::forward<Args>(args)...);
    fmt::print("\n");
    std::fflush(stdout);
  }
}

} // namespace dcip
