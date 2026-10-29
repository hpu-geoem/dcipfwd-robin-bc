// Small helpers shared by the library and the programs.

#pragma once

#include <mfem.hpp>

#include <string>

namespace dcip {

constexpr double PI = 3.14159265358979323846;

/// Project an (x, y, z) survey coordinate into model coordinates: (y, z) for two-dimensional (2.5D) models.
inline mfem::Vector project_xyz_to_model(const mfem::Vector &xyz, int dim) {
  if (dim == 2) {
    return mfem::Vector({ xyz[1], xyz[2] });
  }
  return mfem::Vector(xyz);
}

/// Return a line with its comment (from '#') and surrounding blanks removed.
inline std::string strip_comment(const std::string &line) {
  std::string result = line.substr(0, line.find('#'));
  const auto first = result.find_first_not_of(" \t\r\n");
  if (first == std::string::npos) {
    return "";
  }
  const auto last = result.find_last_not_of(" \t\r\n");
  return result.substr(first, last - first + 1);
}

} // namespace dcip
