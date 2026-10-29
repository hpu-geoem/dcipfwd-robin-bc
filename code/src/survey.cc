#include "survey.h"

#include "utils.h"

#include <fmt/format.h>

#include <algorithm>
#include <cmath>
#include <fstream>
#include <limits>
#include <set>
#include <sstream>

namespace dcip {

namespace {

/// Return the next non-empty line of a stream with its comment removed, or an empty string at the end.
std::string next_line(std::istream &input) {
  std::string line;
  while (std::getline(input, line)) {
    line = strip_comment(line);
    if (!line.empty()) {
      return line;
    }
  }
  return "";
}

} // namespace

Survey::Survey(const std::string &file) {
  std::ifstream input(file);
  MFEM_VERIFY(input.good(), fmt::format("Cannot open the survey file {}.", file));
  const int n_electrodes = std::stoi(next_line(input));
  for (int index = 0; index < n_electrodes; ++index) {
    const std::string line = next_line(input);
    std::istringstream row(line);
    mfem::Vector position(3);
    row >> position[0] >> position[1] >> position[2];
    MFEM_VERIFY(!row.fail(), fmt::format("Invalid electrode row '{}' in {}.", line, file));
    electrodes_.push_back(position);
  }
  const int n_measurements = std::stoi(next_line(input));
  for (int index = 0; index < n_measurements; ++index) {
    const std::string line = next_line(input);
    std::istringstream row(line);
    Measurement measurement{};
    row >> measurement.a >> measurement.b >> measurement.m >> measurement.n;
    MFEM_VERIFY(!row.fail(), fmt::format("Invalid measurement row '{}' in {}.", line, file));
    measurements_.push_back(measurement);
  }
  validate();
}

Survey::Survey(std::vector<mfem::Vector> electrodes, std::vector<Measurement> measurements)
    : electrodes_(std::move(electrodes)), measurements_(std::move(measurements)) {
  validate();
}

void Survey::validate() const {
  const int n = static_cast<int>(electrodes_.size());
  for (const mfem::Vector &position : electrodes_) {
    MFEM_VERIFY(position.Size() == 3, "Electrode positions have three coordinates.");
  }
  for (const Measurement &row : measurements_) {
    for (int index : { row.a, row.b, row.m, row.n }) {
      MFEM_VERIFY(index >= -1 && index < n, fmt::format("Electrode index {} is out of range.", index));
    }
    MFEM_VERIFY(row.a >= 0 || row.b >= 0, "A measurement needs a current electrode.");
    MFEM_VERIFY(row.m >= 0 || row.n >= 0, "A measurement needs a potential electrode.");
  }
}

std::vector<int> Survey::current_electrodes() const {
  std::set<int> result;
  for (const Measurement &row : measurements_) {
    for (int index : { row.a, row.b }) {
      if (index >= 0) {
        result.insert(index);
      }
    }
  }
  return { result.begin(), result.end() };
}

std::vector<int> Survey::potential_electrodes() const {
  std::set<int> result;
  for (const Measurement &row : measurements_) {
    for (int index : { row.m, row.n }) {
      if (index >= 0) {
        result.insert(index);
      }
    }
  }
  return { result.begin(), result.end() };
}

double Survey::geometric_factor(const Measurement &measurement, double surface_z) const {
  const auto inverse_distance = [&](int source, int receiver) {
    if (source < 0 || receiver < 0) {
      return 0.0;
    }
    const mfem::Vector &a = electrodes_[source], &m = electrodes_[receiver];
    const double distance = a.DistanceTo(m);
    const double image =
        std::sqrt(distance * distance + 4.0 * std::max(0.0, a[2] - surface_z) * std::max(0.0, m[2] - surface_z));
    return 1.0 / distance + 1.0 / image;
  };
  const double sum = potential_difference(measurement, inverse_distance);
  return 4.0 * PI / sum;
}

std::pair<double, double> Survey::transverse_distance_range() const {
  double minimum = std::numeric_limits<double>::infinity(), maximum = 0.0;
  for (const Measurement &row : measurements_) {
    for (int source : { row.a, row.b }) {
      for (int receiver : { row.m, row.n }) {
        if (source >= 0 && receiver >= 0) {
          const mfem::Vector &a = electrodes_[source], &m = electrodes_[receiver];
          const double distance = std::hypot(a[1] - m[1], a[2] - m[2]);
          minimum = std::min(minimum, distance);
          maximum = std::max(maximum, distance);
        }
      }
    }
  }
  return { minimum, maximum };
}

mfem::Vector Survey::current_centroid() const {
  const std::vector<int> currents = current_electrodes();
  MFEM_VERIFY(!currents.empty(), "The survey has no current electrode.");
  mfem::Vector centroid(3);
  centroid = 0.0;
  for (int index : currents) {
    centroid += electrodes_[index];
  }
  centroid /= static_cast<double>(currents.size());
  return centroid;
}

} // namespace dcip
