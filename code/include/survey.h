// Electrodes, four-electrode measurements and geometric factors.

#pragma once

#include <mfem.hpp>

#include <string>
#include <vector>

namespace dcip {

/// Four-electrode measurement: indices of A, B, M and N in the electrode list, -1 for a pole at infinity.
struct Measurement {
  int a, b, m, n;
};

/// Electrode positions (x, y, z; z down) and the measurements made with them.
/// Every current is unit; a response is the potential difference per ampere.
class Survey {
public:
  /// Read a survey file: the electrode count, one "x y z" row per electrode,
  /// the measurement count and one "a b m n" row per measurement. '#' starts a comment.
  explicit Survey(const std::string &file);

  Survey(std::vector<mfem::Vector> electrodes, std::vector<Measurement> measurements);

  const std::vector<mfem::Vector> &electrodes() const { return electrodes_; }
  const std::vector<Measurement> &measurements() const { return measurements_; }

  /// Return the sorted indices of the electrodes that carry current.
  std::vector<int> current_electrodes() const;

  /// Return the sorted indices of the electrodes that measure potential.
  std::vector<int> potential_electrodes() const;

  /// Return 4 pi / sum(+-(1/r + 1/r')) with the image r' of a flat surface at depth surface_z.
  double geometric_factor(const Measurement &measurement, double surface_z) const;

  /// Return the smallest and largest current-to-potential electrode distance in the (y, z) plane.
  std::pair<double, double> transverse_distance_range() const;

  /// Return the centroid of the current electrodes.
  mfem::Vector current_centroid() const;

private:
  void validate() const;

  std::vector<mfem::Vector> electrodes_;
  std::vector<Measurement> measurements_;
};

/// Return the potential difference of a measurement from the potentials of unit currents,
/// where potential(e, r) is the potential at electrode r of the current at electrode e.
template <typename Potential> double potential_difference(const Measurement &row, const Potential &potential) {
  const auto value = [&](int source, int receiver) { return source >= 0 && receiver >= 0 ? potential(source, receiver) : 0.0; };
  return value(row.a, row.m) - value(row.a, row.n) - value(row.b, row.m) + value(row.b, row.n);
}

} // namespace dcip
