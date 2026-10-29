// Layered-earth Green's function and the Robin coefficients it defines on the truncation boundary.

#pragma once

#include <mfem.hpp>

#include <algorithm>
#include <cstdint>
#include <limits>
#include <map>
#include <memory>
#include <vector>

namespace dcip {

/// Layer of a horizontally layered earth: depth of its top below the surface and its conductivity.
struct ConductiveLayer {
  double top;
  double sigma;
};

/// Potential of a unit point current in a horizontally layered earth below an insulating surface.
class LayeredPotential {
public:
  /// Layers from the surface down with increasing tops, the first at depth 0; the source lies at a depth below the surface.
  LayeredPotential(const std::vector<ConductiveLayer> &layers, double source_depth);

  /// Return the potential at horizontal offset r and depth z, and its derivatives with respect to r and z.
  double potential(double r, double z, double &d_dr, double &d_dz) const;

  /// Default k L above which 2.5D boundary coefficients use the source and its image only (L: distance to the source).
  static constexpr double kDecaySwitch = 15.0;

  /// Largest k L at which transformed_potential may be evaluated.
  static constexpr double kLargestDecay = 40.0;

  /// Return half of the cosine transform along x, int_0^inf V cos(k x) dx, of the potential at wavenumber k > 0,
  /// transverse offset y and depth z, and its derivatives with respect to y and z.
  double transformed_potential(double k, double y, double z, double &d_dy, double &d_dz) const;

  /// Return the conductivity of the layer of the source.
  double source_conductivity() const { return sigma_[source_layer_]; }

private:
  struct Kernel {
    double value;
    double d_dz;
  };

  int layer_of(double z) const;
  Kernel secondary_kernel(double lambda, double z, int receiver_layer) const;
  double secondary_path(double z, int receiver_layer) const;

  std::vector<double> tops_; // Top of every layer and infinity below the last
  std::vector<double> sigma_;
  double source_depth_;
  int source_layer_;
  double low_scale_ = std::numeric_limits<double>::infinity(); // Smallest wavenumber at which the kernel changes shape
};

/// Layer of the background column: depth of its top in model coordinates (z down), DC conductivity and chargeability.
struct BackgroundLayer {
  double top;
  double sigma;
  double eta;
};

/// Return the instantaneous conductivity sigma / (1 - eta) of a medium of DC conductivity sigma and integral chargeability eta.
double chargeable_conductivity(double sigma, double eta);

/// Reference source and layered background of the Robin boundary condition.
/// G denotes the potential of a unit current at the reference source in 3D and its full cosine transform
/// int V cos(k x) dx over the strike in 2.5D; both use the DC or the instantaneous conductivity of the background.
class DCBoundary {
public:
  /// Take the ground layers of the background column from the surface down, the reference source at (x, y, z), and
  /// the k L above which 2.5D coefficients come from the source and its image only.
  DCBoundary(int dim, const std::vector<BackgroundLayer> &column, const mfem::Vector &reference,
             double decay_switch = LayeredPotential::kDecaySwitch);

  /// Return the reference source in model coordinates.
  const mfem::Vector &reference() const { return reference_; }

  /// Return the layers of the background with equal neighbours merged.
  std::vector<BackgroundLayer> layers() const;

  double decay_switch() const { return decay_switch_; }

  /// Return -dG/dn / G, not clipped, at a boundary point with unit outward normal for wavenumber k (zero in 3D);
  /// with log_value, also ln G.
  double robin_ratio(const mfem::Vector &position, const mfem::Vector &unit_normal, double k, bool chargeable,
                     double *log_value = nullptr) const;

  /// Return the Robin coefficient max(0, -dG/dn / G).
  double robin_factor(const mfem::Vector &position, const mfem::Vector &unit_normal, double k, bool chargeable) const {
    return std::max(0.0, robin_ratio(position, unit_normal, k, chargeable));
  }

  /// Return G at horizontal (3D) or transverse (2D) offset r >= 0 and depth z in model coordinates, and its
  /// derivatives along the offset and along z, up to the factor exp(log_scale(r, z, k, chargeable)).
  double potential_gradient(double r, double z, double k, bool chargeable, double &d_dr, double &d_dz) const;

  /// Return the logarithm of the factor that turns the values of potential_gradient into G.
  double log_scale(double r, double z, double k, bool chargeable) const;

  /// Return the vertical distance from the reference source to depth z, both clipped to the background surface as in
  /// potential_gradient.
  double vertical_separation(double z) const;

  /// Return whether potential_gradient uses the layered transform at offset r and depth z (always in 3D).
  bool resolved(double r, double z, double k) const;

private:
  const LayeredPotential &potential(bool chargeable) const;

  int dim_;
  mfem::Vector reference_;
  double decay_switch_;
  double surface_ = 0.0;
  std::vector<double> tops_, sigma_, eta_;
  mutable std::unique_ptr<LayeredPotential> potentials_[2];
};

/// Offset range of a set of boundary points at one depth (model coordinates).
struct DepthRange {
  double z;
  double r_min;
  double r_max;
};

/// Interpolation tables of the Robin ratio of the reference source of a DCBoundary at fixed depths.
/// For each depth the table holds L dG/dr / G and L dG/dz / G, with L the distance to the source, and optionally ln G,
/// at offsets r = a sinh(s) on a uniform grid in s with a = max(|z - z_source|, 1e-6 r_max); cubic Lagrange
/// interpolation in s. The ranks of the communicator, which must all pass the same ranges, compute the columns in turn
/// and share them. Interpolation stays on one side of the 2.5D decay switch; with too few nodes there the ratio is
/// evaluated directly.
class RobinTable {
public:
  RobinTable(const DCBoundary &boundary, double k, bool chargeable, const std::vector<DepthRange> &ranges,
             MPI_Comm comm = MPI_COMM_SELF, bool with_values = false, double step = 1.0 / 16.0);

  /// Return the number of potential evaluations a table of these ranges needs.
  static long long cost(const DCBoundary &boundary, const std::vector<DepthRange> &ranges, double step = 1.0 / 16.0);

  /// Return -dG/dn / G, not clipped, at a tabulated depth z, offset r and normal components along r and z; with
  /// log_value (tables built with values), also ln G.
  double ratio(double z, double r, double normal_r, double normal_z, double *log_value = nullptr) const;

  /// Return the number of ratios evaluated directly instead of interpolated.
  long long direct_evaluations() const { return direct_evaluations_; }

private:
  struct Column {
    double a, s_min, z;
    int resolved = 0;                                // Nodes before the image-only range of k L
    std::vector<double> radial, vertical, log_value; // L dG/dr / G, L dG/dz / G and ln G at the nodes
  };

  static void grid(const DCBoundary &boundary, const DepthRange &range, double step, double &a, double &s_min, int &nodes);
  double direct(double z, double r, double normal_r, double normal_z, double *log_value) const;

  const DCBoundary &boundary_;
  double k_;
  bool chargeable_;
  bool with_values_;
  double step_;
  std::map<std::uint64_t, Column> columns_;
  mutable long long direct_evaluations_ = 0;
};

} // namespace dcip
