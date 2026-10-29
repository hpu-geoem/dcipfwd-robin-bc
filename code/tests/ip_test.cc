// Integral-chargeability IP tests.
// The model resistivity is the DC resistivity, the instantaneous conductivity is sigma / (1 - eta)
// and the apparent chargeability is 1 - V_inst / V_DC.

#include "test_common.h"

#include <fmt/format.h>

#include <cmath>
#include <vector>

namespace {

using dc_test::make_mesh;
using dc_test::make_model;
using dc_test::solve_dc;

constexpr double kRho1 = 100.0, kRho2 = 10.0;

mfem::Vector line_point(bool spectral, double coordinate) {
  return spectral ? mfem::Vector({ 0.0, coordinate, 0.0 }) : mfem::Vector({ coordinate, 0.0, 0.0 });
}

DCSolverOptions options(bool spectral, bool dc_coefficients = false) {
  DCSolverOptions result = dc_test::dc_options(2);
  result.dc_coefficients = dc_coefficients;
  if (spectral) {
    result.kx_min = 1.0e-3 / 300.0;
    result.kx_max = 15.0 / 20.0;
    result.kx_points_per_decade = 6;
  }
  return result;
}

/// Apparent chargeability of a pole-pole and a dipole-dipole row on the line.
std::vector<double> apparent_chargeability(const Model &model, bool spectral, int refinements, bool dc_coefficients = false) {
  const std::vector<mfem::Vector> electrodes = { line_point(spectral, -60.0), line_point(spectral, -40.0) };
  const std::vector<mfem::Vector> receivers = { line_point(spectral, 0.0), line_point(spectral, 20.0) };
  const auto dc = solve_dc(model, electrodes, options(spectral), refinements, receivers, refinements);
  const auto instantaneous =
      solve_dc(model, electrodes, options(spectral, dc_coefficients), refinements, receivers, refinements, true);
  const std::vector<double> a = dc->potentials(0, receivers), b = dc->potentials(1, receivers);
  const std::vector<double> ai = instantaneous->potentials(0, receivers), bi = instantaneous->potentials(1, receivers);
  const double pole = a[0], pole_instantaneous = ai[0];
  const double dipole = a[0] - a[1] - b[0] + b[1], dipole_instantaneous = ai[0] - ai[1] - bi[0] + bi[1];
  return { 1.0 - pole_instantaneous / pole, 1.0 - dipole_instantaneous / dipole };
}

double two_layer_difference(double rho1, double rho2, double thickness, bool dipole) {
  const auto v = [&](double distance) { return dc_test::two_layer_surface_potential(rho1, rho2, thickness, distance, 4000); };
  return dipole ? v(60.0) - v(80.0) - v(40.0) + v(60.0) : v(60.0);
}

} // namespace

TEST_CASE("Uniform integral chargeability scales DC potentials in three and two dimensions", "[ip]") {
  for (bool spectral : { false, true }) {
    for (double eta : { 0.0, 0.23 }) {
      CAPTURE(spectral, eta);
      const auto index = [](const mfem::Vector &center) { return center[center.Size() - 1] < dc_test::plane_depth(7) ? 0 : 1; };
      const auto model = make_model(make_mesh(spectral, index, 0, 12), { kRho1, kRho2 }, { eta, eta });
      // A uniform chargeability scales every conductivity alike, so the Robin coefficients of the two states agree.
      for (bool dc_coefficients : { false, true }) {
        for (double response : apparent_chargeability(*model, spectral, 0, dc_coefficients)) {
          CHECK(response == test_common::approx(eta, 1.0e-10, 1.0e-12));
        }
      }
    }
  }
}

TEST_CASE("Layered apparent chargeability agrees with the analytic two-layer potentials", "[ip][layered]") {
  // A shallow chargeable basement gives apparent chargeabilities of a few percent at these offsets.
  const double thickness = dc_test::plane_depth(3);
  const double eta1 = 0.0, eta2 = 0.30;
  const double rho1_instantaneous = kRho1 * (1.0 - eta1), rho2_instantaneous = kRho2 * (1.0 - eta2);
  const auto index = [&](const mfem::Vector &center) { return center[2] < thickness ? 0 : 1; };
  const auto model = make_model(make_mesh(false, index, 3, 16), { kRho1, kRho2 }, { eta1, eta2 });
  const std::vector<double> responses = apparent_chargeability(*model, false, 2);
  for (int row = 0; row < 2; ++row) {
    const double plain = two_layer_difference(kRho1, kRho2, thickness, row == 1);
    const double instantaneous = two_layer_difference(rho1_instantaneous, rho2_instantaneous, thickness, row == 1);
    const double expected = 1.0 - instantaneous / plain;
    if (test_common::is_root()) {
      fmt::print("IP 3D layered row {}: computed {:.6f}, analytic {:.6f}, relative error {:.3e}\n", row, responses[row], expected,
                 std::abs(responses[row] / expected - 1.0));
    }
    // Measured: 3.1e-3 (pole) and 4.8e-3 (dipole); the coarse cells at the 65 m interface dominate.
    CHECK(responses[row] == test_common::approx(expected, 1.0e-2));
  }
}

TEST_CASE("The instantaneous pass uses the boundary data and coefficients of its own background", "[ip][boundary]") {
  // A chargeable basement changes the instantaneous background, so the two states have different Robin coefficients.
  const double thickness = dc_test::plane_depth(3);
  const auto index = [&](const mfem::Vector &center) { return center[center.Size() - 1] < thickness ? 0 : 1; };
  for (bool spectral : { false, true }) {
    CAPTURE(spectral);
    const auto small = make_model(make_mesh(spectral, index), { kRho1, 1000.0 }, { 0.0, 0.3 });
    const auto large = make_model(make_mesh(spectral, index, 6), { kRho1, 1000.0 }, { 0.0, 0.3 });
    const std::vector<mfem::Vector> electrodes = { line_point(spectral, -60.0), line_point(spectral, 40.0) };
    const std::vector<mfem::Vector> points = { line_point(spectral, 0.0), line_point(spectral, 120.0) };
    const auto potentials = [&](const Model &model, DCBoundaryReference reference, bool dc_coefficients) {
      DCSolverOptions config = options(spectral, dc_coefficients);
      config.reference = reference;
      const auto solver = solve_dc(model, electrodes, config, 2, points, 2, true);
      std::vector<double> values = solver->potentials(0, points);
      const std::vector<double> second = solver->potentials(1, points);
      values.insert(values.end(), second.begin(), second.end());
      return values;
    };
    const auto difference = [](const std::vector<double> &a, const std::vector<double> &b) {
      double worst = 0.0;
      for (std::size_t i = 0; i < a.size(); ++i) {
        worst = std::max(worst, std::abs(a[i] / b[i] - 1.0));
      }
      return worst;
    };
    const auto far = potentials(*large, DCBoundaryReference::PerElectrode, false);
    const double le = difference(potentials(*small, DCBoundaryReference::PerElectrode, false), far);
    const double le_dc = difference(potentials(*small, DCBoundaryReference::PerElectrode, true), far);
    const double li = difference(potentials(*small, DCBoundaryReference::BoundaryData, false), far);
    const double li_dc = difference(potentials(*small, DCBoundaryReference::BoundaryData, true), far);
    if (test_common::is_root()) {
      fmt::print("Instantaneous truncation errors ({}): LE {:.2e}, LE with DC coefficients {:.2e}, LI {:.2e}, LI with DC "
                 "coefficients {:.2e}\n", spectral ? "2.5D" : "3D", le, le_dc, li, li_dc);
    }
    // The coefficients of the DC background are inexact for the instantaneous background; boundary data stay exact.
    CHECK(le_dc > 10.0 * le);
    CHECK(li < 2.0 * le);
    CHECK(li_dc < 2.0 * le);
  }
}
