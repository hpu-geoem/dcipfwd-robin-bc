// 2.5D DC forward tests: half-space and two-layer potentials, truncation and wavenumber resolution, 3D extrusion, solvers.

#include "test_common.h"

#include <fmt/format.h>

#include <array>
#include <cmath>
#include <memory>
#include <vector>

namespace {

using dc_test::half_space_potential;
using dc_test::make_mesh;
using dc_test::make_model;
using dc_test::max_relative_error;
using dc_test::solve_dc;
using dc_test::two_layer_surface_potential;

mfem::Vector point(double x, double y, double z) { return mfem::Vector({ x, y, z }); }

DCSolverOptions options(int order, double k_min = 1.0e-3 / 300.0, double k_max = 15.0 / 60.0, int points = 4) {
  DCSolverOptions result = dc_test::dc_options(order);
  result.kx_min = k_min;
  result.kx_max = k_max;
  result.kx_points_per_decade = points;
  return result;
}

/// Geometric factor of a surface four-electrode array on the line.
double geometric_factor(double a, double b, double m, double n) {
  const double sum = 1.0 / std::abs(m - a) - 1.0 / std::abs(n - a) - 1.0 / std::abs(m - b) + 1.0 / std::abs(n - b);
  return 2.0 * PI / sum;
}

} // namespace

TEST_CASE("2.5D DC half-space potentials converge for surface and buried electrodes", "[dc][25d]") {
  constexpr double resistivity = 100.0;
  const auto model = make_model(make_mesh(true, [](const mfem::Vector &) { return 0; }), { resistivity });
  const std::vector<mfem::Vector> points = { point(0.0, -120.0, 0.0), point(0.0, -60.0, 0.0),  point(0.0, 40.0, 0.0),
                                             point(0.0, 80.0, 0.0),   point(0.0, 120.0, 60.0), point(0.0, -100.0, 100.0) };
  for (const mfem::Vector &electrode : { point(0.0, 0.0, 0.0), point(0.0, 0.0, 40.0) }) {
    double errors[2];
    for (int order : { 1, 2 }) {
      const auto solver = solve_dc(*model, { electrode }, options(order), 4, points, 2);
      errors[order - 1] = max_relative_error(
          *solver, 0, points, [&](const mfem::Vector &sample) { return half_space_potential(resistivity, electrode, sample); });
      for (double current : solver->source_currents(0)) {
        CHECK(current == test_common::approx(1.0, 1.0e-9));
      }
    }
    CAPTURE(electrode[1], electrode[2], errors[0], errors[1]);
    if (test_common::is_root()) {
      fmt::print("DC 2.5D half-space z={} order-1/order-2 errors: {:.3e}/{:.3e}\n", electrode[2], errors[0], errors[1]);
    }
    CHECK(errors[0] < 5.0e-2);
    CHECK(errors[1] < 2.5e-3);
    CHECK(errors[1] < errors[0]);
  }
}

TEST_CASE("2.5D DC layered responses separate domain truncation and wavenumber resolution", "[dc][25d][boundary]") {
  constexpr double rho1 = 100.0;
  const double thickness = dc_test::plane_depth(8);
  const std::vector<mfem::Vector> points = { point(0.0, -240.0, 0.0), point(0.0, -120.0, 0.0), point(0.0, 60.0, 0.0),
                                             point(0.0, 150.0, 0.0), point(0.0, 300.0, 0.0) };
  for (double rho2 : { 10.0, 1000.0 }) {
    const auto potential = [&](double source, double receiver) {
      return two_layer_surface_potential(rho1, rho2, thickness, std::abs(receiver - source), 4000);
    };
    const double factor = geometric_factor(-120.0, -60.0, 60.0, 120.0);
    const double expected_response =
        factor * (potential(-120.0, 60.0) - potential(-120.0, 120.0) - potential(-60.0, 60.0) + potential(-60.0, 120.0));
    const auto errors = [&](const DCSolverOptions &config, int padding = 6) {
      const auto model =
          make_model(make_mesh(true, [&](const mfem::Vector &center) { return center[1] < thickness ? 0 : 1; }, padding),
                     { rho1, rho2 });
      const auto solver =
          solve_dc(*model, { point(0.0, 0.0, 0.0), point(0.0, -120.0, 0.0), point(0.0, -60.0, 0.0) }, config, 4, points, 2);
      const double pole = max_relative_error(*solver, 0, points, [&](const mfem::Vector &sample) {
        return two_layer_surface_potential(rho1, rho2, thickness, std::abs(sample[1]), 4000);
      });
      const std::vector<double> a = solver->potentials(1, { point(0.0, 60.0, 0.0), point(0.0, 120.0, 0.0) });
      const std::vector<double> b = solver->potentials(2, { point(0.0, 60.0, 0.0), point(0.0, 120.0, 0.0) });
      const double response = factor * (a[0] - a[1] - b[0] + b[1]);
      return std::array<double, 2>{ pole, std::abs(response - expected_response) / std::abs(expected_response) };
    };
    const auto default_errors = errors(options(2));
    const auto deficient_min_errors = errors(options(2, 10.0 / 300.0));
    const auto deficient_max_errors = errors(options(2, 1.0e-3 / 300.0, 1.5 / 60.0));
    if (rho2 == 1000.0) {
      DCSolverOptions homogeneous_boundary = options(2);
      homogeneous_boundary.boundary_condition = DCBoundaryCondition::DeyMorrison;
      for (int padding : { 0, 3, 6 }) {
        const auto domain_errors = padding == 6 ? default_errors : errors(options(2), padding);
        const auto homogeneous_errors = errors(homogeneous_boundary, padding);
        if (test_common::is_root()) {
          fmt::print("DC 2.5D rho2={} extent={:.0f} m pole/dipole errors: layered {:.3e}/{:.3e}, Dey-Morrison {:.3e}/{:.3e}\n",
                     rho2, dc_test::stretched(1.0 + static_cast<double>(padding) / dc_test::kGroundCells), domain_errors[0],
                     domain_errors[1], homogeneous_errors[0], homogeneous_errors[1]);
        }
        // The layered background removes the truncation error that a homogeneous one leaves in small domains.
        CHECK(domain_errors[0] < 2.0e-3);
        if (padding == 0) {
          CHECK(homogeneous_errors[0] > 10.0 * domain_errors[0]);
        }
      }
    }
    if (test_common::is_root()) {
      fmt::print("DC 2.5D rho2={} pole/dipole errors: default {:.3e}/{:.3e}, underresolved-low {:.3e}/{:.3e}, "
                 "short-high {:.3e}/{:.3e}\n",
                 rho2, default_errors[0], default_errors[1], deficient_min_errors[0], deficient_min_errors[1],
                 deficient_max_errors[0], deficient_max_errors[1]);
    }
    CHECK(default_errors[0] < 5.0e-3);
    CHECK(default_errors[1] < 5.0e-3);
    CHECK(deficient_min_errors[0] > default_errors[0] + 1.0e-2);
    CHECK(deficient_max_errors[0] > default_errors[0] + 1.0e-2);
  }
}

TEST_CASE("2.5D DC transverse contact agrees with a three-dimensional extrusion", "[dc][25d]") {
  const auto model_2d = make_model(make_mesh(true, [](const mfem::Vector &center) { return center[0] > 0.0 ? 1 : 0; }),
                                   { 100.0, 20.0 });
  const auto model_3d = make_model(make_mesh(false, [](const mfem::Vector &center) { return center[1] > 0.0 ? 1 : 0; }),
                                   { 100.0, 20.0 });
  const std::vector<mfem::Vector> electrodes = { point(0.0, -40.0, 0.0) };
  const std::vector<mfem::Vector> points = { point(0.0, -80.0, 0.0), point(0.0, -10.0, 0.0), point(0.0, 30.0, 0.0),
                                             point(0.0, 90.0, 0.0) };
  const auto solver_2d = solve_dc(*model_2d, electrodes, options(2, 1.0e-3 / 130.0, 15.0 / 30.0, 6), 4, points, 2);
  const auto solver_3d = solve_dc(*model_3d, electrodes, dc_test::dc_options(2), 4, points, 2);
  const std::vector<double> values_2d = solver_2d->potentials(0, points);
  const std::vector<double> values_3d = solver_3d->potentials(0, points);
  for (std::size_t index = 0; index < points.size(); ++index) {
    CAPTURE(index, values_2d[index], values_3d[index]);
    CHECK(values_2d[index] == test_common::approx(values_3d[index], 1.0e-2));
  }
}

TEST_CASE("2.5D DC solver choices agree", "[dc][25d][solver]") {
  const auto model = make_model(make_mesh(true, [](const mfem::Vector &center) { return center[1] > 40.0 ? 1 : 0; }),
                                { 100.0, 10.0 });
  const std::vector<mfem::Vector> electrodes = { point(0.0, -40.0, 0.0), point(0.0, 40.0, 0.0) };
  const std::vector<mfem::Vector> points = { point(0.0, -100.0, 0.0), point(0.0, 0.0, 0.0), point(0.0, 100.0, 0.0) };
  const auto direct = solve_dc(*model, electrodes, options(2), 2);
  const std::vector<double> expected = direct->potentials(0, points);
  for (DCPreconditionerType preconditioner : { DCPreconditionerType::AMG, DCPreconditionerType::LOR }) {
    DCSolverOptions config = options(2);
    config.solver = DCSolverType::CG;
    config.preconditioner = preconditioner;
    config.rtol = 1.0e-12;
    config.max_iter = 500;
    const auto iterative = solve_dc(*model, electrodes, config, 2);
    const std::vector<double> values = iterative->potentials(0, points);
    for (std::size_t index = 0; index < values.size(); ++index) {
      CHECK(values[index] == test_common::approx(expected[index], 1.0e-8));
    }
  }
}
