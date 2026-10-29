// 3D DC forward tests: half-space and two-layer potentials, layered truncation, reciprocity, solvers.

#include "test_common.h"

#include <fmt/format.h>

#include <cmath>
#include <memory>
#include <vector>

namespace {

using dc_test::half_space_potential;
using dc_test::make_mesh;
using dc_test::make_model;
using dc_test::max_relative_error;
using dc_test::plane_depth;
using dc_test::solve_dc;
using dc_test::two_layer_surface_potential;

/// Points on the surface and at depth, 20 to 120 m from the origin.
std::vector<mfem::Vector> sample_points() {
  std::vector<mfem::Vector> points;
  for (double distance : { 20.0, 40.0, 80.0, 120.0 }) {
    points.push_back(mfem::Vector({ distance, 0.0, 0.0 }));
    points.push_back(mfem::Vector({ 0.0, -distance, 0.0 }));
    points.push_back(mfem::Vector({ -distance / std::sqrt(2.0), distance / std::sqrt(2.0), 0.0 }));
    points.push_back(mfem::Vector({ 0.6 * distance, 0.0, 0.8 * distance }));
  }
  return points;
}

} // namespace

TEST_CASE("DC potential of a surface electrode matches the homogeneous half-space", "[dc][3d]") {
  const double resistivity = 100.0;
  const auto model = make_model(make_mesh(false, [](const mfem::Vector &) { return 0; }), { resistivity });
  const mfem::Vector electrode({ 0.0, 0.0, 0.0 });
  const auto analytic = [&](const mfem::Vector &point) { return half_space_potential(resistivity, electrode, point); };

  double errors[2];
  for (int order : { 1, 2 }) {
    const auto solver = solve_dc(*model, { electrode }, dc_test::dc_options(order), 5, sample_points(), 2);
    errors[order - 1] = max_relative_error(*solver, 0, sample_points(), analytic);
    // The unit current leaves through the truncation boundary.
    CHECK(solver->source_currents(0).front() == test_common::approx(1.0, 1.0e-9));
  }
  CAPTURE(errors[0], errors[1]);
  CHECK(errors[1] < 4.0e-3);
  CHECK(errors[1] < 0.2 * errors[0]);
}

TEST_CASE("DC potential of a buried electrode matches the half-space image solution", "[dc][3d]") {
  const double resistivity = 50.0;
  const auto model = make_model(make_mesh(false, [](const mfem::Vector &) { return 0; }), { resistivity });
  const mfem::Vector electrode({ 0.0, 0.0, 20.0 });
  const auto solver = solve_dc(*model, { electrode }, dc_test::dc_options(2), 5, sample_points(), 2);
  const double error = max_relative_error(*solver, 0, sample_points(), [&](const mfem::Vector &point) {
    return half_space_potential(resistivity, electrode, point);
  });
  CAPTURE(error);
  CHECK(error < 3.0e-3);
}

TEST_CASE("DC surface potential matches the two-layer earth", "[dc][3d]") {
  const double rho1 = 100.0, thickness = plane_depth(2);
  for (double rho2 : { 10.0, 1000.0 }) {
    CAPTURE(rho2);
    const auto model =
        make_model(make_mesh(false, [&](const mfem::Vector &center) { return center[2] < thickness ? 0 : 1; }), { rho1, rho2 });
    std::vector<mfem::Vector> points;
    for (double distance : { 10.0, 20.0, 40.0, 80.0, 120.0 }) {
      points.push_back(mfem::Vector({ distance, 0.0, 0.0 }));
      points.push_back(mfem::Vector({ -0.6 * distance, 0.8 * distance, 0.0 }));
    }
    const auto analytic = [&](const mfem::Vector &point) {
      return two_layer_surface_potential(rho1, rho2, thickness, point.Norml2());
    };
    const double error = max_relative_error(*solve_dc(*model, { mfem::Vector({ 0.0, 0.0, 0.0 }) }, dc_test::dc_options(2), 5, points, 2),
                                            0, points, analytic);
    CAPTURE(error);
    CHECK(error < 1.5e-2);
  }
}

TEST_CASE("DC layered boundary condition removes the truncation error over a resistive basement", "[dc][3d][boundary]") {
  const double rho1 = 100.0, rho2 = 1000.0, thickness = plane_depth(2);
  const auto index = [&](const mfem::Vector &center) { return center[2] < thickness ? 0 : 1; };
  std::vector<mfem::Vector> points;
  for (double distance : { 20.0, 60.0, 120.0, 240.0 }) {
    points.push_back(mfem::Vector({ distance, 0.0, 0.0 }));
    points.push_back(mfem::Vector({ -0.6 * distance, 0.8 * distance, 0.0 }));
  }
  const auto potentials = [&](int padding, DCBoundaryCondition boundary_condition) {
    const auto model = make_model(make_mesh(false, index, padding), { rho1, rho2 });
    DCSolverOptions config = dc_test::dc_options(2);
    config.boundary_condition = boundary_condition;
    return solve_dc(*model, { mfem::Vector({ 0.0, 0.0, 0.0 }) }, config, 3, points, 1)->potentials(0, points);
  };
  // The same core mesh with a far boundary isolates the truncation error.
  const std::vector<double> reference = potentials(6, DCBoundaryCondition::Layered);
  const std::vector<double> layered = potentials(0, DCBoundaryCondition::Layered);
  const std::vector<double> homogeneous = potentials(0, DCBoundaryCondition::DeyMorrison);
  double layered_error = 0.0, homogeneous_error = 0.0;
  for (std::size_t index = 0; index < points.size(); ++index) {
    layered_error = std::max(layered_error, std::abs(layered[index] / reference[index] - 1.0));
    homogeneous_error = std::max(homogeneous_error, std::abs(homogeneous[index] / reference[index] - 1.0));
  }
  CAPTURE(layered_error, homogeneous_error);
  if (test_common::is_root()) {
    fmt::print("DC 3D truncation over a resistive basement: layered {:.3e}, Dey-Morrison {:.3e}\n", layered_error,
               homogeneous_error);
  }
  CHECK(layered_error < 0.1 * homogeneous_error);
}

TEST_CASE("DC layered boundary condition of a homogeneous earth is the Dey-Morrison condition", "[dc][3d][boundary]") {
  const auto model = make_model(make_mesh(false, [](const mfem::Vector &) { return 0; }), { 100.0 });
  const std::vector<mfem::Vector> electrodes = { mfem::Vector({ -40.0, 0.0, 0.0 }), mfem::Vector({ 40.0, 40.0, 0.0 }) };
  DCSolverOptions homogeneous_boundary = dc_test::dc_options(1);
  homogeneous_boundary.boundary_condition = DCBoundaryCondition::DeyMorrison;
  const auto layered = solve_dc(*model, electrodes, dc_test::dc_options(1), 2);
  const auto homogeneous = solve_dc(*model, electrodes, homogeneous_boundary, 2);
  for (int electrode : { 0, 1 }) {
    const std::vector<double> expected = homogeneous->potentials(electrode, sample_points());
    const std::vector<double> values = layered->potentials(electrode, sample_points());
    for (std::size_t index = 0; index < values.size(); ++index) {
      CHECK(values[index] == test_common::approx(expected[index], 1.0e-10));
    }
  }
}

TEST_CASE("DC electrodes share one symmetric system and have reciprocal potentials", "[dc][3d][solver]") {
  const auto model = make_model(
      make_mesh(false, [](const mfem::Vector &center) { return center[2] > 40.0 && center[0] > 0.0 ? 1 : 0; }),
      { 100.0, 10.0 });
  const std::vector<mfem::Vector> electrodes = { mfem::Vector({ -40.0, 0.0, 0.0 }), mfem::Vector({ 40.0, 20.0, 0.0 }),
                                                 mfem::Vector({ 0.0, -60.0, 10.0 }) };
  const auto solver = solve_dc(*model, electrodes, dc_test::dc_options(2), 3);
  // V_i(x_j) = V_j(x_i) when every electrode uses the same symmetric matrix.
  for (int i = 0; i < 3; ++i) {
    for (int j = i + 1; j < 3; ++j) {
      const double forward = solver->potentials(i, { electrodes[j] }).front();
      const double backward = solver->potentials(j, { electrodes[i] }).front();
      CAPTURE(i, j, forward, backward);
      CHECK(forward == test_common::approx(backward, 1.0e-9));
    }
  }
}

TEST_CASE("DC solver choices agree in three dimensions", "[dc][3d][solver]") {
  const auto model = make_model(
      make_mesh(false, [](const mfem::Vector &center) { return center[2] > 40.0 && center[0] > 0.0 ? 1 : 0; }),
      { 100.0, 10.0 });
  const std::vector<mfem::Vector> electrodes = { mfem::Vector({ -40.0, 0.0, 0.0 }), mfem::Vector({ 40.0, 20.0, 0.0 }) };
  const auto direct = solve_dc(*model, electrodes, dc_test::dc_options(2), 2);
  const std::vector<double> expected = direct->potentials(0, sample_points());
  for (DCPreconditionerType preconditioner : { DCPreconditionerType::AMG, DCPreconditionerType::LOR }) {
    DCSolverOptions options = dc_test::dc_options(2);
    options.solver = DCSolverType::CG;
    options.preconditioner = preconditioner;
    options.rtol = 1.0e-12;
    options.max_iter = 500;
    const auto iterative = solve_dc(*model, electrodes, options, 2);
    const std::vector<double> values = iterative->potentials(0, sample_points());
    if (test_common::is_root()) {
      fmt::print("DC 3D order 2 CG iterations with {}: {}\n", preconditioner == DCPreconditionerType::AMG ? "AMG" : "LOR-AMG",
                 iterative->stats().max_iterations);
    }
    for (std::size_t index = 0; index < values.size(); ++index) {
      CHECK(values[index] == test_common::approx(expected[index], 1.0e-8));
    }
  }
}
