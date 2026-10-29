// Per-electrode references and per-electrode boundary data against separate electrode matrices and far boundaries;
// Robin tables against direct evaluation.

#include "test_common.h"

#include <fmt/format.h>

#include <cmath>
#include <map>
#include <memory>
#include <string>
#include <vector>

namespace {

using dc_test::make_mesh;
using dc_test::make_model;

constexpr double kRho1 = 100.0, kRho2 = 1000.0;

/// Solve electrodes with one solver on the mesh refined around all survey electrodes; the shared reference is the given
/// point.
std::unique_ptr<DCSolver> solve(const Model &model, const std::vector<mfem::Vector> &survey,
                                const std::vector<mfem::Vector> &electrodes, const DCSolverOptions &options,
                                const mfem::Vector &reference) {
  auto boundary = std::make_shared<const DCBoundary>(model.dim(), model.corner_background(), reference);
  auto solver = std::make_unique<DCSolver>(MPI_COMM_WORLD, model, options, boundary);
  solver->add_points_to_refine(survey, 2);
  solver->refine_mesh();
  solver->setup_system();
  for (std::size_t electrode = 0; electrode < electrodes.size(); ++electrode) {
    solver->add_electrode(static_cast<int>(electrode), electrodes[electrode]);
  }
  solver->solve();
  return solver;
}

DCSolverOptions options(bool spectral, DCBoundaryCondition condition) {
  DCSolverOptions result = dc_test::dc_options(2);
  result.boundary_condition = condition;
  if (spectral) {
    result.kx_min = 1.0e-3 / 300.0;
    result.kx_max = 15.0 / 20.0;
  }
  return result;
}

mfem::Vector line_point(bool spectral, double coordinate) {
  return spectral ? mfem::Vector({ 0.0, coordinate, 0.0 }) : mfem::Vector({ coordinate, 0.0, 0.0 });
}

/// Compare per-electrode references through the shared preconditioner with one solver per electrode.
void check_against_electrode_matrices(bool spectral, DCBoundaryCondition condition, DCRobinTable table) {
  const double thickness = dc_test::plane_depth(2);
  const auto index = [&](const mfem::Vector &center) { return center[center.Size() - 1] < thickness ? 0 : 1; };
  const auto model = make_model(make_mesh(spectral, index), { kRho1, kRho2 });
  const std::vector<mfem::Vector> electrodes = { line_point(spectral, -100.0), line_point(spectral, 20.0),
                                                 line_point(spectral, 80.0) };
  const std::vector<mfem::Vector> points = { line_point(spectral, -60.0), line_point(spectral, 0.0), line_point(spectral, 140.0),
                                             line_point(spectral, 300.0) };

  // Per-electrode references, MUMPS factorization of the centroid matrix as the preconditioner of CG.
  DCSolverOptions per_electrode = options(spectral, condition);
  per_electrode.reference = DCBoundaryReference::PerElectrode;
  per_electrode.robin_table = table;
  const auto shared_preconditioner = solve(*model, electrodes, electrodes, per_electrode, dc_test::centroid(electrodes));
  CHECK(shared_preconditioner->stats().max_iterations <= 10);
  if (condition == DCBoundaryCondition::Layered && table == DCRobinTable::Always) {
    CHECK(shared_preconditioner->stats().robin_tables >= 1);
  }

  for (std::size_t electrode = 0; electrode < electrodes.size(); ++electrode) {
    CAPTURE(spectral, electrode);
    const auto own = solve(*model, electrodes, { electrodes[electrode] }, options(spectral, condition), electrodes[electrode]);
    const std::vector<double> expected = own->potentials(0, points);
    const std::vector<double> values = shared_preconditioner->potentials(static_cast<int>(electrode), points);
    // Table interpolation (relative error below 1e-6 in the coefficient) changes potentials by about 1e-8.
    const double tolerance = table == DCRobinTable::Never ? 1.0e-8 : 1.0e-7;
    for (std::size_t point = 0; point < points.size(); ++point) {
      CHECK(values[point] == test_common::approx(expected[point], tolerance));
    }
    for (double current : shared_preconditioner->source_currents(static_cast<int>(electrode))) {
      CHECK(current == test_common::approx(1.0, 1.0e-8));
    }
  }
}

/// Largest relative difference of the potentials of all electrodes at the points between two solvers.
double max_difference(const DCSolver &solver, const DCSolver &reference, std::size_t electrodes,
                      const std::vector<mfem::Vector> &points) {
  double worst = 0.0;
  for (std::size_t electrode = 0; electrode < electrodes; ++electrode) {
    const std::vector<double> values = solver.potentials(static_cast<int>(electrode), points);
    const std::vector<double> expected = reference.potentials(static_cast<int>(electrode), points);
    for (std::size_t point = 0; point < points.size(); ++point) {
      worst = std::max(worst, std::abs(values[point] / expected[point] - 1.0));
    }
  }
  return worst;
}

} // namespace

TEST_CASE("Per-electrode layered references match one matrix per electrode in 3D", "[dc][reference][3d]") {
  for (DCRobinTable table : { DCRobinTable::Never, DCRobinTable::Always }) {
    check_against_electrode_matrices(false, DCBoundaryCondition::Layered, table);
  }
}

TEST_CASE("Per-electrode layered references match one matrix per electrode in 2.5D", "[dc][reference][25d]") {
  for (DCRobinTable table : { DCRobinTable::Never, DCRobinTable::Always }) {
    check_against_electrode_matrices(true, DCBoundaryCondition::Layered, table);
  }
}

TEST_CASE("Per-electrode Dey-Morrison references match one matrix per electrode", "[dc][reference]") {
  check_against_electrode_matrices(false, DCBoundaryCondition::DeyMorrison, DCRobinTable::Auto);
}

TEST_CASE("Robin tables reproduce direct layered coefficients", "[dc][reference][table]") {
  // Surface source over a resistive basement; depths and offsets like those of a truncation boundary.
  const std::vector<BackgroundLayer> layers = { { 0.0, 1.0 / kRho1, 0.0 }, { 50.0, 1.0 / kRho2, 0.0 } };
  for (int dim : { 3, 2 }) {
    for (double k : dim == 3 ? std::vector<double>{ 0.0 } : std::vector<double>{ 1.0e-4, 3.0e-3, 0.05 }) {
      const DCBoundary boundary(dim, layers, mfem::Vector({ 0.0, 0.0, 0.0 }));
      std::vector<DepthRange> ranges;
      for (double z : { 0.7, 12.0, 49.0, 51.0, 300.0, 1000.0 }) {
        ranges.push_back({ z, z == 1000.0 ? 0.0 : 400.0, 1500.0 });
      }
      const RobinTable table(boundary, k, false, ranges);
      double worst = 0.0;
      for (const DepthRange &range : ranges) {
        for (double r : { range.r_min, 0.5 * (range.r_min + range.r_max) + 3.7, range.r_max, 777.7 }) {
          mfem::Vector position(dim), normal(dim);
          position = 0.0;
          normal = 0.0;
          position[0] = r;
          position[dim - 1] = range.z;
          // An oblique outward normal exercises both components.
          normal[0] = 0.6;
          normal[dim - 1] = 0.8;
          const double direct = boundary.robin_factor(position, normal, k, false);
          const double interpolated = std::max(0.0, table.ratio(range.z, r, 0.6, 0.8));
          worst = std::max(worst, std::abs(interpolated - direct) / std::max(direct, 1.0e-300));
        }
      }
      CAPTURE(dim, k, worst);
      if (test_common::is_root()) {
        fmt::print("Robin table {}D k={:.1e}: worst relative difference {:.2e}\n", dim, k, worst);
      }
      CHECK(worst < 1.0e-6);
    }
  }
}

TEST_CASE("Robin tables do not interpolate across the image-only range of k L", "[dc][reference][table]") {
  const std::vector<BackgroundLayer> layers = { { 0.0, 1.0 / kRho1, 0.0 }, { 50.0, 1.0 / kRho2, 0.0 } };
  const DCBoundary boundary(2, layers, mfem::Vector({ 0.0, 0.0, 0.0 }));
  const double k = 0.02, z = 49.0;
  const RobinTable table(boundary, k, false, { { z, 400.0, 1500.0 } });
  // k L = 15 at r = sqrt(750^2 - 49^2), about 748.4 m.
  const double r_switch = std::sqrt(750.0 * 750.0 - z * z);
  double worst = 0.0;
  for (double r : { r_switch - 0.1, r_switch - 2.0, r_switch + 0.1, r_switch + 2.0 }) {
    const mfem::Vector position({ r, z }), normal({ 0.6, 0.8 });
    const double direct = boundary.robin_factor(position, normal, k, false);
    worst = std::max(worst, std::abs(table.ratio(z, r, 0.6, 0.8) / direct - 1.0));
  }
  CAPTURE(worst);
  CHECK(worst < 1.0e-6);
}

TEST_CASE("Robin tables switch where the potential does for a source above the background surface", "[dc][reference][table]") {
  const std::vector<BackgroundLayer> layers = { { 0.0, 1.0 / kRho1, 0.0 }, { 50.0, 1.0 / kRho2, 0.0 } };
  // The source lies 200 m above the surface of the background, so its depth is clipped to the surface.
  const DCBoundary boundary(2, layers, mfem::Vector({ 0.0, 0.0, -200.0 }));
  const double k = 0.02, z = 49.0;
  const RobinTable table(boundary, k, false, { { z, 400.0, 1500.0 } });
  const double r_switch = std::sqrt(750.0 * 750.0 - z * z);
  double worst = 0.0;
  for (double r : { r_switch - 0.1, r_switch + 0.1, 600.0, 1000.0 }) {
    const mfem::Vector position({ r, z }), normal({ 0.6, 0.8 });
    worst = std::max(worst, std::abs(table.ratio(z, r, 0.6, 0.8) / boundary.robin_factor(position, normal, k, false) - 1.0));
  }
  CAPTURE(worst);
  CHECK(worst < 1.0e-6);
}

TEST_CASE("Robin tables resolve small offsets for a source above the background surface", "[dc][reference][table]") {
  // Homogeneous 3D background with the source 200 m above its surface; the table grid follows the clipped depths.
  const DCBoundary boundary(3, { { 0.0, 1.0 / kRho1, 0.0 } }, mfem::Vector({ 0.0, 0.0, -200.0 }));
  const double z = 49.0;
  const RobinTable table(boundary, 0.0, false, { { z, 1.0, 1500.0 } });
  double worst = 0.0;
  for (double r : { 1.0, 2.0, 5.0, 20.0, 60.0, 300.0, 1500.0 }) {
    const mfem::Vector position({ r, 0.0, z }), normal({ 1.0, 0.0, 0.0 });
    // Error relative to 1 / L, the scale of the coefficient; a radial normal at small r gives a coefficient near zero.
    const double difference = table.ratio(z, r, 1.0, 0.0) - boundary.robin_ratio(position, normal, 0.0, false);
    worst = std::max(worst, std::abs(difference) * std::hypot(r, z));
  }
  CAPTURE(worst);
  CHECK(worst < 2.0e-6);
}

TEST_CASE("Per-electrode references and boundary data remove the truncation error of a layered earth", "[dc][reference]") {
  const double thickness = dc_test::plane_depth(2);
  for (bool spectral : { false, true }) {
    CAPTURE(spectral);
    const auto index = [&](const mfem::Vector &center) { return center[center.Size() - 1] < thickness ? 0 : 1; };
    const auto small = make_model(make_mesh(spectral, index), { kRho1, kRho2 });
    const auto large = make_model(make_mesh(spectral, index, 6), { kRho1, kRho2 });
    const std::vector<mfem::Vector> electrodes = { line_point(spectral, -100.0), line_point(spectral, 20.0),
                                                   line_point(spectral, 80.0) };
    const std::vector<mfem::Vector> points = { line_point(spectral, -60.0), line_point(spectral, 0.0),
                                               line_point(spectral, 140.0), line_point(spectral, 300.0) };
    const mfem::Vector centroid = dc_test::centroid(electrodes);
    const auto run = [&](const Model &model, DCBoundaryCondition condition, DCBoundaryReference reference) {
      DCSolverOptions config = options(spectral, condition);
      config.reference = reference;
      return solve(model, electrodes, electrodes, config, centroid);
    };
    // The reference shares the core mesh and has a far boundary.
    const auto far = run(*large, DCBoundaryCondition::Layered, DCBoundaryReference::PerElectrode);
    std::map<std::string, double> errors;
    const auto error = [&](const std::string &name, DCBoundaryCondition condition, DCBoundaryReference reference) {
      const auto solver = run(*small, condition, reference);
      errors[name] = max_difference(*solver, *far, electrodes.size(), points);
      for (std::size_t electrode = 0; electrode < electrodes.size(); ++electrode) {
        for (double current : solver->source_currents(static_cast<int>(electrode))) {
          CHECK(current == test_common::approx(1.0, 1.0e-8));
        }
      }
    };
    error("DS", DCBoundaryCondition::DeyMorrison, DCBoundaryReference::Shared);
    error("DE", DCBoundaryCondition::DeyMorrison, DCBoundaryReference::PerElectrode);
    error("LS", DCBoundaryCondition::Layered, DCBoundaryReference::Shared);
    error("LE", DCBoundaryCondition::Layered, DCBoundaryReference::PerElectrode);
    error("LI", DCBoundaryCondition::Layered, DCBoundaryReference::BoundaryData);
    if (test_common::is_root()) {
      fmt::print("Truncation errors {}: DS {:.2e}, DE {:.2e}, LS {:.2e}, LE {:.2e}, LI {:.2e}\n", spectral ? "2.5D" : "3D",
                 errors["DS"], errors["DE"], errors["LS"], errors["LE"], errors["LI"]);
    }
    // Both per-electrode treatments are exact for the background of every electrode (measured: 3D LS 1.2e-4,
    // LE and LI 3.4e-5; 2.5D LS 4.4e-5, LE and LI 2.2e-5; DS and DE 7e-3 and 4.5e-3).
    CHECK(errors["LE"] < 0.6 * errors["LS"]);
    CHECK(errors["LI"] < 0.6 * errors["LS"]);
    CHECK(std::abs(errors["LI"] / errors["LE"] - 1.0) < 0.1);
    CHECK(errors["LS"] < 0.1 * errors["DS"]);
    CHECK(errors["DE"] < 1.05 * errors["DS"]);
  }
}

TEST_CASE("Boundary data from tables match direct evaluation", "[dc][reference][table]") {
  const double thickness = dc_test::plane_depth(2);
  for (bool spectral : { false, true }) {
    CAPTURE(spectral);
    const auto index = [&](const mfem::Vector &center) { return center[center.Size() - 1] < thickness ? 0 : 1; };
    const auto model = make_model(make_mesh(spectral, index), { kRho1, kRho2 });
    const std::vector<mfem::Vector> electrodes = { line_point(spectral, -100.0), line_point(spectral, 20.0) };
    const std::vector<mfem::Vector> points = { line_point(spectral, -60.0), line_point(spectral, 140.0) };
    DCSolverOptions config = options(spectral, DCBoundaryCondition::Layered);
    config.reference = DCBoundaryReference::BoundaryData;
    config.robin_table = DCRobinTable::Never;
    const auto direct = solve(*model, electrodes, electrodes, config, dc_test::centroid(electrodes));
    config.robin_table = DCRobinTable::Always;
    const auto tabulated = solve(*model, electrodes, electrodes, config, dc_test::centroid(electrodes));
    CHECK(tabulated->stats().robin_tables >= 1);
    CHECK(max_difference(*tabulated, *direct, electrodes.size(), points) < 1.0e-7);
  }
}

TEST_CASE("Layered boundary values are the potential in 3D and its full cosine transform in 2.5D", "[dc][reference]") {
  const double sigma = 1.0 / kRho1;
  const DCBoundary three(3, { { 0.0, sigma, 0.0 } }, mfem::Vector({ 0.0, 0.0, 0.0 }));
  for (double r : { 30.0, 700.0 }) {
    double log_value = 0.0;
    const double ratio = three.robin_ratio(mfem::Vector({ r, 0.0, 40.0 }), mfem::Vector({ 1.0, 0.0, 0.0 }), 0.0, false, &log_value);
    const double distance = std::hypot(r, 40.0);
    CHECK(std::exp(log_value) == test_common::approx(1.0 / (2.0 * PI * sigma * distance), 1.0e-8));
    CHECK(ratio == test_common::approx(r / (distance * distance), 1.0e-8));
  }
  // Below and above the decay switch: K0(k L) / (pi sigma) for a surface source on a half-space.
  const DCBoundary two(2, { { 0.0, sigma, 0.0 } }, mfem::Vector({ 0.0, 0.0, 0.0 }));
  const double k = 0.02;
  for (double r : { 300.0, 700.0, 900.0 }) {
    double log_value = 0.0;
    two.robin_ratio(mfem::Vector({ r, 0.0 }), mfem::Vector({ 1.0, 0.0 }), k, false, &log_value);
    CAPTURE(r, k * r);
    CHECK(log_value == test_common::approx(std::log(bessel_k0(k * r) / (PI * sigma)), 1.0e-7));
  }
}

TEST_CASE("Robin tables reproduce ln G and follow the decay switch", "[dc][reference][table]") {
  const std::vector<BackgroundLayer> layers = { { 0.0, 1.0 / kRho1, 0.0 }, { 50.0, 1.0 / kRho2, 0.0 } };
  for (int dim : { 3, 2 }) {
    for (double decay_switch : { LayeredPotential::kDecaySwitch, 20.0 }) {
      const DCBoundary boundary(dim, layers, mfem::Vector({ 0.0, 0.0, 0.0 }), decay_switch);
      const double k = dim == 3 ? 0.0 : 0.02, z = 49.0;
      const RobinTable table(boundary, k, false, { { z, 400.0, 1500.0 } }, MPI_COMM_SELF, true);
      const double r_switch = std::sqrt(std::pow(decay_switch / 0.02, 2) - z * z);
      double worst_ratio = 0.0, worst_log = 0.0;
      for (double r : { 450.0, r_switch - 1.0, r_switch + 1.0, 1400.0 }) {
        mfem::Vector position(dim), normal(dim);
        position = 0.0;
        normal = 0.0;
        position[0] = r;
        position[dim - 1] = z;
        normal[0] = 0.6;
        normal[dim - 1] = 0.8;
        double direct_log = 0.0, table_log = 0.0;
        const double direct = boundary.robin_ratio(position, normal, k, false, &direct_log);
        const double interpolated = table.ratio(z, r, 0.6, 0.8, &table_log);
        worst_ratio = std::max(worst_ratio, std::abs(interpolated / direct - 1.0));
        worst_log = std::max(worst_log, std::abs(table_log - direct_log));
      }
      CAPTURE(dim, decay_switch, worst_ratio, worst_log);
      CHECK(worst_ratio < 1.0e-6);
      CHECK(worst_log < 5.0e-6);
    }
  }
}

TEST_CASE("Negative ratios are clipped and reported, and per-electrode coefficients where alpha_0 = 0 are counted",
          "[dc][reference][diagnostics]") {
  // A shared reference beyond the wall x = +kExtent makes -dG/dn / G negative on that wall.
  const auto model = make_model(make_mesh(false, [](const mfem::Vector &) { return 0; }), { kRho1 });
  const std::vector<mfem::Vector> electrodes = { line_point(false, -100.0), line_point(false, 80.0) };
  const mfem::Vector outside({ 2.0 * dc_test::kExtent, 0.0, 0.0 });
  for (DCBoundaryCondition condition : { DCBoundaryCondition::Layered, DCBoundaryCondition::DeyMorrison }) {
    CAPTURE(condition == DCBoundaryCondition::Layered);
    DCSolverOptions config = options(false, condition);
    const auto shared = solve(*model, electrodes, electrodes, config, outside);
    CHECK(shared->stats().clipped_points > 0);
    CHECK(shared->stats().clipped_points < shared->stats().boundary_points);
    config.reference = DCBoundaryReference::PerElectrode;
    config.max_iter = 200;
    const auto per_electrode = solve(*model, electrodes, electrodes, config, outside);
    CHECK(per_electrode->stats().unbounded_points > 0);
    CHECK(per_electrode->stats().electrode_clipped_points == 0);
    for (std::size_t electrode = 0; electrode < electrodes.size(); ++electrode) {
      for (double current : per_electrode->source_currents(static_cast<int>(electrode))) {
        CHECK(current == test_common::approx(1.0, 1.0e-8));
      }
    }
  }
}
