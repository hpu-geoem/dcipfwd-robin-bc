// Meshes, models and analytic potentials shared by the unit tests.

#pragma once

#include "dc_boundary.h"
#include "dc_solver.h"
#include "model.h"
#include "utils.h"

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include <mfem.hpp>

#include <algorithm>
#include <cmath>
#include <functional>
#include <map>
#include <memory>
#include <vector>

using namespace dcip;

namespace test_common {

inline Catch::Approx approx(double expected, double rel_tol = 0.0, double abs_tol = 1e-12) {
  Catch::Approx comparison(expected);
  comparison.margin(abs_tol);
  comparison.epsilon(rel_tol);
  return comparison;
}

inline bool is_root() {
  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  return rank == 0;
}

inline mfem::Vector element_center(mfem::Mesh &mesh, int element) {
  mfem::Vector center(mesh.SpaceDimension());
  mesh.GetElementTransformation(element)->Transform(mfem::Geometries.GetCenter(mesh.GetElementGeometry(element)), center);
  return center;
}

template <typename AttributeFunc> inline void assign_attributes(mfem::Mesh &mesh, const AttributeFunc &attribute_for_point) {
  for (int element = 0; element < mesh.GetNE(); ++element) {
    mesh.SetAttribute(element, attribute_for_point(element_center(mesh, element)));
  }
  mesh.SetAttributes();
}

} // namespace test_common

namespace dc_test {

constexpr double kExtent = 3000.0; // Half width and depth of the ground
constexpr double kStretch = 5.0;
constexpr int kHorizontalCells = 24, kGroundCells = 12;

/// Coordinate of a mesh plane at the fraction of the extent, with cells growing away from zero.
inline double stretched(double fraction) { return kExtent * std::sinh(kStretch * fraction) / std::sinh(kStretch); }

/// Depth of the mesh plane below a number of ground cells.
inline double plane_depth(int cells) { return stretched(static_cast<double>(cells) / kGroundCells); }

/// Ground mesh in 3D or, when spectral, in the y-z plane, with optional outer padding.
/// Attributes are 1 + the index returned for each cell center.
inline mfem::Mesh make_mesh(bool spectral, const std::function<int(const mfem::Vector &)> &index, int padding = 0,
                            int base_horizontal_cells = kHorizontalCells) {
  const int horizontal_cells = base_horizontal_cells + 2 * padding;
  const int vertical_cells = kGroundCells + padding;
  mfem::Mesh mesh = spectral ? mfem::Mesh::MakeCartesian2D(horizontal_cells, vertical_cells, mfem::Element::QUADRILATERAL, true,
                                                           1.0, 1.0, false)
                             : mfem::Mesh::MakeCartesian3D(horizontal_cells, horizontal_cells, vertical_cells,
                                                           mfem::Element::HEXAHEDRON, 1.0, 1.0, 1.0, false);
  const int dim = spectral ? 2 : 3;
  mesh.Transform([&](const mfem::Vector &input, mfem::Vector &output) {
    output.SetSize(dim);
    for (int axis = 0; axis + 1 < dim; ++axis) {
      output[axis] =
          stretched((input[axis] * horizontal_cells - (base_horizontal_cells / 2 + padding)) / (base_horizontal_cells / 2));
    }
    output[dim - 1] = stretched(input[dim - 1] * vertical_cells / kGroundCells);
  });
  test_common::assign_attributes(mesh, [&](const mfem::Vector &center) { return index(center) + 1; });
  mesh.EnsureNodes();
  mesh.EnsureNCMesh(false);
  return mesh;
}

/// Model of a mesh whose attribute i + 1 has resistivities[i] and chargeabilities[i] (zero when not given).
inline std::unique_ptr<Model> make_model(mfem::Mesh mesh, const std::vector<double> &resistivities,
                                         const std::vector<double> &chargeabilities = {}) {
  std::map<int, CellProperty> properties;
  for (std::size_t index = 0; index < resistivities.size(); ++index) {
    properties[static_cast<int>(index) + 1] = { resistivities[index],
                                                index < chargeabilities.size() ? chargeabilities[index] : 0.0 };
  }
  return std::make_unique<Model>(std::move(mesh), properties);
}

/// Return the direct-solver options used by the DC unit tests.
inline DCSolverOptions dc_options(int order = 1) {
  DCSolverOptions options;
  options.order = order;
  options.solver = DCSolverType::Mumps;
  options.rtol = 1.0e-12;
  return options;
}

/// Centroid of a set of points.
inline mfem::Vector centroid(const std::vector<mfem::Vector> &points) {
  mfem::Vector result(points.front().Size());
  result = 0.0;
  for (const mfem::Vector &point : points) {
    result += point;
  }
  result /= static_cast<double>(points.size());
  return result;
}

/// Solve the unit-current potentials of electrodes with the boundary reference at their centroid and the corner background.
inline std::unique_ptr<DCSolver> solve_dc(const Model &model, const std::vector<mfem::Vector> &electrodes,
                                          const DCSolverOptions &options, int electrode_refinements,
                                          const std::vector<mfem::Vector> &points = {}, int point_refinements = 0,
                                          bool chargeable = false) {
  auto boundary = std::make_shared<const DCBoundary>(model.dim(), model.corner_background(), centroid(electrodes));
  auto solver = std::make_unique<DCSolver>(MPI_COMM_WORLD, model, options, boundary, chargeable);
  solver->add_points_to_refine(electrodes, electrode_refinements);
  solver->add_points_to_refine(points, point_refinements);
  solver->refine_mesh();
  solver->setup_system();
  for (std::size_t electrode = 0; electrode < electrodes.size(); ++electrode) {
    solver->add_electrode(static_cast<int>(electrode), electrodes[electrode]);
  }
  solver->solve();
  return solver;
}

/// Largest relative deviation of a sampled potential from an analytic one.
inline double max_relative_error(const DCSolver &solver, int electrode, const std::vector<mfem::Vector> &points,
                                 const std::function<double(const mfem::Vector &)> &analytic) {
  const std::vector<double> values = solver.potentials(electrode, points);
  double error = 0.0;
  for (std::size_t index = 0; index < points.size(); ++index) {
    error = std::max(error, std::abs(values[index] - analytic(points[index])) / std::abs(analytic(points[index])));
  }
  return error;
}

/// Half-space potential of a point current electrode and its surface image.
inline double half_space_potential(double resistivity, const mfem::Vector &source, const mfem::Vector &point) {
  mfem::Vector image(source);
  image[2] = -source[2];
  return resistivity / (4.0 * PI) * (1.0 / point.DistanceTo(source) + 1.0 / point.DistanceTo(image));
}

/// Surface potential of a layer over a half-space from an image series.
inline double two_layer_surface_potential(double rho1, double rho2, double thickness, double distance, int n_images = 2000) {
  const double reflection = (rho2 - rho1) / (rho2 + rho1);
  double sum = 1.0 / distance, factor = 1.0;
  for (int image = 1; image <= n_images; ++image) {
    factor *= reflection;
    sum += 2.0 * factor / std::hypot(distance, 2.0 * image * thickness);
  }
  return rho1 / (2.0 * PI) * sum;
}

} // namespace dc_test
