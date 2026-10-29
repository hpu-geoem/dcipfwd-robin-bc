// Tests of octree (quadtree) models refined to follow a voxel model.

#include "octree_model.h"
#include "test_common.h"

#include <fmt/format.h>

#include <cmath>
#include <sstream>
#include <vector>

namespace {

/// Two-layer voxel model: log10 resistivity and chargeability above and below an interface depth.
VoxelModel two_layer_voxels(const std::vector<double> &origin, double spacing, const std::vector<int> &shape,
                            double interface, double log10_rho1, double log10_rho2, double eta1, double eta2) {
  VoxelModel voxels;
  voxels.origin = origin;
  voxels.spacing = spacing;
  voxels.shape = shape;
  std::size_t count = 1;
  for (int n : shape) {
    count *= static_cast<std::size_t>(n);
  }
  const std::size_t layer_stride = count / static_cast<std::size_t>(shape.back());
  for (std::size_t index = 0; index < count; ++index) {
    const double depth = origin.back() + spacing * (static_cast<double>(index / layer_stride) + 0.5);
    voxels.log10_resistivity.push_back(depth < interface ? log10_rho1 : log10_rho2);
    voxels.chargeability.push_back(depth < interface ? eta1 : eta2);
  }
  return voxels;
}

double cell_size(const OctreeCell &cell) {
  double size = 0.0;
  for (std::size_t axis = 0; axis < cell.lower.size(); ++axis) {
    size = std::max(size, cell.upper[axis] - cell.lower[axis]);
  }
  return size;
}

} // namespace

TEST_CASE("Octree cells follow a voxel interface, size rules and the background", "[octree]") {
  // 600 x 600 x 500 m of 100 m root cells; the inner 400 x 400 x 400 m are the core on a 25 m voxel grid.
  mfem::Mesh mesh = mfem::Mesh::MakeCartesian3D(6, 6, 5, mfem::Element::HEXAHEDRON, 600.0, 600.0, 500.0, false);
  test_common::assign_attributes(mesh, [](const mfem::Vector &center) {
    return center[0] > 100.0 && center[0] < 500.0 && center[1] > 100.0 && center[1] < 500.0 && center[2] < 400.0 ? 1 : 2;
  });
  const double interface = 125.0;
  const VoxelModel voxels = two_layer_voxels({ 100.0, 100.0, 0.0 }, 25.0, { 16, 16, 16 }, interface, 2.0, 3.0, 0.05, 0.1);
  const std::vector<BackgroundLayer> background = { { 0.0, 0.01, 0.05 }, { interface, 0.001, 0.1 } };
  OctreeSettings settings;
  settings.min_size = { { 200.0, 25.0 }, { 1.0e30, 50.0 } };
  settings.surface_box = { 200.0, 400.0, 200.0, 400.0 };
  settings.surface_size = 50.0;
  const std::vector<OctreeCell> cells = refine_to_voxels(mesh, voxels, background, settings);

  REQUIRE(static_cast<int>(cells.size()) == mesh.GetNE());
  double volume = 0.0;
  int fine = 0;
  for (const OctreeCell &cell : cells) {
    volume += (cell.upper[0] - cell.lower[0]) * (cell.upper[1] - cell.lower[1]) * (cell.upper[2] - cell.lower[2]);
    const double depth = 0.5 * (cell.lower[2] + cell.upper[2]);
    if (cell.core) {
      // Every core leaf lies on one side of the interface and takes its voxel values.
      CHECK(cell.log10_resistivity == test_common::approx(depth < interface ? 2.0 : 3.0, 0.0, 1.0e-12));
      CHECK(cell.chargeability == test_common::approx(depth < interface ? 0.05 : 0.1, 0.0, 1.0e-12));
      CHECK(cell_size(cell) >= (cell.lower[2] < 200.0 ? 25.0 : 50.0) - 1.0e-9);
      fine += cell_size(cell) < 26.0 ? 1 : 0;
      const bool in_box = cell.upper[0] > 200.0 && cell.lower[0] < 400.0 && cell.upper[1] > 200.0 && cell.lower[1] < 400.0;
      if (cell.lower[2] == 0.0 && in_box) {
        CHECK(cell_size(cell) <= 50.0 + 1.0e-9);
      }
    } else {
      // Cells outside the core take the background layer at their centre.
      CHECK(cell.log10_resistivity == test_common::approx(depth < interface ? 2.0 : 3.0, 0.0, 1.0e-12));
      CHECK(mesh.GetAttribute(&cell - cells.data()) == cell.attribute);
    }
  }
  CHECK(volume == test_common::approx(600.0 * 600.0 * 500.0, 1.0e-12));
  CHECK(fine > 0);
  CHECK(mesh.GetNE() < 4096 + 6 * 6 * 5);
  CHECK(mesh.Nonconforming());

  // The nonconforming mesh and its leaf attributes survive writing and reading.
  std::stringstream stream;
  stream.precision(16);
  mesh.Print(stream);
  mfem::Mesh read(stream, 1, 1);
  REQUIRE(read.GetNE() == mesh.GetNE());
  for (int element = 0; element < read.GetNE(); ++element) {
    CHECK(read.GetAttribute(element) == mesh.GetAttribute(element));
  }
}

TEST_CASE("2.5D DC potentials on a quadtree two-layer model match the image series", "[octree][dc][25d]") {
  // 6 x 3 km of 375 m root cells, all core; the voxels are 375 / 32 m and the interface lies on a voxel face.
  mfem::Mesh mesh = mfem::Mesh::MakeCartesian2D(16, 8, mfem::Element::QUADRILATERAL, false, 6000.0, 3000.0, false);
  mesh.Transform([](const mfem::Vector &input, mfem::Vector &output) {
    output.SetSize(2);
    output[0] = input[0] - 3000.0;
    output[1] = input[1];
  });
  test_common::assign_attributes(mesh, [](const mfem::Vector &) { return 1; });
  const double spacing = 375.0 / 32.0, thickness = 4.0 * spacing, rho1 = 100.0, rho2 = 1000.0;
  const VoxelModel voxels = two_layer_voxels({ -3000.0, 0.0 }, spacing, { 512, 256 }, thickness, 2.0, 3.0, 0.0, 0.0);
  OctreeSettings settings;
  settings.surface_box = { -300.0, 300.0 };
  settings.surface_size = 2.0 * spacing;
  const std::vector<OctreeCell> cells =
      refine_to_voxels(mesh, voxels, { { 0.0, 1.0 / rho1, 0.0 }, { thickness, 1.0 / rho2, 0.0 } }, settings);
  CHECK(mesh.GetNE() < 512 * 256 / 20);
  mesh.EnsureNodes();
  const Model model(std::move(mesh), cell_properties(cells));

  DCSolverOptions options = dc_test::dc_options(2);
  options.kx_min = 1.0e-3 / 300.0;
  options.kx_max = 15.0 / 20.0;
  options.kx_points_per_decade = 4;
  const std::vector<mfem::Vector> points = { mfem::Vector({ 0.0, -240.0, 0.0 }), mfem::Vector({ 0.0, -120.0, 0.0 }),
                                             mfem::Vector({ 0.0, 20.0, 0.0 }),   mfem::Vector({ 0.0, 60.0, 0.0 }),
                                             mfem::Vector({ 0.0, 150.0, 0.0 }) };
  const auto solver = dc_test::solve_dc(model, { mfem::Vector({ 0.0, 0.0, 0.0 }) }, options, 4, points, 2);
  const double error = dc_test::max_relative_error(*solver, 0, points, [&](const mfem::Vector &sample) {
    return dc_test::two_layer_surface_potential(rho1, rho2, thickness, std::abs(sample[1]), 4000);
  });
  if (test_common::is_root()) {
    fmt::print("Quadtree two-layer 2.5D: {} cells, largest surface potential error {:.3e}\n", model.mesh().GetNE(), error);
  }
  CHECK(error < 5.0e-3);
}

TEST_CASE("Octree cells that cannot be split on voxel faces keep the mean of their voxels", "[octree]") {
  // One 75 m square over three 25 m voxels per axis with an interface after the first voxel row.
  mfem::Mesh mesh = mfem::Mesh::MakeCartesian2D(1, 1, mfem::Element::QUADRILATERAL, false, 75.0, 75.0, false);
  test_common::assign_attributes(mesh, [](const mfem::Vector &) { return 1; });
  const VoxelModel voxels = two_layer_voxels({ 0.0, 0.0 }, 25.0, { 3, 3 }, 25.0, 2.0, 3.0, 0.0, 0.1);
  OctreeSettings settings;
  settings.log10_resistivity_std = 0.01;
  const std::vector<OctreeCell> cells = refine_to_voxels(mesh, voxels, { { 0.0, 0.01, 0.0 } }, settings);
  REQUIRE(cells.size() == 1);
  CHECK(cells[0].log10_resistivity == test_common::approx((2.0 + 2.0 * 3.0) / 3.0, 1.0e-12));
  CHECK(cells[0].chargeability == test_common::approx(0.2 / 3.0, 1.0e-12));
}
