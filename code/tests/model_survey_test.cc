// Tests of the model, background column and survey inputs.

#include "survey.h"
#include "test_common.h"

#include <cstdio>
#include <filesystem>
#include <fstream>

namespace {

std::filesystem::path temporary_file(const std::string &name, const std::string &content) {
  const std::filesystem::path path = std::filesystem::temp_directory_path() / ("dcipfwd_test_" + name);
  std::ofstream(path) << content;
  return path;
}

} // namespace

TEST_CASE("The corner column of a layered model lists its layers", "[model]") {
  const double interface = dc_test::plane_depth(3);
  const auto index = [&](const mfem::Vector &center) {
    if (std::abs(center[0]) < 100.0 && center[2] < 200.0) {
      return 2; // A block away from the corners
    }
    return center[2] < interface ? 0 : 1;
  };
  for (bool spectral : { false, true }) {
    CAPTURE(spectral);
    const auto spectral_index = [&](const mfem::Vector &center) {
      return spectral ? index(mfem::Vector({ center[0], 0.0, center[1] })) : index(center);
    };
    const auto model = dc_test::make_model(dc_test::make_mesh(spectral, spectral_index), { 100.0, 1000.0, 5.0 },
                                           { 0.0, 0.1, 0.2 });
    const std::vector<BackgroundLayer> layers = model->corner_background();
    REQUIRE(layers.size() == 2);
    CHECK(layers[0].top == test_common::approx(0.0, 0.0, 1.0e-9));
    CHECK(layers[0].sigma == test_common::approx(0.01, 1.0e-12));
    CHECK(layers[1].top == test_common::approx(interface, 1.0e-12));
    CHECK(layers[1].sigma == test_common::approx(0.001, 1.0e-12));
    CHECK(layers[1].eta == test_common::approx(0.1, 1.0e-12));
    CHECK(model->has_chargeability());
  }
}

TEST_CASE("Survey files give electrodes, measurements and image geometric factors", "[survey]") {
  const auto path = temporary_file("survey.txt", "# electrodes\n4\n0 0 0\n10 0 0\n20 0 0\n30 0 0\n"
                                                 "# measurements\n2\n0 3 1 2  # Wenner\n0 -1 1 -1\n");
  const Survey survey(path.string());
  REQUIRE(survey.electrodes().size() == 4);
  REQUIRE(survey.measurements().size() == 2);
  CHECK(survey.current_electrodes() == std::vector<int>{ 0, 3 });
  CHECK(survey.potential_electrodes() == std::vector<int>{ 1, 2 });
  // Wenner with spacing a: K = 2 pi a; pole-pole: K = 2 pi r.
  CHECK(survey.geometric_factor(survey.measurements()[0], 0.0) == test_common::approx(2.0 * PI * 10.0, 1.0e-12));
  CHECK(survey.geometric_factor(survey.measurements()[1], 0.0) == test_common::approx(2.0 * PI * 10.0, 1.0e-12));
  std::filesystem::remove(path);
}

TEST_CASE("Background files list layer tops, resistivities and chargeabilities", "[model]") {
  const auto path = temporary_file("background.txt", "# top rho eta\n0 100\n50 1000 0.05\n");
  const std::vector<BackgroundLayer> layers = read_background(path.string());
  REQUIRE(layers.size() == 2);
  CHECK(layers[1].top == 50.0);
  CHECK(layers[1].sigma == test_common::approx(1.0e-3, 1.0e-14));
  CHECK(layers[1].eta == 0.05);
  CHECK(column_conductivity(layers, 10.0) == test_common::approx(0.01, 1.0e-14));
  std::filesystem::remove(path);
}

TEST_CASE("Background columns find sloping interfaces between mesh vertices", "[model]") {
  // Two quadrilaterals in the (y, z) plane share the edge z = 1 + y.
  mfem::Mesh mesh(2, 6, 2, 0, 2);
  for (const auto &[y, z] : std::vector<std::pair<double, double>>{ { 0, 0 }, { 1, 0 }, { 1, 2 }, { 0, 1 }, { 1, 3 }, { 0, 3 } }) {
    const double vertex[2] = { y, z };
    mesh.AddVertex(vertex);
  }
  const int upper[4] = { 0, 1, 2, 3 }, lower[4] = { 3, 2, 4, 5 };
  mesh.AddQuad(upper, 1);
  mesh.AddQuad(lower, 2);
  mesh.FinalizeQuadMesh(1, 1, true);
  const Model model(std::move(mesh), { { 1, { 100.0, 0.0 } }, { 2, { 10.0, 0.0 } } });
  const std::vector<BackgroundLayer> layers = model.column(mfem::Vector({ 0.25 }));
  REQUIRE(layers.size() == 2);
  CHECK(layers[1].top == test_common::approx(1.25, 1.0e-6));
}

TEST_CASE("Background columns start where the vertical line enters a mesh under topography", "[model]") {
  // One quadrilateral in the (y, z) plane whose top rises from z = 0 at y = 0 to z = 1 at y = 1.
  mfem::Mesh mesh(2, 4, 1, 0, 2);
  for (const auto &[y, z] : std::vector<std::pair<double, double>>{ { 0, 0 }, { 1, 1 }, { 1, 3 }, { 0, 3 } }) {
    const double vertex[2] = { y, z };
    mesh.AddVertex(vertex);
  }
  const int quad[4] = { 0, 1, 2, 3 };
  mesh.AddQuad(quad, 1);
  mesh.FinalizeQuadMesh(1, 1, true);
  const Model model(std::move(mesh), { { 1, { 100.0, 0.0 } } });
  const std::vector<BackgroundLayer> layers = model.column(mfem::Vector({ 1.0 }));
  REQUIRE(layers.size() == 1);
  CHECK(layers[0].top == test_common::approx(1.0, 0.0, 1.0e-5));
}
