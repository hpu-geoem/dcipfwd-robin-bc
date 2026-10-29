// Refine a conforming root mesh to an octree (quadtree in 2D) that follows a voxel model, and write the MFEM
// nonconforming mesh, the property table and the leaf cells.

#include "model.h"
#include "octree_model.h"

#include <fmt/format.h>

#include <mfem.hpp>

#include <cmath>
#include <fstream>
#include <map>
#include <sstream>
#include <string>
#include <vector>

using namespace dcip;

namespace {

/// Parse "a,b,..." into numbers.
std::vector<double> parse_list(const std::string &text) {
  std::vector<double> values;
  std::stringstream stream(text);
  std::string item;
  while (std::getline(stream, item, ',')) {
    if (!item.empty()) {
      values.push_back(std::stod(item));
    }
  }
  return values;
}

} // namespace

int main(int argc, char *argv[]) {
  mfem::Mpi::Init(argc, argv);
  const char *mesh_file = "root.mesh", *voxel_prefix = "voxels", *background_file = "background.txt";
  const char *output_prefix = "model", *min_size = "1e30:0", *surface_box = "";
  OctreeSettings settings;
  mfem::OptionsParser args(argc, argv);
  args.AddOption(&mesh_file, "-m", "--mesh", "Conforming root mesh; core cells align with the voxel grid.");
  args.AddOption(&voxel_prefix, "-g", "--voxels", "Voxel model prefix (.txt header and .bin data).");
  args.AddOption(&background_file, "-bg", "--background", "Background rows 'top resistivity [chargeability]'.");
  args.AddOption(&output_prefix, "-o", "--output", "Output prefix (.mesh, .prop, .cells).");
  args.AddOption(&settings.core_attribute, "-core", "--core-attribute", "Attribute of the root cells of the core.");
  args.AddOption(&settings.log10_resistivity_std, "-tol-rho", "--tol-rho",
                 "Refine a cell whose log10 resistivity has a larger standard deviation.");
  args.AddOption(&settings.chargeability_std, "-tol-eta", "--tol-eta",
                 "Refine a cell whose chargeability has a larger standard deviation.");
  args.AddOption(&min_size, "-min-size", "--min-size",
                 "Smallest cell size by depth of the cell top: 'depth:size,...'; the first deeper limit applies.");
  args.AddOption(&surface_box, "-surface-box", "--surface-box", "Horizontal box of the survey: 'min,max[,min,max]'.");
  args.AddOption(&settings.surface_size, "-surface-size", "--surface-size", "Size of surface cells in the box.");
  args.AddOption(&settings.nc_limit, "-nc-limit", "--nc-limit", "Largest level difference of neighbouring cells.");
  args.Parse();
  if (!args.Good()) {
    if (mfem::Mpi::Root()) {
      args.PrintUsage(mfem::out);
    }
    return 1;
  }
  if (!mfem::Mpi::Root()) {
    return 0;
  }

  settings.min_size.clear();
  std::stringstream rules(min_size);
  std::string rule;
  while (std::getline(rules, rule, ',')) {
    const auto colon = rule.find(':');
    MFEM_VERIFY(colon != std::string::npos, fmt::format("Invalid size rule '{}'; use depth:size.", rule));
    settings.min_size.push_back({ std::stod(rule.substr(0, colon)), std::stod(rule.substr(colon + 1)) });
  }
  settings.surface_box = parse_list(surface_box);

  mfem::Mesh mesh(mesh_file, 1, 1);
  const VoxelModel voxels = VoxelModel::read(voxel_prefix);
  const std::vector<OctreeCell> cells = refine_to_voxels(mesh, voxels, read_background(background_file), settings);

  std::ofstream mesh_output(fmt::format("{}.mesh", output_prefix));
  mesh_output.precision(16);
  mesh.Print(mesh_output);

  std::ofstream properties(fmt::format("{}.prop", output_prefix));
  properties << "# attribute resistivity chargeability\n";
  properties.precision(10);
  for (const auto &[attribute, property] : cell_properties(cells)) {
    properties << attribute << ' ' << property.resistivity << ' ' << property.chargeability << '\n';
  }

  std::ofstream cell_output(fmt::format("{}.cells", output_prefix));
  cell_output << "# lower[dim] upper[dim] log10_resistivity chargeability core, one leaf cell per row\n";
  cell_output.precision(10);
  std::map<double, int> core_sizes;
  for (const OctreeCell &cell : cells) {
    for (double value : cell.lower) {
      cell_output << value << ' ';
    }
    for (double value : cell.upper) {
      cell_output << value << ' ';
    }
    cell_output << cell.log10_resistivity << ' ' << cell.chargeability << ' ' << (cell.core ? 1 : 0) << '\n';
    if (cell.core) {
      ++core_sizes[std::round(cell.upper[0] - cell.lower[0])];
    }
  }
  mesh_output.close();
  properties.close();
  cell_output.close();
  if (mesh_output.fail() || properties.fail() || cell_output.fail()) {
    mfem::err << "Writing the octree model failed.\n";
    return 1;
  }
  mfem::out << fmt::format("cells {} core {} vertices {}\n", mesh.GetNE(), [&] {
    int count = 0;
    for (const auto &[size, number] : core_sizes) {
      count += number;
    }
    return count;
  }(), mesh.GetNV());
  for (const auto &[size, number] : core_sizes) {
    mfem::out << fmt::format("core cells of width {} m: {}\n", size, number);
  }
  return 0;
}
