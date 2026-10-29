#include "octree_model.h"

#include "utils.h"

#include <fmt/format.h>

#include <algorithm>
#include <cmath>
#include <fstream>
#include <limits>
#include <sstream>

namespace dcip {

namespace {

/// Mean and standard deviation of the voxels in an index box.
struct VoxelStatistics {
  double mean_log10_resistivity = 0.0, std_log10_resistivity = 0.0;
  double mean_chargeability = 0.0, std_chargeability = 0.0;
};

/// Index box [first, last) of the voxels that a cell overlaps, and whether its faces lie on voxel faces.
std::vector<std::pair<int, int>> voxel_box(const VoxelModel &voxels, const std::vector<double> &lower,
                                           const std::vector<double> &upper, bool &aligned) {
  std::vector<std::pair<int, int>> box(voxels.dim());
  aligned = true;
  for (int axis = 0; axis < voxels.dim(); ++axis) {
    const double first = (lower[axis] - voxels.origin[axis]) / voxels.spacing;
    const double last = (upper[axis] - voxels.origin[axis]) / voxels.spacing;
    aligned = aligned && std::abs(first - std::round(first)) < 1.0e-6 && std::abs(last - std::round(last)) < 1.0e-6;
    box[axis] = { static_cast<int>(std::floor(first + 1.0e-6)), static_cast<int>(std::ceil(last - 1.0e-6)) };
    MFEM_VERIFY(box[axis].first >= 0 && box[axis].second <= voxels.shape[axis] && box[axis].first < box[axis].second,
                fmt::format("A core cell along axis {} ({} to {}) leaves the voxel grid.", axis, lower[axis], upper[axis]));
  }
  return box;
}

VoxelStatistics statistics(const VoxelModel &voxels, const std::vector<std::pair<int, int>> &box) {
  double sum_r = 0.0, sum_rr = 0.0, sum_e = 0.0, sum_ee = 0.0;
  long count = 0;
  const int nx = voxels.shape[0], ny = voxels.dim() == 3 ? voxels.shape[1] : 1;
  const int k_first = voxels.dim() == 3 ? box[2].first : 0, k_last = voxels.dim() == 3 ? box[2].second : 1;
  for (int k = k_first; k < k_last; ++k) {
    for (int j = box[1].first; j < box[1].second; ++j) {
      for (int i = box[0].first; i < box[0].second; ++i) {
        const std::size_t index = static_cast<std::size_t>(i) + static_cast<std::size_t>(nx) * (j + static_cast<std::size_t>(ny) * k);
        const double r = voxels.log10_resistivity[index], e = voxels.chargeability[index];
        sum_r += r;
        sum_rr += r * r;
        sum_e += e;
        sum_ee += e * e;
        ++count;
      }
    }
  }
  VoxelStatistics result;
  result.mean_log10_resistivity = sum_r / count;
  result.mean_chargeability = sum_e / count;
  result.std_log10_resistivity = std::sqrt(std::max(0.0, sum_rr / count - result.mean_log10_resistivity * result.mean_log10_resistivity));
  result.std_chargeability = std::sqrt(std::max(0.0, sum_ee / count - result.mean_chargeability * result.mean_chargeability));
  return result;
}

/// Bounding box of an element of a mesh; with check, the element must be an axis-aligned rectangle or box.
void element_box(const mfem::Mesh &mesh, int element, std::vector<double> &lower, std::vector<double> &upper,
                 bool check = false) {
  const int dim = mesh.SpaceDimension();
  lower.assign(dim, std::numeric_limits<double>::max());
  upper.assign(dim, std::numeric_limits<double>::lowest());
  mfem::Array<int> vertices;
  mesh.GetElementVertices(element, vertices);
  for (int vertex : vertices) {
    const double *point = mesh.GetVertex(vertex);
    for (int axis = 0; axis < dim; ++axis) {
      lower[axis] = std::min(lower[axis], point[axis]);
      upper[axis] = std::max(upper[axis], point[axis]);
    }
  }
  if (!check) {
    return;
  }
  const mfem::Geometry::Type geometry = mesh.GetElementGeometry(element);
  MFEM_VERIFY((dim == 2 && geometry == mfem::Geometry::SQUARE) || (dim == 3 && geometry == mfem::Geometry::CUBE),
              "Core cells of an octree model must be quadrilaterals (2D) or hexahedra (3D).");
  std::vector<int> corners;
  for (int vertex : vertices) {
    const double *point = mesh.GetVertex(vertex);
    int corner = 0;
    for (int axis = 0; axis < dim; ++axis) {
      const double tolerance = 1.0e-9 * std::max(1.0, upper[axis] - lower[axis]);
      const bool at_lower = std::abs(point[axis] - lower[axis]) <= tolerance;
      MFEM_VERIFY(at_lower || std::abs(point[axis] - upper[axis]) <= tolerance,
                  "Core cells of an octree model must be axis-aligned rectangles or boxes.");
      corner |= (at_lower ? 0 : 1) << axis;
    }
    corners.push_back(corner);
  }
  std::sort(corners.begin(), corners.end());
  MFEM_VERIFY(std::adjacent_find(corners.begin(), corners.end()) == corners.end() &&
                  static_cast<int>(corners.size()) == (1 << dim),
              "Core cells of an octree model must be axis-aligned rectangles or boxes.");
}

/// Smallest size to which a cell whose top is at a depth is refined.
double min_size_at(const OctreeSettings &settings, double depth) {
  for (const auto &[limit, size] : settings.min_size) {
    if (depth < limit) {
      return size;
    }
  }
  return settings.min_size.empty() ? 0.0 : settings.min_size.back().second;
}

/// Return whether a cell on the surface overlaps the horizontal box of the survey.
bool in_surface_box(const OctreeSettings &settings, const std::vector<double> &lower, const std::vector<double> &upper) {
  const int horizontal = static_cast<int>(lower.size()) - 1;
  if (static_cast<int>(settings.surface_box.size()) != 2 * horizontal) {
    return false;
  }
  for (int axis = 0; axis < horizontal; ++axis) {
    if (upper[axis] <= settings.surface_box[2 * axis] || lower[axis] >= settings.surface_box[2 * axis + 1]) {
      return false;
    }
  }
  return true;
}

} // namespace

VoxelModel VoxelModel::read(const std::string &prefix) {
  VoxelModel voxels;
  std::ifstream header(prefix + ".txt");
  MFEM_VERIFY(header.good(), fmt::format("Cannot open the voxel header {}.txt.", prefix));
  int dim = 0;
  std::string line;
  while (std::getline(header, line)) {
    line = strip_comment(line);
    if (line.empty()) {
      continue;
    }
    std::istringstream row(line);
    std::string key;
    row >> key;
    if (key == "dimension") {
      row >> dim;
    } else if (key == "spacing") {
      row >> voxels.spacing;
    } else if (key == "origin" || key == "shape") {
      MFEM_VERIFY(dim == 2 || dim == 3, "The voxel header must give the dimension (2 or 3) first.");
      for (int axis = 0; axis < dim; ++axis) {
        double value = 0.0;
        row >> value;
        if (key == "origin") {
          voxels.origin.push_back(value);
        } else {
          voxels.shape.push_back(static_cast<int>(value));
        }
      }
    }
    MFEM_VERIFY(!row.fail(), fmt::format("Invalid voxel header row '{}'.", line));
  }
  MFEM_VERIFY(static_cast<int>(voxels.origin.size()) == dim && static_cast<int>(voxels.shape.size()) == dim &&
                  voxels.spacing > 0.0,
              fmt::format("The voxel header {}.txt needs the dimension, origin, spacing and shape.", prefix));
  std::size_t count = 1;
  for (int n : voxels.shape) {
    MFEM_VERIFY(n > 0, "Voxel counts must be positive.");
    count *= static_cast<std::size_t>(n);
  }
  std::ifstream data(prefix + ".bin", std::ios::binary);
  MFEM_VERIFY(data.good(), fmt::format("Cannot open the voxel data {}.bin.", prefix));
  voxels.log10_resistivity.resize(count);
  voxels.chargeability.resize(count);
  data.read(reinterpret_cast<char *>(voxels.log10_resistivity.data()), static_cast<std::streamsize>(count * sizeof(double)));
  data.read(reinterpret_cast<char *>(voxels.chargeability.data()), static_cast<std::streamsize>(count * sizeof(double)));
  MFEM_VERIFY(data.good() && data.peek() == std::char_traits<char>::eof(),
              fmt::format("The voxel data {}.bin must hold {} float64 values.", prefix, 2 * count));
  for (std::size_t index = 0; index < count; ++index) {
    MFEM_VERIFY(std::isfinite(voxels.log10_resistivity[index]) && voxels.chargeability[index] >= 0.0 &&
                    voxels.chargeability[index] < 1.0,
                "Voxel resistivities must be finite and chargeabilities in [0, 1).");
  }
  return voxels;
}

std::vector<OctreeCell> refine_to_voxels(mfem::Mesh &mesh, const VoxelModel &voxels,
                                         const std::vector<BackgroundLayer> &background, const OctreeSettings &settings) {
  const int dim = mesh.Dimension();
  MFEM_VERIFY(voxels.dim() == dim, "The voxel model and the mesh need the same dimension.");
  MFEM_VERIFY(!background.empty(), "The octree model needs a background for the cells outside the core.");
  mesh.EnsureNCMesh(false);
  const double surface = voxels.origin[dim - 1];
  std::vector<double> lower, upper;
  while (true) {
    mfem::Array<int> marked;
    for (int element = 0; element < mesh.GetNE(); ++element) {
      if (mesh.GetAttribute(element) != settings.core_attribute) {
        continue;
      }
      element_box(mesh, element, lower, upper, true);
      bool aligned = false;
      const auto box = voxel_box(voxels, lower, upper, aligned);
      // Children stay on voxel faces only when every edge spans an even number of voxels.
      bool divisible = aligned;
      double size = 0.0;
      for (int axis = 0; axis < dim; ++axis) {
        const int across = box[axis].second - box[axis].first;
        divisible = divisible && across >= 2 && across % 2 == 0;
        size = std::max(size, upper[axis] - lower[axis]);
      }
      if (!divisible) {
        continue;
      }
      const bool on_surface = std::abs(lower[dim - 1] - surface) < 1.0e-6 * voxels.spacing;
      bool refine = on_surface && settings.surface_size > 0.0 && size > settings.surface_size * (1.0 + 1.0e-9) &&
                    in_surface_box(settings, lower, upper);
      if (!refine && size > min_size_at(settings, lower[dim - 1] - surface) * (1.0 + 1.0e-9)) {
        const VoxelStatistics stats = statistics(voxels, box);
        refine = stats.std_log10_resistivity > settings.log10_resistivity_std ||
                 stats.std_chargeability > settings.chargeability_std;
      }
      if (refine) {
        marked.Append(element);
      }
    }
    if (marked.Size() == 0) {
      break;
    }
    mesh.GeneralRefinement(marked, 1, settings.nc_limit);
  }

  // Core leaves get their own attributes 1..n; other leaves share one attribute per background layer.
  std::vector<OctreeCell> cells(mesh.GetNE());
  int core_count = 0;
  for (int element = 0; element < mesh.GetNE(); ++element) {
    cells[element].core = mesh.GetAttribute(element) == settings.core_attribute;
    core_count += cells[element].core ? 1 : 0;
  }
  int next_core = 0;
  for (int element = 0; element < mesh.GetNE(); ++element) {
    OctreeCell &cell = cells[element];
    element_box(mesh, element, cell.lower, cell.upper);
    if (cell.core) {
      // Cells refined below the voxel size by balancing take the voxels they overlap.
      bool aligned = false;
      const VoxelStatistics stats = statistics(voxels, voxel_box(voxels, cell.lower, cell.upper, aligned));
      cell.attribute = ++next_core;
      cell.log10_resistivity = stats.mean_log10_resistivity;
      cell.chargeability = stats.mean_chargeability;
    } else {
      const double depth = 0.5 * (cell.lower[dim - 1] + cell.upper[dim - 1]);
      std::size_t layer = 0;
      while (layer + 1 < background.size() && background[layer + 1].top <= depth) {
        ++layer;
      }
      cell.attribute = core_count + 1 + static_cast<int>(layer);
      cell.log10_resistivity = -std::log10(background[layer].sigma);
      cell.chargeability = background[layer].eta;
    }
    mesh.SetAttribute(element, cell.attribute);
    if (mesh.ncmesh) {
      mesh.ncmesh->SetAttribute(element, cell.attribute);
    }
  }
  mesh.SetAttributes();
  return cells;
}

std::map<int, CellProperty> cell_properties(const std::vector<OctreeCell> &cells) {
  std::map<int, CellProperty> properties;
  for (const OctreeCell &cell : cells) {
    properties[cell.attribute] = { std::pow(10.0, cell.log10_resistivity), cell.chargeability };
  }
  return properties;
}

} // namespace dcip
