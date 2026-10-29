#include "model.h"

#include "mesh_finder.h"
#include "utils.h"

#include <fmt/format.h>

#include <algorithm>
#include <cmath>
#include <fstream>
#include <limits>
#include <sstream>

namespace dcip {

namespace {

/// Return the bounding box of the vertices of a mesh.
std::pair<mfem::Vector, mfem::Vector> bounding_box(const mfem::Mesh &mesh) {
  const int dim = mesh.SpaceDimension();
  mfem::Vector lower(dim), upper(dim);
  lower = std::numeric_limits<double>::max();
  upper = std::numeric_limits<double>::lowest();
  for (int vertex = 0; vertex < mesh.GetNV(); ++vertex) {
    const double *point = mesh.GetVertex(vertex);
    for (int axis = 0; axis < dim; ++axis) {
      lower[axis] = std::min(lower[axis], point[axis]);
      upper[axis] = std::max(upper[axis], point[axis]);
    }
  }
  return { lower, upper };
}

} // namespace

Model::Model(const std::string &mesh_file, const std::string &property_file) : mesh_(mesh_file.c_str(), 1, 1) {
  std::ifstream input(property_file);
  MFEM_VERIFY(input.good(), fmt::format("Cannot open the property file {}.", property_file));
  std::string line;
  while (std::getline(input, line)) {
    line = strip_comment(line);
    if (line.empty()) {
      continue;
    }
    std::istringstream row(line);
    int attribute = 0;
    CellProperty property{ 0.0, 0.0 };
    row >> attribute >> property.resistivity;
    MFEM_VERIFY(!row.fail(), fmt::format("Invalid property row '{}' in {}.", line, property_file));
    row >> property.chargeability;
    if (row.fail()) {
      property.chargeability = 0.0;
    }
    properties_[attribute] = property;
  }
  // Nodes for point location and a nonconforming mesh for local refinement.
  mesh_.EnsureNodes();
  mesh_.EnsureNCMesh(false);
  validate();
}

Model::Model(mfem::Mesh mesh, std::map<int, CellProperty> properties) : mesh_(std::move(mesh)), properties_(std::move(properties)) {
  mesh_.EnsureNodes();
  mesh_.EnsureNCMesh(false);
  validate();
}

void Model::validate() const {
  MFEM_VERIFY(dim() == 2 || dim() == 3, "A DC model needs a two- or three-dimensional mesh.");
  for (int element = 0; element < mesh_.GetNE(); ++element) {
    const int attribute = mesh_.GetAttribute(element);
    const auto found = properties_.find(attribute);
    MFEM_VERIFY(found != properties_.end(), fmt::format("Attribute {} has no property.", attribute));
    MFEM_VERIFY(found->second.resistivity > 0.0 && std::isfinite(found->second.resistivity),
                fmt::format("Attribute {} needs a positive finite resistivity.", attribute));
    MFEM_VERIFY(found->second.chargeability >= 0.0 && found->second.chargeability < 1.0,
                fmt::format("Attribute {} needs a chargeability in [0, 1).", attribute));
  }
}

const CellProperty &Model::property(int attribute) const {
  const auto found = properties_.find(attribute);
  MFEM_VERIFY(found != properties_.end(), fmt::format("Attribute {} has no property.", attribute));
  return found->second;
}

bool Model::has_chargeability() const {
  for (int element = 0; element < mesh_.GetNE(); ++element) {
    const int attribute = mesh_.GetAttribute(element);
    if (chargeability(attribute) > 0.0) {
      return true;
    }
  }
  return false;
}

std::vector<BackgroundLayer> Model::column(const mfem::Vector &horizontal) const {
  const int depth_axis = dim() - 1;
  MFEM_VERIFY(horizontal.Size() == depth_axis, "A column position has one coordinate fewer than the mesh.");
  const auto [lower, upper] = bounding_box(mesh_);
  double extent = 0.0;
  for (int axis = 0; axis < dim(); ++axis) {
    extent = std::max(extent, upper[axis] - lower[axis]);
  }
  const double tolerance = 1.0e-12 * extent;

  // Walk down the vertical line just inside the box; bisection finds where it leaves each element.
  mfem::Vector sample(dim());
  for (int axis = 0; axis < depth_axis; ++axis) {
    const double center = 0.5 * (lower[axis] + upper[axis]);
    sample[axis] = horizontal[axis] + 1.0e-7 * extent * (center > horizontal[axis] ? 1.0 : -1.0);
  }
  const MeshFinder finder(mesh_);
  // Element containing the sample at depth z, or -1 outside the mesh; the located point must map back to the sample.
  mfem::Vector mapped(dim());
  const auto element_at = [&](double z) {
    sample[depth_axis] = z;
    for (const ElementPoint &location : finder.find_all_elements_around_point(sample)) {
      mfem::ElementTransformation *transformation = const_cast<mfem::Mesh &>(mesh_).GetElementTransformation(location.elem);
      transformation->Transform(location.ip, mapped);
      if (mapped.DistanceTo(sample) <= 1.0e-10 * extent) {
        return location.elem;
      }
    }
    return -1;
  };
  const double bottom = upper[depth_axis];
  MFEM_VERIFY(element_at(bottom - tolerance) >= 0, "A background column does not reach the bottom of the mesh.");
  // Under topography the line may enter the mesh below the top of the bounding box.
  double top = lower[depth_axis];
  if (element_at(top + tolerance) < 0) {
    double outside = top, inside = bottom - tolerance;
    while (inside - outside > tolerance) {
      const double middle = 0.5 * (inside + outside);
      (element_at(middle) < 0 ? outside : inside) = middle;
    }
    top = inside;
  }
  std::vector<BackgroundLayer> layers;
  double reported_top = top;
  int element = element_at(top + tolerance);
  while (true) {
    double inside = top + tolerance, outside = bottom - tolerance;
    if (element_at(outside) == element) {
      inside = outside;
    } else {
      while (outside - inside > tolerance) {
        const double middle = 0.5 * (inside + outside);
        (element_at(middle) == element ? inside : outside) = middle;
      }
    }
    const int attribute = mesh_.GetAttribute(element);
    const double sigma = conductivity(attribute), eta = chargeability(attribute);
    if (layers.empty() || layers.back().sigma != sigma || layers.back().eta != eta) {
      layers.push_back({ reported_top, sigma, eta });
    }
    if (inside >= bottom - 2.0 * tolerance) {
      break;
    }
    // The walk continues from the bisected exit; the reported depth takes a vertex depth within 1e-8 of the extent.
    reported_top = outside;
    mfem::Array<int> vertices;
    mesh_.GetElementVertices(element, vertices);
    for (int vertex : vertices) {
      const double depth = mesh_.GetVertex(vertex)[depth_axis];
      if (std::abs(depth - outside) <= 1.0e-8 * extent) {
        reported_top = depth;
      }
    }
    const int next = element_at(outside);
    if (next < 0) {
      break;
    }
    MFEM_VERIFY(next != element && outside > top, "The background column walk does not advance.");
    top = outside;
    element = next;
  }
  MFEM_VERIFY(!layers.empty(), "A background column has no ground.");
  return layers;
}

double column_conductivity(const std::vector<BackgroundLayer> &column, double depth) {
  MFEM_VERIFY(!column.empty(), "An empty column has no conductivity.");
  double sigma = column.front().sigma;
  for (const BackgroundLayer &layer : column) {
    if (depth >= layer.top) {
      sigma = layer.sigma;
    }
  }
  return sigma;
}

std::vector<BackgroundLayer> Model::corner_background() const {
  const auto [lower, upper] = bounding_box(mesh_);
  std::vector<mfem::Vector> corners;
  if (dim() == 2) {
    corners = { mfem::Vector({ lower[0] }), mfem::Vector({ upper[0] }) };
  } else {
    corners = { mfem::Vector({ lower[0], lower[1] }), mfem::Vector({ upper[0], lower[1] }), mfem::Vector({ upper[0], upper[1] }),
                mfem::Vector({ lower[0], upper[1] }) };
  }

  std::vector<BackgroundLayer> best;
  double best_difference = std::numeric_limits<double>::infinity();
  mfem::Vector center(dim());
  for (const mfem::Vector &corner : corners) {
    const std::vector<BackgroundLayer> layers = column(corner);
    double difference = 0.0;
    for (int element = 0; element < mesh_.GetNE(); ++element) {
      const int attribute = mesh_.GetAttribute(element);
      const_cast<mfem::Mesh &>(mesh_).GetElementTransformation(element)->Transform(
          mfem::Geometries.GetCenter(mesh_.GetElementGeometry(element)), center);
      difference += std::abs(conductivity(attribute) - column_conductivity(layers, center[dim() - 1]));
    }
    if (difference < best_difference) {
      best_difference = difference;
      best = layers;
    }
  }
  return best;
}

std::vector<BackgroundLayer> read_background(const std::string &file) {
  std::ifstream input(file);
  MFEM_VERIFY(input.good(), fmt::format("Cannot open the background file {}.", file));
  std::vector<BackgroundLayer> layers;
  std::string line;
  while (std::getline(input, line)) {
    line = strip_comment(line);
    if (line.empty()) {
      continue;
    }
    std::istringstream row(line);
    double top = 0.0, resistivity = 0.0, eta = 0.0;
    row >> top >> resistivity;
    MFEM_VERIFY(!row.fail() && resistivity > 0.0, fmt::format("Invalid background row '{}' in {}.", line, file));
    row >> eta;
    if (row.fail()) {
      eta = 0.0;
    }
    MFEM_VERIFY(layers.empty() || top > layers.back().top, "Background layer tops must increase with depth.");
    layers.push_back({ top, 1.0 / resistivity, eta });
  }
  MFEM_VERIFY(!layers.empty(), fmt::format("The background file {} has no layer.", file));
  return layers;
}

} // namespace dcip
