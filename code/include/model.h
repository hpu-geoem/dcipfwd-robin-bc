// Mesh and cell properties of a DC/IP model, and the layered background column.

#pragma once

#include "dc_boundary.h"

#include <mfem.hpp>

#include <map>
#include <string>
#include <vector>

namespace dcip {

/// Resistivity and integral chargeability of a cell attribute.
struct CellProperty {
  double resistivity;
  double chargeability = 0.0;
};

/// Ground mesh of a DC/IP model with the properties of its element attributes; z points down.
/// A 3D mesh uses (x, y, z); a 2D mesh is the (y, z) section of a 2.5D model with strike along x.
class Model {
public:
  /// Read an MFEM mesh and a table with rows "attribute resistivity [chargeability]".
  Model(const std::string &mesh_file, const std::string &property_file);

  /// Take a mesh and the properties of its attributes.
  Model(mfem::Mesh mesh, std::map<int, CellProperty> properties);

  int dim() const { return mesh_.Dimension(); }
  const mfem::Mesh &mesh() const { return mesh_; }

  double resistivity(int attribute) const { return property(attribute).resistivity; }
  double conductivity(int attribute) const { return 1.0 / resistivity(attribute); }
  double chargeability(int attribute) const { return property(attribute).chargeability; }

  /// Return whether an attribute of the mesh has a positive chargeability.
  bool has_chargeability() const;

  /// Return the layers met by the vertical line through a horizontal position ((x, y) in 3D, y in 2D), from the
  /// surface down, with equal neighbours merged.
  std::vector<BackgroundLayer> column(const mfem::Vector &horizontal) const;

  /// Return the column at the side (2D) or corner (3D) of the bounding box that differs least from the model,
  /// measured by the sum over cells of |sigma_cell - sigma_column(z_cell)|.
  std::vector<BackgroundLayer> corner_background() const;

private:
  const CellProperty &property(int attribute) const;
  void validate() const;

  mfem::Mesh mesh_;
  std::map<int, CellProperty> properties_;
};

/// Read background layers from rows "top resistivity [chargeability]" with tops increasing downwards.
std::vector<BackgroundLayer> read_background(const std::string &file);

/// Return the conductivity of a layer column at a depth (the layer above the first top for shallower points).
double column_conductivity(const std::vector<BackgroundLayer> &column, double depth);

} // namespace dcip
