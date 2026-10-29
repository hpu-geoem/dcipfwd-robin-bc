// Octree (quadtree in 2D) meshes that follow a voxel model of resistivity and chargeability.

#pragma once

#include "dc_boundary.h"
#include "model.h"

#include <mfem.hpp>

#include <map>
#include <string>
#include <utility>
#include <vector>

namespace dcip {

/// Regular voxel grid of log10 resistivity and chargeability; axis 0 varies fastest, the last axis is depth (down).
struct VoxelModel {
  std::vector<double> origin; // Lower corner per axis
  double spacing = 0.0;       // Voxel edge length
  std::vector<int> shape;     // Voxels per axis
  std::vector<double> log10_resistivity;
  std::vector<double> chargeability;

  int dim() const { return static_cast<int>(shape.size()); }

  /// Read prefix.txt (lines "dimension d", "origin ...", "spacing h", "shape ...") and prefix.bin (log10
  /// resistivity, then chargeability, as little-endian float64).
  static VoxelModel read(const std::string &prefix);
};

/// Settings of an octree (quadtree in 2D) that follows a voxel model.
struct OctreeSettings {
  int core_attribute = 1;               // Root cells with this attribute take the voxel model; others the background
  double log10_resistivity_std = 0.1;   // Refine a cell whose voxels vary more than this (standard deviation)
  double chargeability_std = 0.02;
  std::vector<std::pair<double, double>> min_size{ { 1.0e30, 0.0 } }; // (depth, size): cells whose top is above depth
                                                                       // are refined down to size; the first match applies
  std::vector<double> surface_box;      // Horizontal box (min, max per horizontal axis) of the survey
  double surface_size = 0.0;            // Size of surface cells inside the box (0: no surface refinement)
  int nc_limit = 1;                     // Largest refinement-level difference of neighbours
};

/// Leaf cell of an octree model: bounding box (min, max per axis) and properties.
struct OctreeCell {
  std::vector<double> lower, upper;
  int attribute;
  double log10_resistivity, chargeability;
  bool core;
};

/// Refine the core cells of a conforming root mesh where the voxel model varies, and give every leaf an attribute:
/// core leaves take the log-mean resistivity and mean chargeability of the voxels they overlap, the other leaves the
/// background layer at their centre. Core cells must be axis-aligned boxes; a cell is split only while its faces lie on
/// voxel faces and every edge spans an even number of voxels. Returns the leaf cells in element order.
std::vector<OctreeCell> refine_to_voxels(mfem::Mesh &mesh, const VoxelModel &voxels,
                                         const std::vector<BackgroundLayer> &background, const OctreeSettings &settings);

/// Properties of the attributes of the leaf cells.
std::map<int, CellProperty> cell_properties(const std::vector<OctreeCell> &cells);

} // namespace dcip
