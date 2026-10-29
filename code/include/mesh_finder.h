// Point location on a mesh (GSLIB, with a neighbour search for points on element boundaries).

#pragma once

#include <mfem.hpp>

#include <memory>
#include <vector>

namespace dcip {

/// One locally owned mesh element containing a physical-space point.
struct ElementPoint {
  int elem = -1;
  mfem::IntegrationPoint ip;
};

/// Point-location queries on the local part of an MFEM mesh.
class MeshFinder {
public:
  explicit MeshFinder(const mfem::Mesh &mesh);
  ~MeshFinder();

  /// Return every locally owned element that contains a physical-space point, sorted by element; a point on a face,
  /// edge or vertex lies in all elements that share it.
  std::vector<ElementPoint> find_all_elements_around_point(const mfem::Vector &p) const;

private:
  /// Build the rank-local face adjacency, including every fine element attached to a nonconforming master face.
  void build_element_face_neighbors() const;

  const mfem::Mesh *mesh;
  int dim_;
  mutable std::unique_ptr<mfem::Table> vertex_to_element;
  mutable std::unique_ptr<mfem::Table> element_face_neighbors;
  mutable mfem::FindPointsGSLIB fp_gslib;
};

} // namespace dcip
