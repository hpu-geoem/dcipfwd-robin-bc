#include "mesh_finder.h"

#include <algorithm>
#include <queue>

namespace dcip {

MeshFinder::MeshFinder(const mfem::Mesh &m) : mesh(&m), dim_(m.SpaceDimension()), fp_gslib(MPI_COMM_SELF) {
  MFEM_VERIFY(dim_ == 2 || dim_ == 3, "MeshFinder requires a two- or three-dimensional mesh.");
  mfem::Mesh &search_mesh = const_cast<mfem::Mesh &>(*mesh);
  fp_gslib.Setup(search_mesh);
}

MeshFinder::~MeshFinder() { fp_gslib.FreeData(); }

void MeshFinder::build_element_face_neighbors() const {
  mfem::Array<mfem::Connection> connections;

  for (int face = 0; face < mesh->GetNumFaces(); ++face) {
    const mfem::Mesh::FaceInformation info = mesh->GetFaceInformation(face);
    if (info.element[0].location != mfem::Mesh::ElementLocation::Local ||
        info.element[1].location != mfem::Mesh::ElementLocation::Local) {
      continue;
    }

    const int elem1 = info.element[0].index;
    const int elem2 = info.element[1].index;
    if (elem1 < 0 || elem1 >= mesh->GetNE() || elem2 < 0 || elem2 >= mesh->GetNE() || elem1 == elem2) {
      continue;
    }

    connections.Append({ elem1, elem2 });
    connections.Append({ elem2, elem1 });
  }

  connections.Sort();
  connections.Unique();

  element_face_neighbors = std::make_unique<mfem::Table>(mesh->GetNE(), connections);
}

std::vector<ElementPoint> MeshFinder::find_all_elements_around_point(const mfem::Vector &p) const {
  MFEM_VERIFY(p.Size() == dim_, "Point dimension must match the mesh space dimension.");

  std::vector<ElementPoint> locations;
  if (mesh->GetNE() == 0) {
    return locations;
  }

  mfem::Vector point(p);
  fp_gslib.FindPoints(point);

  const unsigned int code = fp_gslib.GetCode()[0];
  if (code == 2) {
    return locations;
  }

  if (code == 0) {
    mfem::IntegrationPoint ip;
    const mfem::Vector &positions = fp_gslib.GetReferencePosition();
    ip.Init(0);
    ip.Set(positions.HostRead(), dim_);
    ip.weight = 1.0;
    locations.push_back({ static_cast<int>(fp_gslib.GetElem()[0]), ip });
    return locations;
  }

  mfem::InverseElementTransformation inverse;
  inverse.SetInitialGuessType(mfem::InverseElementTransformation::ClosestRefNode);
  inverse.SetInitGuessRelOrder(1);

  mfem::Mesh &search_mesh = const_cast<mfem::Mesh &>(*mesh);
  if (!vertex_to_element) {
    vertex_to_element.reset(search_mesh.GetVertexToElementTable());
  }
  if (!element_face_neighbors) {
    build_element_face_neighbors();
  }

  std::queue<int> pending;
  std::vector<char> visited(search_mesh.GetNE(), false);

  auto enqueue = [&](int elem) {
    if (elem >= 0 && elem < search_mesh.GetNE() && !visited[elem]) {
      visited[elem] = true;
      pending.push(elem);
    }
  };

  auto enqueue_neighbors = [&](int elem) {
    mfem::Array<int> vertices;
    search_mesh.GetElementVertices(elem, vertices);
    for (int vertex : vertices) {
      mfem::Array<int> incident_elements;
      vertex_to_element->GetRow(vertex, incident_elements);
      for (int incident_elem : incident_elements) {
        enqueue(incident_elem);
      }
    }

    mfem::Array<int> face_neighbors;
    element_face_neighbors->GetRow(elem, face_neighbors);
    for (int neighbor : face_neighbors) {
      enqueue(neighbor);
    }
  };

  const int seed_elem = static_cast<int>(fp_gslib.GetElem()[0]);
  const mfem::Vector &seed_position = fp_gslib.GetReferencePosition();
  mfem::IntegrationPoint seed_ip;
  seed_ip.Init(0);
  seed_ip.Set(seed_position.HostRead(), dim_);
  seed_ip.weight = 1.0;
  locations.push_back({ seed_elem, seed_ip });
  visited[seed_elem] = true;
  enqueue_neighbors(seed_elem);

  while (!pending.empty()) {
    const int elem = pending.front();
    pending.pop();

    mfem::ElementTransformation *transformation = search_mesh.GetElementTransformation(elem);
    inverse.SetTransformation(*transformation);
    mfem::IntegrationPoint ip;
    if (inverse.Transform(point, ip) == mfem::InverseElementTransformation::Inside) {
      locations.push_back({ elem, ip });
      enqueue_neighbors(elem);
    }
  }

  std::sort(locations.begin(), locations.end(),
            [](const ElementPoint &left, const ElementPoint &right) { return left.elem < right.elem; });

  return locations;
}

} // namespace dcip
