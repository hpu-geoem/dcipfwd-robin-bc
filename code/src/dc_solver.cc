#include "dc_solver.h"

#include "dc_boundary.h"
#include "integral_transforms.h"
#include "log.h"
#include "mesh_finder.h"
#include "model.h"
#include "utils.h"

#include <fmt/format.h>

#include <algorithm>
#include <cmath>
#include <limits>
#include <map>

namespace dcip {

namespace {

/// Boundary attributes of the ground mesh.
enum GroundBoundary { EarthSurface = 1, Truncation = 2 };

// Vertical component of the unit outward normal below which a boundary face points up; z is positive downwards.
constexpr double kUpwardNormal = -1.0e-6;

/// Return the CalcOrtho normal at the centre of a boundary element.
mfem::Vector center_normal(mfem::ParMesh &mesh, int boundary_element) {
  mfem::ElementTransformation *transformation = mesh.GetBdrElementTransformation(boundary_element);
  transformation->SetIntPoint(&mfem::Geometries.GetCenter(mesh.GetBdrElementGeometry(boundary_element)));
  mfem::Vector normal(mesh.SpaceDimension());
  mfem::CalcOrtho(transformation->Jacobian(), normal);
  return normal;
}

/// Return the sign that turns the CalcOrtho normal of a boundary element outwards.
double outward_orientation(mfem::ParMesh &mesh, int boundary_element) {
  const mfem::IntegrationPoint &center = mfem::Geometries.GetCenter(mesh.GetBdrElementGeometry(boundary_element));
  mfem::Vector face_center;
  mesh.GetBdrElementTransformation(boundary_element)->Transform(center, face_center);
  int element = -1, info = -1;
  mesh.GetBdrElementAdjacentElement(boundary_element, element, info);
  mfem::Vector element_center;
  mesh.GetElementTransformation(element)->Transform(mfem::Geometries.GetCenter(mesh.GetElementGeometry(element)), element_center);
  face_center -= element_center;
  return face_center * center_normal(mesh, boundary_element) < 0.0 ? -1.0 : 1.0;
}

/// Tag the boundary of the ground mesh: EarthSurface where the outward normal points up, Truncation elsewhere.
void tag_ground_boundaries(mfem::ParMesh &mesh) {
  const int depth_axis = mesh.SpaceDimension() - 1;
  for (int boundary_element = 0; boundary_element < mesh.GetNBE(); ++boundary_element) {
    mfem::Vector normal = center_normal(mesh, boundary_element);
    const double upward = outward_orientation(mesh, boundary_element) * normal[depth_axis] / normal.Norml2();
    mesh.SetBdrAttribute(boundary_element, upward < kUpwardNormal ? EarthSurface : Truncation);
  }
  mesh.SetAttributes(false, true);
}

/// Return the marker of the truncation boundary.
mfem::Array<int> truncation_marker(const mfem::ParMesh &mesh) {
  mfem::Array<int> marker(mesh.bdr_attributes.Size() > 0 ? mesh.bdr_attributes.Max() : 0);
  marker = 0;
  if (marker.Size() >= Truncation) {
    marker[Truncation - 1] = 1;
  }
  return marker;
}

/// Return -dG/dn / G, not clipped, of a source on a homogeneous half-space (Dey and Morrison): cos(theta) / L in 3D
/// and k K1(k L) cos(theta) / K0(k L) in 2.5D.
double homogeneous_ratio(const mfem::Vector &position, const mfem::Vector &unit_normal, const mfem::Vector &source, double k) {
  mfem::Vector displacement(position);
  displacement -= source;
  const double distance = displacement.Norml2();
  const double ratio = (displacement * unit_normal) / (distance * distance);
  return k == 0.0 ? ratio : ratio * distance * k * bessel_k1_over_k0(k * distance);
}

/// Offset r >= 0 of a boundary point from a source along the ground and the component of the normal along it.
double ground_offset(int dim, const mfem::Vector &position, const mfem::Vector &normal, const mfem::Vector &source,
                     double &normal_r) {
  if (dim == 3) {
    const double dx = position[0] - source[0], dy = position[1] - source[1];
    const double r = std::hypot(dx, dy);
    normal_r = r > 0.0 ? (dx * normal[0] + dy * normal[1]) / r : 0.0;
    return r;
  }
  const double dy = position[0] - source[0];
  normal_r = dy == 0.0 ? 0.0 : (dy < 0.0 ? -normal[0] : normal[0]);
  return std::abs(dy);
}

/// Return the elements of a mesh that contain any of the points, given as packed coordinates.
mfem::Array<int> find_elements_around_points(mfem::Mesh &mesh, const mfem::Vector &points) {
  const int dim = mesh.SpaceDimension();
  MeshFinder finder(mesh);
  mfem::Array<int> elements;
  for (int point_index = 0; point_index < points.Size() / dim; ++point_index) {
    mfem::Vector point(dim);
    for (int axis = 0; axis < dim; ++axis) {
      point[axis] = points[point_index * dim + axis];
    }
    for (const ElementPoint &location : finder.find_all_elements_around_point(point)) {
      elements.Append(location.elem);
    }
  }
  elements.Sort();
  elements.Unique();
  return elements;
}

/// Refine the elements of a parallel mesh around points once and keep their attributes.
void refine_elements_around_points(mfem::ParMesh &mesh, const mfem::Vector &points) {
  mfem::DG_FECollection fec(0, mesh.Dimension());
  mfem::ParFiniteElementSpace fes(&mesh, &fec);
  mfem::ParGridFunction attributes(&fes);
  for (int element = 0; element < mesh.GetNE(); ++element) {
    attributes[element] = mesh.GetAttribute(element);
  }
  mesh.GeneralRefinement(find_elements_around_points(mesh, points), -1, 1);
  fes.Update();
  attributes.Update();
  if (!mesh.Conforming()) {
    mesh.Rebalance();
    fes.Update();
    attributes.Update();
  }
  for (int element = 0; element < mesh.GetNE(); ++element) {
    mesh.SetAttribute(element, static_cast<int>(std::lround(attributes[element])));
  }
  mesh.SetAttributes();
}

/// DC or instantaneous conductivity of the attribute of an element.
class AttributeConductivityCoefficient : public mfem::Coefficient {
public:
  AttributeConductivityCoefficient(const Model &model, bool chargeable) : model_(model), chargeable_(chargeable) {}

  double Eval(mfem::ElementTransformation &transformation, const mfem::IntegrationPoint &) override {
    const double sigma = model_.conductivity(transformation.Attribute);
    return chargeable_ ? chargeable_conductivity(sigma, model_.chargeability(transformation.Attribute)) : sigma;
  }

private:
  const Model &model_;
  bool chargeable_;
};

/// Boundary term sigma (alpha_e - alpha_0) of one electrode and wavenumber on true DOFs, P^T S P, with S the boundary
/// mass matrix on the local DOFs of the truncation faces.
class BoundaryCorrectionOperator : public mfem::Operator {
public:
  BoundaryCorrectionOperator(const mfem::Operator &prolongation, const mfem::Array<int> &dofs,
                             std::unique_ptr<mfem::SparseMatrix> matrix)
      : mfem::Operator(prolongation.Width()), prolongation_(prolongation), dofs_(dofs), matrix_(std::move(matrix)),
        local_x_(prolongation.Height()), local_y_(prolongation.Height()), boundary_x_(dofs.Size()), boundary_y_(dofs.Size()) {}

  void Mult(const mfem::Vector &x, mfem::Vector &y) const override {
    prolongation_.Mult(x, local_x_);
    local_x_.GetSubVector(dofs_, boundary_x_);
    matrix_->Mult(boundary_x_, boundary_y_);
    local_y_ = 0.0;
    local_y_.SetSubVector(dofs_, boundary_y_);
    prolongation_.MultTranspose(local_y_, y);
  }

private:
  const mfem::Operator &prolongation_;
  const mfem::Array<int> &dofs_;
  std::unique_ptr<mfem::SparseMatrix> matrix_;
  mutable mfem::Vector local_x_, local_y_, boundary_x_, boundary_y_;
};

} // namespace

std::unique_ptr<mfem::HypreBoomerAMG> make_dc_amg(mfem::HypreParMatrix &matrix, double strength_threshold) {
  auto amg = std::make_unique<mfem::HypreBoomerAMG>(matrix);
  amg->SetStrengthThresh(strength_threshold);
  amg->SetPrintLevel(0);
  mfem::Vector b(matrix.Height()), x(matrix.Height());
  b = 0.0;
  x = 0.0;
  amg->Setup(b, x);
  return amg;
}

/// Unit current of one electrode: position in model coordinates, source vector, potential, and per truncation face
/// either its own coefficient alpha_e (per-electrode reference) or the Robin data sigma (dG_e/dn + alpha_0 G_e)
/// (boundary data), wavenumber-major.
struct DCSolver::ElectrodeSystem {
  mfem::Vector position;
  mfem::Vector rhs;
  std::unique_ptr<mfem::ParGridFunction> potential;
  std::vector<std::vector<double>> boundary;
  bool boundary_ready = false; // Whether boundary holds this rank's faces, which may be none
};

/// Quadrature points of the truncation faces of a mesh: boundary element, face rule, points and unit outward normals
/// (dim x point), the conductivity of the cell behind the face, and optionally the local DOFs, shape values
/// (DOF x point) and quadrature weights times |J|.
struct DCSolver::TruncationFaces {
  std::vector<int> face_of;  // Face index of every boundary element, -1 off the truncation boundary
  std::vector<const mfem::IntegrationRule *> rules;
  std::vector<mfem::DenseMatrix> positions, normals;
  std::vector<double> sigma;
  std::vector<mfem::Array<int>> vdofs;
  std::vector<mfem::DenseMatrix> shapes;
  std::vector<mfem::Vector> weights;
  mfem::Array<int> dofs;                // Local DOFs of all truncation faces
  std::vector<mfem::Array<int>> packed; // Face DOFs as indices into dofs
  int face_dim = 0;

  /// Return the index of an integration point in the rule of a face; only face coordinates are compared, since MFEM
  /// leaves the others uninitialized.
  int point_index(int face, const mfem::IntegrationPoint &point) const {
    const mfem::IntegrationRule &rule = *rules[face];
    for (int index = 0; index < rule.GetNPoints(); ++index) {
      const mfem::IntegrationPoint &candidate = rule.IntPoint(index);
      if (&candidate == &point || (candidate.x == point.x && (face_dim < 2 || candidate.y == point.y))) {
        return index;
      }
    }
    MFEM_ABORT("A boundary quadrature point is missing from the tabulated face rule.");
  }
};

/// Matrix, preconditioner or factorization of one wavenumber.
struct DCSolver::WavenumberSystem {
  double k, weight;
  std::unique_ptr<mfem::HypreParMatrix> matrix;
  std::unique_ptr<mfem::HypreParMatrix> lor_matrix;
  std::unique_ptr<mfem::Solver> solver;
};

namespace {

/// Conductivity times the tabulated Robin coefficient of one wavenumber on the truncation faces.
template <typename Faces> class RobinCoefficient : public mfem::Coefficient {
public:
  RobinCoefficient(const Faces &faces, const std::vector<std::vector<double>> &alpha, int spectral)
      : faces_(faces), alpha_(alpha), spectral_(spectral) {}

  double Eval(mfem::ElementTransformation &transformation, const mfem::IntegrationPoint &point) override {
    const int face = faces_.face_of[transformation.ElementNo];
    MFEM_VERIFY(face >= 0, "The boundary element is not on the truncation boundary.");
    const int n_points = faces_.rules[face]->GetNPoints();
    return faces_.sigma[face] * alpha_[face][spectral_ * n_points + faces_.point_index(face, point)];
  }

private:
  const Faces &faces_;
  const std::vector<std::vector<double>> &alpha_;
  int spectral_;
};

} // namespace

DCSolver::DCSolver(MPI_Comm comm, const Model &model, const DCSolverOptions &options, std::shared_ptr<const DCBoundary> boundary,
                   bool chargeable)
    : comm_(comm), model_(model), options_(options), chargeable_(chargeable), boundary_(std::move(boundary)) {
  MFEM_VERIFY(boundary_, "The DC solver requires its boundary condition data.");
  MFEM_VERIFY(model_.dim() == 2 || model_.dim() == 3, "The DC solver requires a two- or three-dimensional model.");
  mesh_ = std::make_unique<mfem::ParMesh>(comm_, const_cast<mfem::Mesh &>(model_.mesh()));
  mesh_->EnsureNodes();
}

DCSolver::~DCSolver() = default;

double DCSolver::max_time(double seconds) const {
  MPI_Allreduce(MPI_IN_PLACE, &seconds, 1, MPI_DOUBLE, MPI_MAX, comm_);
  return seconds;
}

DCSolver::ElectrodeSystem &DCSolver::system(int electrode) {
  const auto found = electrodes_.find(electrode);
  MFEM_VERIFY(found != electrodes_.end(), fmt::format("Electrode {} has not been added.", electrode));
  return *found->second;
}

const DCSolver::ElectrodeSystem &DCSolver::system(int electrode) const { return const_cast<DCSolver *>(this)->system(electrode); }

double DCSolver::conductivity(int attribute) const {
  const double sigma = model_.conductivity(attribute);
  return chargeable_ ? chargeable_conductivity(sigma, model_.chargeability(attribute)) : sigma;
}

void DCSolver::refine_mesh() {
  MFEM_VERIFY(!fe_system_setup_, "Refine the mesh before setting up the DC system.");
  const double start = MPI_Wtime();
  for (int i = 0; i < options_.uniform_refinements; ++i) {
    mesh_->UniformRefinement();
  }
  if (options_.uniform_refinements > 0) {
    mesh_->Finalize(true);
  }
  // Pass i refines around the points that ask for more than i refinements.
  int passes = 0;
  for (const auto &entry : points_to_refine_) {
    passes = std::max(passes, entry.second);
  }
  const int dim = model_.dim();
  for (int pass = 0; pass < passes; ++pass) {
    std::vector<mfem::Vector> points;
    for (const auto &[candidates, count] : points_to_refine_) {
      if (count > pass) {
        points.insert(points.end(), candidates.begin(), candidates.end());
      }
    }
    if (points.empty()) {
      continue;
    }
    mfem::Vector packed(static_cast<int>(points.size()) * dim);
    for (std::size_t index = 0; index < points.size(); ++index) {
      const mfem::Vector point = project_xyz_to_model(points[index], dim);
      for (int axis = 0; axis < dim; ++axis) {
        packed(static_cast<int>(index) * dim + axis) = point[axis];
      }
    }
    refine_elements_around_points(*mesh_, packed);
  }
  stats_.refine_time += max_time(MPI_Wtime() - start);
}

void DCSolver::setup_system() {
  setup_fe_system();
  setup_linear_system();
  setup_linear_solver();
}

DCSolver::TruncationFaces DCSolver::truncation_faces(mfem::ParMesh &mesh, const mfem::ParFiniteElementSpace &fespace,
                                                    const mfem::IntegrationRule *face_rule, bool with_shapes) const {
  TruncationFaces faces;
  faces.face_of.assign(mesh.GetNBE(), -1);
  faces.face_dim = mesh.Dimension() - 1;
  const int dim = mesh.SpaceDimension();
  mfem::Vector position, normal(dim), shape;
  for (int boundary_element = 0; boundary_element < mesh.GetNBE(); ++boundary_element) {
    if (mesh.GetBdrAttribute(boundary_element) != Truncation) {
      continue;
    }
    // The orientation and the neighbour cell first, since they reuse the boundary transformation.
    const double orientation = outward_orientation(mesh, boundary_element);
    int element = -1, info = -1;
    mesh.GetBdrElementAdjacentElement(boundary_element, element, info);
    const mfem::FiniteElement &face_fe = *fespace.GetBE(boundary_element);
    mfem::ElementTransformation &transformation = *mesh.GetBdrElementTransformation(boundary_element);
    const mfem::IntegrationRule &rule = face_rule ? *face_rule : mfem::MassIntegrator::GetRule(face_fe, face_fe, transformation);
    const int n_points = rule.GetNPoints();
    faces.face_of[boundary_element] = static_cast<int>(faces.rules.size());
    faces.rules.push_back(&rule);
    faces.sigma.push_back(conductivity(mesh.GetAttribute(element)));
    mfem::DenseMatrix &positions = faces.positions.emplace_back(dim, n_points);
    mfem::DenseMatrix &normals = faces.normals.emplace_back(dim, n_points);
    if (with_shapes) {
      fespace.GetBdrElementVDofs(boundary_element, faces.vdofs.emplace_back());
      faces.shapes.emplace_back(face_fe.GetDof(), n_points);
      faces.weights.emplace_back(n_points);
      shape.SetSize(face_fe.GetDof());
    }
    for (int point = 0; point < n_points; ++point) {
      const mfem::IntegrationPoint &ip = rule.IntPoint(point);
      transformation.SetIntPoint(&ip);
      transformation.Transform(ip, position);
      mfem::CalcOrtho(transformation.Jacobian(), normal);
      normal *= orientation / normal.Norml2();
      positions.SetCol(point, position);
      normals.SetCol(point, normal);
      if (with_shapes) {
        face_fe.CalcPhysShape(transformation, shape);
        faces.shapes.back().SetCol(point, shape);
        faces.weights.back()[point] = ip.weight * transformation.Weight();
      }
    }
  }
  if (with_shapes) {
    std::map<int, int> index;
    for (const mfem::Array<int> &face_dofs : faces.vdofs) {
      for (int dof : face_dofs) {
        index.emplace(dof, 0);
      }
    }
    for (auto &[dof, position] : index) {
      position = faces.dofs.Size();
      faces.dofs.Append(dof);
    }
    for (const mfem::Array<int> &face_dofs : faces.vdofs) {
      mfem::Array<int> &packed = faces.packed.emplace_back(face_dofs.Size());
      for (int i = 0; i < face_dofs.Size(); ++i) {
        packed[i] = index.at(face_dofs[i]);
      }
    }
  }
  return faces;
}

double DCSolver::shared_ratio(const mfem::Vector &position, const mfem::Vector &normal, double k) const {
  if (options_.boundary_condition == DCBoundaryCondition::DeyMorrison) {
    return homogeneous_ratio(position, normal, boundary_->reference(), k);
  }
  return boundary_->robin_ratio(position, normal, k, coefficient_state());
}

std::vector<std::vector<double>> DCSolver::shared_coefficients(const TruncationFaces &faces, long long *clipped) const {
  const int spectra = static_cast<int>(wavenumber_systems_.size());
  std::vector<std::vector<double>> alpha(faces.rules.size());
  mfem::Vector position, normal;
  for (std::size_t face = 0; face < faces.rules.size(); ++face) {
    const int n_points = faces.rules[face]->GetNPoints();
    alpha[face].resize(spectra * n_points);
    for (int point = 0; point < n_points; ++point) {
      faces.positions[face].GetColumn(point, position);
      faces.normals[face].GetColumn(point, normal);
      for (int spectral = 0; spectral < spectra; ++spectral) {
        const double ratio = shared_ratio(position, normal, wavenumber_systems_[spectral]->k);
        if (clipped && ratio < 0.0) {
          ++*clipped;
        }
        alpha[face][spectral * n_points + point] = std::max(0.0, ratio);
      }
    }
  }
  return alpha;
}

void DCSolver::setup_fe_system() {
  if (fe_system_setup_) {
    return;
  }
  const double start = MPI_Wtime();
  tag_ground_boundaries(*mesh_);

  fec_ = std::make_unique<mfem::H1_FECollection>(options_.order, mesh_->Dimension());
  const std::vector<WavenumberNode> nodes =
      model_.dim() == 3              ? std::vector<WavenumberNode>{ { 0.0, 1.0 } }
      : !options_.wavenumbers.empty() ? options_.wavenumbers
                                      : make_log_wavenumber_quadrature(options_.kx_min, options_.kx_max, options_.kx_points_per_decade);
  for (const auto &node : nodes) {
    auto system = std::make_unique<WavenumberSystem>();
    system->k = node.k;
    system->weight = model_.dim() == 3 ? 1.0 : node.weight / PI;
    wavenumber_systems_.push_back(std::move(system));
  }
  base_fespace_ = std::make_unique<mfem::ParFiniteElementSpace>(mesh_.get(), fec_.get());
  fespace_ = base_fespace_.get();
  if (model_.dim() == 2) {
    spectral_fespace_ = std::make_unique<mfem::ParFiniteElementSpace>(mesh_.get(), fec_.get(), static_cast<int>(nodes.size()),
                                                                      mfem::Ordering::byVDIM);
    fespace_ = spectral_fespace_.get();
  }
  finder_ = std::make_unique<MeshFinder>(*mesh_);
  sigma_ = std::make_shared<AttributeConductivityCoefficient>(model_, chargeable_);
  truncation_marker_ = truncation_marker(*mesh_);
  faces_ = std::make_unique<TruncationFaces>(truncation_faces(*mesh_, *base_fespace_, nullptr, true));

  const double boundary_start = MPI_Wtime();
  long long clipped = 0, points = 0;
  alpha_ = shared_coefficients(*faces_, &clipped);
  for (const auto &values : alpha_) {
    points += static_cast<long long>(values.size());
  }
  stats_.boundary_time += max_time(MPI_Wtime() - boundary_start);

  long long n_faces = static_cast<long long>(faces_->rules.size());
  MPI_Allreduce(MPI_IN_PLACE, &n_faces, 1, MPI_LONG_LONG, MPI_SUM, comm_);
  MPI_Allreduce(MPI_IN_PLACE, &clipped, 1, MPI_LONG_LONG, MPI_SUM, comm_);
  MPI_Allreduce(MPI_IN_PLACE, &points, 1, MPI_LONG_LONG, MPI_SUM, comm_);
  stats_.cells = mesh_->GetGlobalNE();
  stats_.dofs = base_fespace_->GlobalTrueVSize();
  stats_.truncation_faces = n_faces;
  stats_.wavenumbers = static_cast<int>(nodes.size());
  stats_.boundary_points = points;
  stats_.clipped_points = clipped;
  log(1, "  DC{}: cells {}, wavenumbers {}, DoFs per wavenumber {}", chargeable_ ? " (chargeable)" : "", stats_.cells,
      stats_.wavenumbers, stats_.dofs);
  fe_system_setup_ = true;
  stats_.fe_time += max_time(MPI_Wtime() - start);
}

void DCSolver::setup_linear_system() {
  if (linear_system_setup_) {
    return;
  }
  MFEM_VERIFY(fe_system_setup_, "Set up the finite-element space before the DC linear system.");
  const double start = MPI_Wtime();
  {
    mfem::ParBilinearForm stiffness(base_fespace_.get());
    stiffness.AddDomainIntegrator(new mfem::DiffusionIntegrator(*sigma_));
    stiffness.Assemble();
    stiffness.Finalize();
    stiffness_.reset(stiffness.ParallelAssemble());
  }
  if (model_.dim() == 2) {
    mfem::ParBilinearForm mass(base_fespace_.get());
    mass.AddDomainIntegrator(new mfem::MassIntegrator(*sigma_));
    mass.Assemble();
    mass.Finalize();
    mass_.reset(mass.ParallelAssemble());
  }
  // A_0 = K + k^2 M + B_0 for every wavenumber.
  for (int spectral = 0; spectral < static_cast<int>(wavenumber_systems_.size()); ++spectral) {
    WavenumberSystem &wavenumber_system = *wavenumber_systems_[spectral];
    RobinCoefficient<TruncationFaces> coefficient(*faces_, alpha_, spectral);
    mfem::ParBilinearForm boundary(base_fespace_.get());
    boundary.AddBoundaryIntegrator(new mfem::MassIntegrator(coefficient), truncation_marker_);
    boundary.Assemble();
    boundary.Finalize();
    const std::unique_ptr<mfem::HypreParMatrix> boundary_matrix(boundary.ParallelAssemble());
    wavenumber_system.matrix.reset(mfem::Add(1.0, *stiffness_, 1.0, *boundary_matrix));
    if (mass_) {
      wavenumber_system.matrix.reset(mfem::Add(1.0, *wavenumber_system.matrix, wavenumber_system.k * wavenumber_system.k, *mass_));
    }
  }
  linear_system_setup_ = true;
  stats_.assemble_time += max_time(MPI_Wtime() - start);
}

void DCSolver::setup_linear_solver() {
  MFEM_VERIFY(linear_system_setup_, "Set up the DC linear system before its solver.");
  const double start = MPI_Wtime();
  if (options_.solver == DCSolverType::CG && options_.preconditioner == DCPreconditionerType::LOR) {
    setup_lor_discretization();
  }
  for (int spectral = 0; spectral < static_cast<int>(wavenumber_systems_.size()); ++spectral) {
    if (!wavenumber_systems_[spectral]->solver) {
      setup_wavenumber_solver(spectral);
    }
  }
  stats_.preconditioner_time += max_time(MPI_Wtime() - start);
}

void DCSolver::setup_lor_discretization() {
  if (lor_) {
    return;
  }
  lor_ = std::make_unique<mfem::ParLORDiscretization>(*base_fespace_);
  mfem::ParFiniteElementSpace &lor_fespace = lor_->GetParFESpace();
  const mfem::Array<int> &lor_to_high = lor_->GetDofPermutation();
  MFEM_VERIFY(lor_fespace.GetVSize() == base_fespace_->GetVSize() &&
                  lor_fespace.GetProlongationMatrix()->Width() == base_fespace_->GetTrueVSize() &&
                  lor_to_high.Size() == base_fespace_->GetTrueVSize(),
              "The DC LOR space must transfer the H1 true-DOF layout.");
  for (int dof = 0; dof < lor_to_high.Size(); ++dof) {
    MFEM_VERIFY(lor_to_high[dof] == dof, "The DC LOR space must have the H1 DOF ordering.");
  }
  mfem::ParMesh &lor_mesh = *lor_fespace.GetParMesh();
  lor_truncation_marker_ = truncation_marker(lor_mesh);

  // Vertex (Gauss-Lobatto) rules on tensor-product cells make the low-order mass and boundary terms diagonal and the
  // diffusion stencil an M-matrix.
  lor_rules_ = std::make_unique<mfem::IntegrationRules>(0, mfem::Quadrature1D::GaussLobatto);
  mfem::Array<mfem::Geometry::Type> geometries;
  mesh_->GetGeometries(mesh_->Dimension(), geometries);
  if (geometries.Size() == 1 && mfem::Geometry::IsTensorProduct(geometries[0])) {
    lor_element_rule_ = &lor_rules_->Get(geometries[0], 1);
    lor_face_rule_ = &lor_rules_->Get(mfem::Geometry::TensorProductGeometry(mesh_->Dimension() - 1), 1);
  }
  const double start = MPI_Wtime();
  lor_faces_ = std::make_unique<TruncationFaces>(truncation_faces(lor_mesh, lor_fespace, lor_face_rule_, false));
  lor_alpha_ = shared_coefficients(*lor_faces_, nullptr);
  stats_.boundary_time += max_time(MPI_Wtime() - start);
}

void DCSolver::setup_wavenumber_solver(int spectral) {
  WavenumberSystem &wavenumber_system = *wavenumber_systems_[spectral];
  const double k = wavenumber_system.k;
  if (options_.solver == DCSolverType::Mumps || options_.preconditioner == DCPreconditionerType::Direct) {
    auto direct = std::make_unique<mfem::MUMPSSolver>(comm_);
    direct->SetPrintLevel(0);
    direct->SetMatrixSymType(mfem::MUMPSSolver::MatType::SYMMETRIC_POSITIVE_DEFINITE);
    if (options_.mumps_blr_tolerance > 0.0) {
      direct->SetBLRTol(options_.mumps_blr_tolerance);
    }
    direct->SetOperator(*wavenumber_system.matrix);
    wavenumber_system.solver = std::move(direct);
  } else if (options_.preconditioner == DCPreconditionerType::AMG) {
    wavenumber_system.solver = make_dc_amg(*wavenumber_system.matrix, options_.amg_strength_threshold);
  } else {
    MFEM_VERIFY(lor_, "The DC LOR discretization is not initialized.");
    mfem::ProductCoefficient lor_mass_coefficient(k * k, *sigma_);
    RobinCoefficient<TruncationFaces> lor_boundary_coefficient(*lor_faces_, lor_alpha_, spectral);
    mfem::ParBilinearForm lor_form(&lor_->GetParFESpace());
    auto *diffusion = new mfem::DiffusionIntegrator(*sigma_);
    if (lor_element_rule_) {
      diffusion->SetIntegrationRule(*lor_element_rule_);
    }
    lor_form.AddDomainIntegrator(diffusion);
    if (k != 0.0) {
      auto *spectral_mass = new mfem::MassIntegrator(lor_mass_coefficient);
      if (lor_element_rule_) {
        spectral_mass->SetIntegrationRule(*lor_element_rule_);
      }
      lor_form.AddDomainIntegrator(spectral_mass);
    }
    auto *boundary_mass = new mfem::MassIntegrator(lor_boundary_coefficient);
    if (lor_face_rule_) {
      boundary_mass->SetIntegrationRule(*lor_face_rule_);
    }
    lor_form.AddBoundaryIntegrator(boundary_mass, lor_truncation_marker_);
    lor_form.Assemble();
    mfem::Array<int> essential_dofs;
    mfem::OperatorHandle lor_operator(mfem::Operator::Hypre_ParCSR);
    lor_form.FormSystemMatrix(essential_dofs, lor_operator);
    wavenumber_system.lor_matrix = std::make_unique<mfem::HypreParMatrix>(*lor_operator.As<mfem::HypreParMatrix>());
    MFEM_VERIFY(wavenumber_system.lor_matrix->Height() == wavenumber_system.matrix->Height(),
                "The DC LOR matrix must use the H1 true-DOF layout.");
    wavenumber_system.solver = make_dc_amg(*wavenumber_system.lor_matrix, options_.amg_strength_threshold);
  }
}

void DCSolver::assemble_source(int electrode, ElectrodeSystem &electrode_system) const {
  const std::vector<ElementPoint> elements = finder_->find_all_elements_around_point(electrode_system.position);
  int n_ranks = 0, rank = 0;
  MPI_Comm_size(comm_, &n_ranks);
  MPI_Comm_rank(comm_, &rank);
  int owner = elements.empty() ? n_ranks : rank;
  MPI_Allreduce(MPI_IN_PLACE, &owner, 1, MPI_INT, MPI_MIN, comm_);
  MFEM_VERIFY(owner < n_ranks, fmt::format("Electrode {} is not in the ground of the mesh.", electrode));

  // Unit current: the source vector holds the shape functions at the electrode, in one element of one rank.
  mfem::Vector local(base_fespace_->GetVSize());
  local = 0.0;
  if (owner == rank) {
    const ElementPoint &location = elements.front();
    const mfem::FiniteElement &element = *base_fespace_->GetFE(location.elem);
    mfem::Vector shape(element.GetDof());
    element.CalcShape(location.ip, shape);
    mfem::Array<int> dofs;
    base_fespace_->GetElementDofs(location.elem, dofs);
    local.AddElementVector(dofs, shape);
  }
  electrode_system.rhs.SetSize(base_fespace_->GetTrueVSize());
  base_fespace_->GetProlongationMatrix()->MultTranspose(local, electrode_system.rhs);
}

void DCSolver::add_electrode(int electrode, const mfem::Vector &position) {
  MFEM_VERIFY(linear_system_setup_, "Set up the DC system before adding electrodes.");
  auto electrode_system = std::make_unique<ElectrodeSystem>();
  electrode_system->position = project_xyz_to_model(position, model_.dim());
  assemble_source(electrode, *electrode_system);
  electrode_system->potential = std::make_unique<mfem::ParGridFunction>(fespace_);
  *electrode_system->potential = 0.0;
  electrodes_[electrode] = std::move(electrode_system);
}

void DCSolver::prepare_electrode_boundaries() {
  if (options_.reference == DCBoundaryReference::Shared) {
    return;
  }
  // All ranks hold the same electrodes, so the table work below is collective.
  std::vector<ElectrodeSystem *> pending;
  for (auto &entry : electrodes_) {
    if (!entry.second->boundary_ready) {
      pending.push_back(entry.second.get());
    }
  }
  if (pending.empty()) {
    return;
  }
  const double start = MPI_Wtime();
  const int dim = model_.dim();
  const int spectra = static_cast<int>(wavenumber_systems_.size());
  const bool data = options_.reference == DCBoundaryReference::BoundaryData;
  // Boundary data always come from the layered potential; per-electrode coefficients from the chosen condition.
  const bool layered = data || options_.boundary_condition == DCBoundaryCondition::Layered;
  const TruncationFaces &faces = *faces_;
  const auto xyz = [&](const mfem::Vector &model_point) {
    mfem::Vector result(3);
    result = 0.0;
    for (int axis = 0; axis < dim; ++axis) {
      result[3 - dim + axis] = model_point[axis];
    }
    return result;
  };

  // Electrodes at one depth share the layered potential, so they share tables.
  std::map<double, std::vector<ElectrodeSystem *>> groups;
  for (ElectrodeSystem *electrode_system : pending) {
    groups[electrode_system->position[dim - 1]].push_back(electrode_system);
  }
  long long evaluations = 0, points = 0, clipped = 0, unbounded = 0;
  double ratio_min = std::numeric_limits<double>::infinity(), ratio_max = 0.0;
  int tables = 0;
  int n_ranks = 1, rank = 0;
  MPI_Comm_size(comm_, &n_ranks);
  MPI_Comm_rank(comm_, &rank);
  mfem::Vector position, normal;
  for (const auto &[depth, members] : groups) {
    (void)depth;
    const DCBoundary group_boundary(dim, boundary_->layers(), xyz(members.front()->position), boundary_->decay_switch());
    std::vector<std::unique_ptr<RobinTable>> table;
    if (layered && options_.robin_table != DCRobinTable::Never) {
      // Offset range of every boundary depth over the members, merged over the ranks so that they share the work.
      std::map<double, DepthRange> ranges;
      long long local_points = 0;
      for (std::size_t face = 0; face < faces.rules.size(); ++face) {
        for (int point = 0; point < faces.rules[face]->GetNPoints(); ++point) {
          faces.positions[face].GetColumn(point, position);
          faces.normals[face].GetColumn(point, normal);
          const double z = position[dim - 1];
          auto &range = ranges.try_emplace(z, DepthRange{ z, std::numeric_limits<double>::infinity(), 0.0 }).first->second;
          for (const ElectrodeSystem *member : members) {
            double normal_r = 0.0;
            const double r = ground_offset(dim, position, normal, member->position, normal_r);
            range.r_min = std::min(range.r_min, r);
            range.r_max = std::max(range.r_max, r);
          }
          ++local_points;
        }
      }
      std::vector<double> local;
      for (const auto &[z, range] : ranges) {
        (void)z;
        local.insert(local.end(), { range.z, range.r_min, range.r_max });
      }
      std::vector<int> counts(n_ranks), displacements(n_ranks, 0);
      const int local_count = static_cast<int>(local.size());
      MPI_Allgather(&local_count, 1, MPI_INT, counts.data(), 1, MPI_INT, comm_);
      for (int other = 1; other < n_ranks; ++other) {
        displacements[other] = displacements[other - 1] + counts[other - 1];
      }
      std::vector<double> gathered(displacements.back() + counts.back());
      MPI_Allgatherv(local.data(), local_count, MPI_DOUBLE, gathered.data(), counts.data(), displacements.data(), MPI_DOUBLE,
                     comm_);
      std::map<double, DepthRange> merged;
      for (std::size_t entry = 0; entry < gathered.size(); entry += 3) {
        auto [range, inserted] =
            merged.try_emplace(gathered[entry], DepthRange{ gathered[entry], gathered[entry + 1], gathered[entry + 2] });
        if (!inserted) {
          range->second.r_min = std::min(range->second.r_min, gathered[entry + 1]);
          range->second.r_max = std::max(range->second.r_max, gathered[entry + 2]);
        }
      }
      std::vector<DepthRange> list;
      for (const auto &[z, range] : merged) {
        (void)z;
        list.push_back(range);
      }
      // A table costs its nodes once; direct evaluation costs every point of the busiest rank for every member.
      const long long table_cost = RobinTable::cost(group_boundary, list) * spectra;
      long long direct_cost = local_points * static_cast<long long>(members.size()) * spectra;
      MPI_Allreduce(MPI_IN_PLACE, &direct_cost, 1, MPI_LONG_LONG, MPI_MAX, comm_);
      if (options_.robin_table == DCRobinTable::Always || table_cost < direct_cost * n_ranks) {
        for (int spectral = 0; spectral < spectra; ++spectral) {
          table.push_back(std::make_unique<RobinTable>(group_boundary, wavenumber_systems_[spectral]->k,
                                                       data ? chargeable_ : coefficient_state(), list,
                                                       comm_, data));
        }
        evaluations += rank == 0 ? table_cost : 0;
        ++tables;
      }
    }
    for (ElectrodeSystem *member : members) {
      const mfem::Vector &source = member->position;
      const DCBoundary electrode_boundary(dim, boundary_->layers(), xyz(source), boundary_->decay_switch());
      member->boundary.assign(faces.rules.size(), {});
      for (std::size_t face = 0; face < faces.rules.size(); ++face) {
        const int n_points = faces.rules[face]->GetNPoints();
        std::vector<double> &values = member->boundary[face];
        values.resize(spectra * n_points);
        for (int point = 0; point < n_points; ++point) {
          faces.positions[face].GetColumn(point, position);
          faces.normals[face].GetColumn(point, normal);
          for (int spectral = 0; spectral < spectra; ++spectral) {
            const int index = spectral * n_points + point;
            const double k = wavenumber_systems_[spectral]->k;
            double ratio = 0.0, log_value = 0.0;
            if (!table.empty()) {
              double normal_r = 0.0;
              const double r = ground_offset(dim, position, normal, source, normal_r);
              ratio = table[spectral]->ratio(position[dim - 1], r, normal_r, normal[dim - 1], data ? &log_value : nullptr);
            } else if (layered) {
              ratio = electrode_boundary.robin_ratio(position, normal, k, data ? chargeable_ : coefficient_state(),
                                                     data ? &log_value : nullptr);
              ++evaluations;
            } else {
              ratio = homogeneous_ratio(position, normal, source, k);
            }
            const double alpha_0 = alpha_[face][index];
            if (data) {
              // sigma (dG/dn + alpha_0 G) with dG/dn = -ratio G.
              values[index] = faces.sigma[face] * std::exp(log_value) * (alpha_0 - ratio);
              continue;
            }
            const double alpha = std::max(0.0, ratio);
            values[index] = alpha;
            ++points;
            clipped += ratio < 0.0 ? 1 : 0;
            if (alpha_0 > 0.0) {
              ratio_min = std::min(ratio_min, alpha / alpha_0);
              ratio_max = std::max(ratio_max, alpha / alpha_0);
            } else if (alpha > 0.0) {
              ++unbounded;
            }
          }
        }
      }
    }
    for (const auto &spectral_table : table) {
      evaluations += spectral_table->direct_evaluations();
    }
  }
  for (ElectrodeSystem *electrode_system : pending) {
    electrode_system->boundary_ready = true;
  }
  long long counts[4] = { evaluations, points, clipped, unbounded };
  MPI_Allreduce(MPI_IN_PLACE, counts, 4, MPI_LONG_LONG, MPI_SUM, comm_);
  MPI_Allreduce(MPI_IN_PLACE, &ratio_min, 1, MPI_DOUBLE, MPI_MIN, comm_);
  MPI_Allreduce(MPI_IN_PLACE, &ratio_max, 1, MPI_DOUBLE, MPI_MAX, comm_);
  stats_.robin_evaluations += counts[0];
  stats_.electrode_points += counts[1];
  stats_.electrode_clipped_points += counts[2];
  stats_.unbounded_points += counts[3];
  stats_.ratio_min = std::min(stats_.ratio_min, ratio_min);
  stats_.ratio_max = std::max(stats_.ratio_max, ratio_max);
  stats_.robin_tables += tables;
  stats_.electrode_boundary_time += max_time(MPI_Wtime() - start);
}

std::unique_ptr<mfem::Operator> DCSolver::boundary_correction(const ElectrodeSystem &electrode_system, int spectral) const {
  const TruncationFaces &faces = *faces_;
  auto matrix = std::make_unique<mfem::SparseMatrix>(faces.dofs.Size());
  mfem::Vector coefficient;
  mfem::DenseMatrix element;
  for (std::size_t face = 0; face < faces.rules.size(); ++face) {
    const int n_points = faces.weights[face].Size();
    coefficient.SetSize(n_points);
    for (int point = 0; point < n_points; ++point) {
      const int index = spectral * n_points + point;
      coefficient[point] =
          faces.sigma[face] * (electrode_system.boundary[face][index] - alpha_[face][index]) * faces.weights[face][point];
    }
    element.SetSize(faces.shapes[face].Height());
    mfem::MultADAt(faces.shapes[face], coefficient, element);
    matrix->AddSubMatrix(faces.packed[face], faces.packed[face], element);
  }
  matrix->Finalize();
  return std::make_unique<BoundaryCorrectionOperator>(*base_fespace_->GetProlongationMatrix(), faces.dofs, std::move(matrix));
}

void DCSolver::boundary_load(const ElectrodeSystem &electrode_system, int spectral, mfem::Vector &load) const {
  const TruncationFaces &faces = *faces_;
  mfem::Vector local(base_fespace_->GetVSize()), values, face_load;
  local = 0.0;
  for (std::size_t face = 0; face < faces.rules.size(); ++face) {
    const int n_points = faces.weights[face].Size();
    values.SetSize(n_points);
    for (int point = 0; point < n_points; ++point) {
      values[point] = electrode_system.boundary[face][spectral * n_points + point] * faces.weights[face][point];
    }
    face_load.SetSize(faces.shapes[face].Height());
    faces.shapes[face].Mult(values, face_load);
    local.AddElementVector(faces.vdofs[face], face_load);
  }
  load.SetSize(base_fespace_->GetTrueVSize());
  base_fespace_->GetProlongationMatrix()->MultTranspose(local, load);
}

void DCSolver::solve_wavenumber(int electrode, int spectral, const mfem::Vector &rhs, mfem::Vector &solution) {
  const WavenumberSystem &wavenumber_system = *wavenumber_systems_[spectral];
  MFEM_VERIFY(wavenumber_system.solver, "The DC linear solver is not initialized.");
  ++stats_.n_solves;
  // With per-electrode references the electrode matrix is the shared one plus a boundary term, preconditioned by the
  // shared preconditioner.
  std::unique_ptr<mfem::Operator> correction;
  std::unique_ptr<mfem::SumOperator> electrode_matrix;
  const mfem::Operator *matrix = wavenumber_system.matrix.get();
  if (options_.reference == DCBoundaryReference::PerElectrode) {
    const double start = MPI_Wtime();
    correction = boundary_correction(system(electrode), spectral);
    stats_.electrode_boundary_time += max_time(MPI_Wtime() - start);
    electrode_matrix = std::make_unique<mfem::SumOperator>(matrix, 1.0, correction.get(), 1.0, false, false);
    matrix = electrode_matrix.get();
  }
  if (options_.solver == DCSolverType::Mumps && !correction) {
    wavenumber_system.solver->Mult(rhs, solution);
    log(2, "      k = {: .6E}, MUMPS direct solve", wavenumber_system.k);
    return;
  }
  mfem::CGSolver cg(comm_);
  cg.SetOperator(*matrix);
  cg.SetPreconditioner(*wavenumber_system.solver);
  cg.iterative_mode = false;
  cg.SetRelTol(options_.rtol);
  cg.SetMaxIter(options_.max_iter);
  cg.SetPrintLevel(-1);
  cg.Mult(rhs, solution);
  stats_.max_iterations = std::max(stats_.max_iterations, cg.GetNumIterations());
  stats_.total_iterations += cg.GetNumIterations();
  const double initial_norm = cg.GetInitialNorm();
  log(2, "      k = {: .6E}, CG iterations = {:3d}, relative preconditioned residual = {: .6E}", wavenumber_system.k,
      cg.GetNumIterations(), initial_norm > 0.0 ? cg.GetFinalNorm() / initial_norm : 0.0);
  MFEM_VERIFY(cg.GetConverged(), fmt::format("The DC solve of electrode {} at k={} did not converge in {} iterations. "
                                             "Increase -maxit or choose -pc lor or -pc direct.",
                                             electrode, wavenumber_system.k, cg.GetNumIterations()));
}

void DCSolver::solve() {
  setup_linear_solver();
  prepare_electrode_boundaries();
  const double start = MPI_Wtime();
  const int spectra = static_cast<int>(wavenumber_systems_.size());
  mfem::Vector rhs, load, solution, state(fespace_->GetTrueVSize());
  for (auto &[electrode, electrode_system] : electrodes_) {
    log(2, "    Electrode {}:", electrode);
    for (int spectral = 0; spectral < spectra; ++spectral) {
      rhs = electrode_system->rhs;
      if (options_.reference == DCBoundaryReference::BoundaryData) {
        boundary_load(*electrode_system, spectral, load);
        rhs += load;
      }
      solution.SetSize(rhs.Size());
      solution = 0.0;
      solve_wavenumber(electrode, spectral, rhs, solution);
      for (int dof = 0; dof < solution.Size(); ++dof) {
        state[fespace_->DofToVDof(dof, spectral)] = solution[dof];
      }
    }
    electrode_system->potential->SetFromTrueDofs(state);
  }
  stats_.solve_time += max_time(MPI_Wtime() - start);
}

std::vector<double> DCSolver::potentials(int electrode, const std::vector<mfem::Vector> &points) const {
  const mfem::ParGridFunction &field = *system(electrode).potential;
  const int dim = model_.dim();
  const int spectra = static_cast<int>(wavenumber_systems_.size());
  // Points on cell boundaries lie in several cells, possibly on several ranks; the values are averaged.
  std::vector<double> values(points.size(), 0.0), counts(points.size(), 0.0);
  mfem::Array<int> dofs;
  mfem::Vector shape;
  for (std::size_t index = 0; index < points.size(); ++index) {
    const mfem::Vector point = project_xyz_to_model(points[index], dim);
    for (const ElementPoint &location : finder_->find_all_elements_around_point(point)) {
      const mfem::FiniteElement &element = *base_fespace_->GetFE(location.elem);
      shape.SetSize(element.GetDof());
      element.CalcShape(location.ip, shape);
      base_fespace_->GetElementDofs(location.elem, dofs);
      // The potential is the weighted sum of the spectral potentials (one term in 3D).
      double value = 0.0;
      for (int spectral = 0; spectral < spectra; ++spectral) {
        double spectral_value = 0.0;
        for (int dof = 0; dof < dofs.Size(); ++dof) {
          const int vdof = fespace_->DofToVDof(dofs[dof], spectral);
          spectral_value += shape[dof] * (vdof >= 0 ? field[vdof] : -field[-1 - vdof]);
        }
        value += wavenumber_systems_[spectral]->weight * spectral_value;
      }
      values[index] += value;
      counts[index] += 1.0;
    }
  }
  MPI_Allreduce(MPI_IN_PLACE, values.data(), static_cast<int>(values.size()), MPI_DOUBLE, MPI_SUM, comm_);
  MPI_Allreduce(MPI_IN_PLACE, counts.data(), static_cast<int>(counts.size()), MPI_DOUBLE, MPI_SUM, comm_);
  for (std::size_t index = 0; index < points.size(); ++index) {
    MFEM_VERIFY(counts[index] > 0.0, fmt::format("Point {} is not in the ground of the mesh.", index));
    values[index] /= counts[index];
  }
  return values;
}

std::vector<double> DCSolver::source_currents(int electrode) const {
  const ElectrodeSystem &electrode_system = system(electrode);
  MFEM_VERIFY(stiffness_, "The DC linear system is not assembled.");
  mfem::Vector state(fespace_->GetTrueVSize()), component(base_fespace_->GetTrueVSize()), flux(component.Size()),
      stiffness_action(component.Size()), load;
  electrode_system.potential->GetTrueDofs(state);
  std::vector<double> currents(wavenumber_systems_.size());
  for (int spectral = 0; spectral < static_cast<int>(wavenumber_systems_.size()); ++spectral) {
    for (int dof = 0; dof < component.Size(); ++dof) {
      component[dof] = state[fespace_->DofToVDof(dof, spectral)];
    }
    // (A u - K u - f) summed over the DOFs: the boundary and k^2 terms tested with the constant function.
    wavenumber_systems_[spectral]->matrix->Mult(component, flux);
    if (options_.reference == DCBoundaryReference::PerElectrode) {
      boundary_correction(electrode_system, spectral)->AddMult(component, flux);
    } else if (options_.reference == DCBoundaryReference::BoundaryData) {
      boundary_load(electrode_system, spectral, load);
      flux -= load;
    }
    stiffness_->Mult(component, stiffness_action);
    flux -= stiffness_action;
    currents[spectral] = flux.Sum();
  }
  MPI_Allreduce(MPI_IN_PLACE, currents.data(), static_cast<int>(currents.size()), MPI_DOUBLE, MPI_SUM, comm_);
  return currents;
}

} // namespace dcip
