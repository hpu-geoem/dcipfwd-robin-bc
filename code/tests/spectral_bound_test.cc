// Spectral equivalence of per-electrode and shared operators, and symmetry and positivity of the AMG preconditioner.

#include "test_common.h"

#include <fmt/format.h>

#include <cmath>
#include <vector>

namespace {

constexpr double kRho1 = 100.0, kRho2 = 1000.0, kThickness = 40.0;
constexpr double kHalfWidth = 400.0, kDepth = 300.0;

/// Section (y, z) of a two-layer earth; MakeCartesian2D marks the surface z = 0 with attribute 1.
mfem::Mesh section(int cells) {
  mfem::Mesh mesh = mfem::Mesh::MakeCartesian2D(2 * cells, cells, mfem::Element::QUADRILATERAL, true, 2.0 * kHalfWidth, kDepth);
  mesh.Transform([](const mfem::Vector &input, mfem::Vector &output) {
    output.SetSize(2);
    output[0] = input[0] - kHalfWidth;
    output[1] = input[1];
  });
  return mesh;
}

double conductivity(double z) { return z < kThickness ? 1.0 / kRho1 : 1.0 / kRho2; }

/// Outward unit normal of a truncation point of the section.
mfem::Vector outward_normal(const mfem::Vector &position) {
  if (position[0] <= -kHalfWidth + 1.0e-9) {
    return mfem::Vector({ -1.0, 0.0 });
  }
  if (position[0] >= kHalfWidth - 1.0e-9) {
    return mfem::Vector({ 1.0, 0.0 });
  }
  return mfem::Vector({ 0.0, 1.0 });
}

/// sigma alpha of a reference source on the truncation boundary.
class RobinTerm : public mfem::Coefficient {
public:
  RobinTerm(const DCBoundary &boundary, double k) : boundary_(boundary), k_(k) {}
  double Eval(mfem::ElementTransformation &transformation, const mfem::IntegrationPoint &point) override {
    mfem::Vector position;
    transformation.Transform(point, position);
    return conductivity(position[1]) * boundary_.robin_factor(position, outward_normal(position), k_, false);
  }

private:
  const DCBoundary &boundary_;
  double k_;
};

class Conductivity : public mfem::Coefficient {
public:
  explicit Conductivity(double scale = 1.0) : scale_(scale) {}
  double Eval(mfem::ElementTransformation &transformation, const mfem::IntegrationPoint &point) override {
    mfem::Vector position;
    transformation.Transform(point, position);
    return scale_ * conductivity(position[1]);
  }

private:
  double scale_;
};

std::vector<BackgroundLayer> background() { return { { 0.0, 1.0 / kRho1, 0.0 }, { kThickness, 1.0 / kRho2, 0.0 } }; }

/// Dense matrix of K + k^2 M + B for the Robin coefficient of a reference source.
mfem::DenseMatrix operator_matrix(mfem::FiniteElementSpace &space, const DCBoundary &boundary, double k) {
  Conductivity sigma, reaction(k * k);
  RobinTerm robin(boundary, k);
  mfem::Array<int> truncation({ 0, 1, 1, 1 });
  mfem::BilinearForm form(&space);
  form.AddDomainIntegrator(new mfem::DiffusionIntegrator(sigma));
  form.AddDomainIntegrator(new mfem::MassIntegrator(reaction));
  form.AddBoundaryIntegrator(new mfem::MassIntegrator(robin), truncation);
  form.Assemble();
  form.Finalize();
  mfem::DenseMatrix dense;
  form.SpMat().ToDenseMatrix(dense);
  mfem::DenseMatrix transpose(dense, 't');
  dense += transpose;
  dense *= 0.5;
  return dense;
}

} // namespace

TEST_CASE("Per-electrode operators are spectrally equivalent to the shared operator", "[dc][bound]") {
  mfem::Mesh mesh = section(6);
  mfem::H1_FECollection fec(2, 2);
  mfem::FiniteElementSpace space(&mesh, &fec);
  const double k = 2.0e-3;
  const DCBoundary shared(2, background(), mfem::Vector({ 0.0, 0.0, 0.0 }));
  for (double offset : { 60.0, 250.0 }) {
    const DCBoundary electrode(2, background(), mfem::Vector({ 0.0, offset, 0.0 }));
    // Range of alpha_e / alpha_0 sampled densely along the truncation boundary.
    double gamma_min = std::numeric_limits<double>::infinity(), gamma_max = 0.0;
    for (int sample = 0; sample <= 4000; ++sample) {
      const double s = static_cast<double>(sample) / 4000.0 * (2.0 * kDepth + 2.0 * kHalfWidth);
      const mfem::Vector position = s < kDepth                       ? mfem::Vector({ -kHalfWidth, s })
                                    : s < kDepth + 2.0 * kHalfWidth ? mfem::Vector({ s - kDepth - kHalfWidth, kDepth })
                                                                    : mfem::Vector({ kHalfWidth, 2.0 * kDepth + 2.0 * kHalfWidth - s });
      const mfem::Vector normal = outward_normal(position);
      const double ratio = electrode.robin_factor(position, normal, k, false) / shared.robin_factor(position, normal, k, false);
      gamma_min = std::min(gamma_min, ratio);
      gamma_max = std::max(gamma_max, ratio);
    }
    // Generalized eigenvalues of (A_e, A_0) from A_0^(-1/2) A_e A_0^(-1/2).
    mfem::DenseMatrix a0 = operator_matrix(space, shared, k), ae = operator_matrix(space, electrode, k);
    mfem::Vector values0, values;
    mfem::DenseMatrix vectors0, vectors;
    a0.Eigensystem(values0, vectors0);
    REQUIRE(values0.Min() > 0.0);
    mfem::DenseMatrix scaled(vectors0), inverse_root(a0.Height()), product(a0.Height()), congruent(a0.Height());
    for (int column = 0; column < scaled.Width(); ++column) {
      for (int row = 0; row < scaled.Height(); ++row) {
        scaled(row, column) /= std::sqrt(values0[column]);
      }
    }
    mfem::MultABt(scaled, vectors0, inverse_root);
    mfem::Mult(inverse_root, ae, product);
    mfem::Mult(product, inverse_root, congruent);
    mfem::DenseMatrix transpose(congruent, 't');
    congruent += transpose;
    congruent *= 0.5;
    congruent.Eigensystem(values, vectors);
    const double lower = std::min(1.0, gamma_min), upper = std::max(1.0, gamma_max);
    CAPTURE(offset, gamma_min, gamma_max, values.Min(), values.Max());
    if (test_common::is_root()) {
      fmt::print("Spectral bound, offset {} m: eigenvalues [{:.4f}, {:.4f}] within [{:.4f}, {:.4f}]\n", offset, values.Min(),
                 values.Max(), lower, upper);
    }
    CHECK(values.Min() >= lower * (1.0 - 1.0e-9));
    CHECK(values.Max() <= upper * (1.0 + 1.0e-9));
    CHECK(values.Max() > 1.0 + 1.0e-6); // The electrode operator differs from the shared one.
  }
}

namespace {

constexpr double kBox = 800.0, kBoxDepth = 300.0;

/// sigma alpha of a shared reference at the centre of the surface of a 3D box; outward normals from the position.
class BoxRobinTerm : public mfem::Coefficient {
public:
  explicit BoxRobinTerm(const DCBoundary &boundary) : boundary_(boundary) {}
  double Eval(mfem::ElementTransformation &transformation, const mfem::IntegrationPoint &point) override {
    mfem::Vector x;
    transformation.Transform(point, x);
    mfem::Vector normal({ 0.0, 0.0, 0.0 });
    if (x[0] < 1.0e-9) {
      normal[0] = -1.0;
    } else if (x[0] > kBox - 1.0e-9) {
      normal[0] = 1.0;
    } else if (x[1] < 1.0e-9) {
      normal[1] = -1.0;
    } else if (x[1] > kBox - 1.0e-9) {
      normal[1] = 1.0;
    } else {
      normal[2] = 1.0;
    }
    return conductivity(x[2]) * boundary_.robin_factor(x, normal, 0.0, false);
  }

private:
  const DCBoundary &boundary_;
};

/// Dense matrix of the action of a preconditioner on the unit vectors.
mfem::DenseMatrix dense_action(mfem::Solver &preconditioner, int n) {
  mfem::DenseMatrix action(n);
  mfem::Vector unit(n), column(n);
  for (int j = 0; j < n; ++j) {
    unit = 0.0;
    unit[j] = 1.0;
    preconditioner.Mult(unit, column);
    action.SetCol(j, column);
  }
  return action;
}

} // namespace

TEST_CASE("The AMG and LOR-AMG preconditioners of a DC matrix are symmetric positive definite", "[dc][bound][solver]") {
  // Two-layer 3D box with hanging nodes, assembled as in DCSolver: AMG on the second-order matrix, and AMG on the
  // low-order-refined matrix with vertex quadrature; each process holds the whole small problem.
  mfem::Mesh serial = mfem::Mesh::MakeCartesian3D(4, 4, 3, mfem::Element::HEXAHEDRON, kBox, kBox, kBoxDepth);
  serial.EnsureNCMesh();
  mfem::Array<int> refine({ 0, 21 });
  serial.GeneralRefinement(refine);
  mfem::ParMesh mesh(MPI_COMM_SELF, serial);
  REQUIRE(!mesh.Conforming());
  mfem::H1_FECollection fec(2, 3);
  mfem::ParFiniteElementSpace space(&mesh, &fec);
  const DCBoundary shared(3, background(), mfem::Vector({ 0.5 * kBox, 0.5 * kBox, 0.0 }));
  Conductivity sigma;
  BoxRobinTerm robin(shared);
  mfem::Array<int> truncation({ 0, 1, 1, 1, 1, 1 }); // MakeCartesian3D: attribute 1 is z = 0, the surface
  std::unique_ptr<mfem::HypreParMatrix> matrix;
  {
    mfem::ParBilinearForm form(&space);
    form.AddDomainIntegrator(new mfem::DiffusionIntegrator(sigma));
    form.AddBoundaryIntegrator(new mfem::MassIntegrator(robin), truncation);
    form.Assemble();
    form.Finalize();
    matrix.reset(form.ParallelAssemble());
  }
  mfem::ParLORDiscretization lor(space);
  mfem::IntegrationRules vertex_rules(0, mfem::Quadrature1D::GaussLobatto);
  std::unique_ptr<mfem::HypreParMatrix> lor_matrix;
  {
    mfem::ParBilinearForm form(&lor.GetParFESpace());
    auto *diffusion = new mfem::DiffusionIntegrator(sigma);
    diffusion->SetIntegrationRule(vertex_rules.Get(mfem::Geometry::CUBE, 1));
    form.AddDomainIntegrator(diffusion);
    auto *boundary = new mfem::MassIntegrator(robin);
    boundary->SetIntegrationRule(vertex_rules.Get(mfem::Geometry::SQUARE, 1));
    form.AddBoundaryIntegrator(boundary, truncation);
    form.Assemble();
    mfem::Array<int> essential;
    mfem::OperatorHandle handle(mfem::Operator::Hypre_ParCSR);
    form.FormSystemMatrix(essential, handle);
    lor_matrix = std::make_unique<mfem::HypreParMatrix>(*handle.As<mfem::HypreParMatrix>());
  }
  REQUIRE(lor_matrix->Height() == matrix->Height());
  for (mfem::HypreParMatrix *operator_matrix : { matrix.get(), lor_matrix.get() }) {
    auto amg = make_dc_amg(*operator_matrix, 0.6);
    const mfem::DenseMatrix action = dense_action(*amg, operator_matrix->Height());
    mfem::DenseMatrix transpose(action, 't'), difference(action);
    difference -= transpose;
    mfem::DenseMatrix symmetric(action);
    symmetric += transpose;
    symmetric *= 0.5;
    mfem::Vector values;
    mfem::DenseMatrix vectors;
    symmetric.Eigensystem(values, vectors);
    CAPTURE(operator_matrix == lor_matrix.get(), difference.MaxMaxNorm(), action.MaxMaxNorm(), values.Min(), values.Max());
    CHECK(difference.MaxMaxNorm() <= 1.0e-10 * action.MaxMaxNorm());
    CHECK(values.Min() > 1.0e-8 * values.Max());
  }
}
