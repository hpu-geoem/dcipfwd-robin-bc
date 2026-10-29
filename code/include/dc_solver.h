// Finite-element DC solver for the potentials of unit point currents in 3D and 2.5D, with a Robin truncation boundary.

#pragma once

#include "dc_boundary.h"
#include "integral_transforms.h"

#include <mfem.hpp>

#include <limits>
#include <map>
#include <memory>
#include <vector>

namespace dcip {

class MeshFinder;
class Model;

/// Linear solver of the electrode systems: preconditioned conjugate gradients or the MUMPS factorization.
enum class DCSolverType { CG, Mumps };

/// Preconditioner of the conjugate gradient solves: BoomerAMG on the high-order matrix, BoomerAMG on its
/// low-order-refined matrix, or the MUMPS factorization of the shared matrix.
enum class DCPreconditionerType { AMG, LOR, Direct };

/// Robin coefficient -dG/dn / G of the truncation boundary: from the layered background or from a homogeneous
/// half-space (Dey and Morrison).
enum class DCBoundaryCondition { Layered, DeyMorrison };

/// How the boundary condition treats the current electrodes.
/// Shared: the reference source of the DCBoundary for every electrode, hence one matrix for all.
/// PerElectrode: every electrode is its own reference; its matrix is the shared matrix plus a boundary mass term, and
///   the shared preconditioner serves all electrodes.
/// BoundaryData: the shared matrix, with the inhomogeneous Robin data dG_e/dn + alpha_0 G_e of the layered potential
///   G_e of each electrode in its right-hand side.
enum class DCBoundaryReference { Shared, PerElectrode, BoundaryData };

/// How per-electrode layered coefficients are evaluated: by interpolation tables per source depth when they need fewer
/// potential evaluations than direct evaluation (Auto), always by tables, or always directly.
enum class DCRobinTable { Auto, Always, Never };

/// Discretization and linear-solver options.
struct DCSolverOptions {
  int order = 1;                                                         // Polynomial order of the H1 space
  int uniform_refinements = 0;                                           // Uniform refinements of the mesh
  DCSolverType solver = DCSolverType::CG;                                // Linear solver of the electrode systems
  DCPreconditionerType preconditioner = DCPreconditionerType::AMG;       // Preconditioner of conjugate gradients
  int max_iter = 100;                                                    // Iteration limit of conjugate gradients
  double rtol = 1E-8;                                                    // Relative residual tolerance of conjugate gradients
  double kx_min = 0.0;                                                   // 2.5D: start of the logarithmic panels
  double kx_max = 0.0;                                                   // 2.5D: largest wavenumber
  int kx_points_per_decade = 4;                                          // 2.5D: Gauss points per logarithmic panel
  std::vector<WavenumberNode> wavenumbers;                               // 2.5D: explicit rule; empty selects the log rule
  DCBoundaryCondition boundary_condition = DCBoundaryCondition::Layered; // Robin coefficient of the truncation boundary
  DCBoundaryReference reference = DCBoundaryReference::Shared;           // Treatment of the current electrodes
  DCRobinTable robin_table = DCRobinTable::Auto;                         // Per-electrode coefficient evaluation
  double amg_strength_threshold = 0.6;                                   // BoomerAMG strength threshold
  double mumps_blr_tolerance = 0.0;                                      // MUMPS block low-rank tolerance; zero disables it
  bool dc_coefficients = false; // Instantaneous pass: Robin coefficients of the DC background (boundary data keep their own)
};

/// Sizes, times (seconds, maximum over ranks) and boundary diagnostics of a DC solver.
/// Boundary points count the quadrature points of the truncation faces times the wavenumbers (and times the
/// electrodes for per-electrode values), summed over ranks.
struct DCSolverStats {
  long long cells = 0;                  // Global cells of the refined mesh
  long long dofs = 0;                   // Global true DOFs of one wavenumber system
  long long truncation_faces = 0;       // Global faces of the truncation boundary
  int wavenumbers = 0;                  // Wavenumber systems (1 in 3D)
  double refine_time = 0.0;             // Uniform and electrode refinement
  double fe_time = 0.0;                 // Spaces, truncation faces and shared Robin coefficients
  double boundary_time = 0.0;           // Shared Robin coefficients (part of fe_time and of the LOR setup)
  double assemble_time = 0.0;           // Assembly of the system matrices
  double preconditioner_time = 0.0;     // AMG, LOR-AMG or MUMPS setup of all wavenumbers
  double solve_time = 0.0;              // All electrode solves
  double electrode_boundary_time = 0.0; // Per-electrode coefficients or boundary data, and boundary terms
  long long robin_evaluations = 0;      // Layered potential evaluations for per-electrode values (all ranks)
  int robin_tables = 0;                 // Source-depth groups whose values came from tables
  int max_iterations = 0;               // Largest CG iteration count of one wavenumber solve
  long long total_iterations = 0;       // CG iterations of all solves
  int n_solves = 0;                     // Wavenumber solves
  long long boundary_points = 0;        // Points of the shared coefficient
  long long clipped_points = 0;         // Points where the shared -dG/dn / G was negative and set to zero
  long long electrode_points = 0;       // Points of the per-electrode coefficients
  long long electrode_clipped_points = 0; // Of these, points where -dG_e/dn / G_e was negative and set to zero
  long long unbounded_points = 0;       // Per-electrode points with alpha_e > 0 where alpha_0 = 0
  double ratio_min = std::numeric_limits<double>::infinity(); // Smallest alpha_e / alpha_0 where alpha_0 > 0
  double ratio_max = 0.0;                                     // Largest alpha_e / alpha_0 where alpha_0 > 0
};

/// Total-potential DC solver on a ground mesh (z down, no air) for unit currents at point electrodes.
/// One matrix and one preconditioner per wavenumber (one in 3D) serve all electrodes.
class DCSolver {
public:
  /// Take the model, the options, the background and shared reference of the boundary condition, and whether to use
  /// the instantaneous conductivity sigma / (1 - eta).
  DCSolver(MPI_Comm comm, const Model &model, const DCSolverOptions &options, std::shared_ptr<const DCBoundary> boundary,
           bool chargeable = false);
  ~DCSolver();

  DCSolver(const DCSolver &) = delete;
  DCSolver &operator=(const DCSolver &) = delete;

  /// Refine the cells around points (x, y, z) a number of times; call before refine_mesh().
  void add_points_to_refine(const std::vector<mfem::Vector> &points, int count) { points_to_refine_.emplace_back(points, count); }

  /// Apply the uniform and point refinements.
  void refine_mesh();

  /// Build the spaces, the shared Robin coefficients, the matrices and their preconditioners or factorizations.
  void setup_system();

  /// Add the unit current of an electrode at (x, y, z).
  void add_electrode(int electrode, const mfem::Vector &position);

  /// Solve the potentials of all added electrodes.
  void solve();

  /// Collectively return the potential of the unit current of an electrode at points (x, y, z).
  std::vector<double> potentials(int electrode, const std::vector<mfem::Vector> &points) const;

  /// Return, per wavenumber, the current that leaves through the truncation boundary plus, in 2.5D, the k^2 mass term;
  /// it equals the unit source current for a converged solution.
  std::vector<double> source_currents(int electrode) const;

  const DCSolverStats &stats() const { return stats_; }

private:
  struct ElectrodeSystem;
  struct WavenumberSystem;
  struct TruncationFaces;

  ElectrodeSystem &system(int electrode);
  const ElectrodeSystem &system(int electrode) const;
  void setup_fe_system();
  void setup_linear_system();
  void setup_linear_solver();
  void setup_lor_discretization();
  void setup_wavenumber_solver(int spectral);
  TruncationFaces truncation_faces(mfem::ParMesh &mesh, const mfem::ParFiniteElementSpace &fespace,
                                   const mfem::IntegrationRule *face_rule, bool with_shapes) const;
  std::vector<std::vector<double>> shared_coefficients(const TruncationFaces &faces, long long *clipped) const;
  double shared_ratio(const mfem::Vector &position, const mfem::Vector &normal, double k) const;
  void prepare_electrode_boundaries();
  std::unique_ptr<mfem::Operator> boundary_correction(const ElectrodeSystem &electrode_system, int spectral) const;
  void boundary_load(const ElectrodeSystem &electrode_system, int spectral, mfem::Vector &load) const;
  void solve_wavenumber(int electrode, int spectral, const mfem::Vector &rhs, mfem::Vector &solution);
  double conductivity(int attribute) const;
  bool coefficient_state() const { return chargeable_ && !options_.dc_coefficients; }
  void assemble_source(int electrode, ElectrodeSystem &electrode_system) const;
  double max_time(double seconds) const;

  MPI_Comm comm_;
  const Model &model_;
  DCSolverOptions options_;
  bool chargeable_ = false;
  std::shared_ptr<const DCBoundary> boundary_;
  std::vector<std::pair<std::vector<mfem::Vector>, int>> points_to_refine_;
  std::unique_ptr<mfem::ParMesh> mesh_;
  std::unique_ptr<mfem::H1_FECollection> fec_;
  std::unique_ptr<mfem::ParFiniteElementSpace> base_fespace_;     // Scalar space of one wavenumber
  std::unique_ptr<mfem::ParFiniteElementSpace> spectral_fespace_; // All wavenumbers (2.5D)
  mfem::ParFiniteElementSpace *fespace_ = nullptr;                // Space of the potentials
  std::unique_ptr<MeshFinder> finder_;
  std::shared_ptr<mfem::Coefficient> sigma_;
  mfem::Array<int> truncation_marker_;
  std::unique_ptr<TruncationFaces> faces_;     // Truncation faces of the mesh with shape values
  std::vector<std::vector<double>> alpha_;     // Shared coefficient per face, wavenumber-major
  std::unique_ptr<mfem::HypreParMatrix> stiffness_;
  std::unique_ptr<mfem::HypreParMatrix> mass_;
  std::unique_ptr<mfem::ParLORDiscretization> lor_;
  mfem::Array<int> lor_truncation_marker_;
  std::unique_ptr<TruncationFaces> lor_faces_; // Truncation faces of the LOR mesh
  std::vector<std::vector<double>> lor_alpha_;
  std::unique_ptr<mfem::IntegrationRules> lor_rules_;
  const mfem::IntegrationRule *lor_element_rule_ = nullptr;
  const mfem::IntegrationRule *lor_face_rule_ = nullptr;
  std::vector<std::unique_ptr<WavenumberSystem>> wavenumber_systems_;
  std::map<int, std::unique_ptr<ElectrodeSystem>> electrodes_;
  bool fe_system_setup_ = false;
  bool linear_system_setup_ = false;
  DCSolverStats stats_;
};

/// Return a BoomerAMG preconditioner of a DC matrix (one V-cycle, symmetric l1 hybrid Gauss-Seidel smoothing) with
/// its setup done, so that the setup time is not counted as solve time.
std::unique_ptr<mfem::HypreBoomerAMG> make_dc_amg(mfem::HypreParMatrix &matrix, double strength_threshold);

} // namespace dcip
