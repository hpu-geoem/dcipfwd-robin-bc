// dcipfwd: forward DC resistivity and integral-chargeability IP responses in 3D and 2.5D.
// Writes <prefix>.dat (responses), <prefix>.json (options, sizes, times, diagnostics) and optionally <prefix>.pot
// (unit-current potentials at the potential electrodes).

#include "dc_boundary.h"
#include "dc_solver.h"
#include "log.h"
#include "model.h"
#include "survey.h"
#include "utils.h"
#include "version.h"
#include "wavenumber_rule.h"

#include <fmt/format.h>
#include <fmt/ranges.h>

#include <mfem.hpp>

#include <sys/resource.h>

#include <algorithm>
#include <cmath>
#include <fstream>
#include <map>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

using namespace dcip;

namespace {

/// Command-line settings of a run.
struct Settings {
  const char *mesh_file = "model.mesh";
  const char *property_file = "model.prop";
  const char *survey_file = "survey.txt";
  const char *output_prefix = "result";
  const char *background = "corner";   // "corner" or a file of "top resistivity [chargeability]" rows
  const char *boundary = "layered";    // "layered" or "dey-morrison"
  const char *reference = "centroid";  // "centroid", "x,y,z", "electrode", "electrode-matrix" or "electrode-data"
  const char *solver = "cg";           // "cg" or "mumps"
  const char *preconditioner = "amg";  // "amg", "lor" or "direct"
  const char *ip = "auto";             // "auto", "yes" or "no"
  const char *robin_table = "auto";    // "auto", "always" or "never"
  const char *wavenumber_rule = "log"; // "log", "layered", "layered-joint" or "layered-separate"
  const char *ip_coefficients = "own"; // "own" or "dc"
  int tx_refinements = 0;
  int rx_refinements = 0;
  double surface_z = std::nan("");
  double decay_switch = LayeredPotential::kDecaySwitch;
  bool write_potentials = false;
  bool verbose = false;
};

/// Potentials of the unit currents at the potential electrodes: values[current][receiver].
using PotentialTable = std::map<int, std::map<int, double>>;

/// Sizes and times of one forward pass (DC or instantaneous).
struct PassReport {
  std::string name;
  std::vector<DCSolverStats> stats; // One entry per solver (one per electrode for electrode-matrix)
  double wall_time = 0.0;
};

/// A-priori errors of the 2.5D rule against the layered background of one conductivity state.
struct RuleReport {
  std::vector<WavenumberNode> nodes;
  double error = std::nan("");     // Rule in use
  double error_log = std::nan(""); // Logarithmic Gauss rule
};

int rank() {
  int value = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &value);
  return value;
}

mfem::Vector parse_point(const std::string &text) {
  std::vector<double> values;
  std::stringstream stream(text);
  std::string item;
  while (std::getline(stream, item, ',')) {
    values.push_back(std::stod(item));
  }
  MFEM_VERIFY(values.size() == 3, "A reference point is given as x,y,z.");
  return mfem::Vector({ values[0], values[1], values[2] });
}

std::vector<mfem::Vector> positions(const Survey &survey, const std::vector<int> &indices) {
  std::vector<mfem::Vector> result;
  for (int index : indices) {
    result.push_back(survey.electrodes()[index]);
  }
  return result;
}

/// Solve the unit-current potentials of a set of current electrodes with one shared reference.
void solve_group(const Model &model, const Survey &survey, const DCSolverOptions &options, const Settings &settings,
                 const std::vector<BackgroundLayer> &background, const mfem::Vector &reference, const std::vector<int> &currents,
                 bool chargeable, PotentialTable &table, PassReport &report) {
  const std::vector<int> receivers = survey.potential_electrodes();
  const std::vector<mfem::Vector> receiver_points = positions(survey, receivers);
  auto boundary = std::make_shared<const DCBoundary>(model.dim(), background, reference, settings.decay_switch);
  DCSolver solver(MPI_COMM_WORLD, model, options, boundary, chargeable);
  // Refine around every electrode so that all groups share one mesh.
  solver.add_points_to_refine(positions(survey, survey.current_electrodes()), settings.tx_refinements);
  solver.add_points_to_refine(receiver_points, settings.rx_refinements);
  solver.refine_mesh();
  solver.setup_system();
  for (int electrode : currents) {
    solver.add_electrode(electrode, survey.electrodes()[electrode]);
  }
  solver.solve();
  for (int electrode : currents) {
    const std::vector<double> values = solver.potentials(electrode, receiver_points);
    for (std::size_t index = 0; index < receivers.size(); ++index) {
      table[electrode][receivers[index]] = values[index];
    }
  }
  report.stats.push_back(solver.stats());
}

PotentialTable forward_pass(const Model &model, const Survey &survey, DCSolverOptions options, const Settings &settings,
                            const std::vector<BackgroundLayer> &background, bool chargeable, PassReport &report) {
  const double start = MPI_Wtime();
  PotentialTable table;
  const std::vector<int> currents = survey.current_electrodes();
  const std::string reference = settings.reference;
  if (reference == "electrode-matrix") {
    // One matrix and preconditioner per current electrode, with the electrode as its own reference.
    for (int electrode : currents) {
      solve_group(model, survey, options, settings, background, survey.electrodes()[electrode], { electrode }, chargeable,
                  table, report);
    }
  } else {
    if (reference == "electrode") {
      options.reference = DCBoundaryReference::PerElectrode;
    } else if (reference == "electrode-data") {
      options.reference = DCBoundaryReference::BoundaryData;
    }
    const bool fixed = reference != "centroid" && reference != "electrode" && reference != "electrode-data";
    const mfem::Vector point = fixed ? parse_point(reference) : survey.current_centroid();
    solve_group(model, survey, options, settings, background, point, currents, chargeable, table, report);
  }
  report.wall_time = MPI_Wtime() - start;
  return table;
}

/// Return the peak resident memory of this process in MiB.
double peak_memory_mib() {
  rusage usage{};
  getrusage(RUSAGE_SELF, &usage);
#ifdef __APPLE__
  return static_cast<double>(usage.ru_maxrss) / (1024.0 * 1024.0);
#else
  return static_cast<double>(usage.ru_maxrss) / 1024.0;
#endif
}

std::string number(double value) { return std::isfinite(value) ? fmt::format("{:.6e}", value) : std::string("null"); }

std::string stats_json(const DCSolverStats &stats) {
  return fmt::format(R"({{"cells": {}, "dofs": {}, "truncation_faces": {}, "wavenumbers": {}, "refine_time": {:.6g}, )"
                     R"("fe_time": {:.6g}, "boundary_time": {:.6g}, "assemble_time": {:.6g}, "preconditioner_time": {:.6g}, )"
                     R"("solve_time": {:.6g}, "electrode_boundary_time": {:.6g}, "robin_evaluations": {}, "robin_tables": {}, )"
                     R"("max_iterations": {}, "total_iterations": {}, "n_solves": {}, "boundary_points": {}, )"
                     R"("clipped_points": {}, "electrode_points": {}, "electrode_clipped_points": {}, "unbounded_points": {}, )"
                     R"("ratio_min": {}, "ratio_max": {}}})",
                     stats.cells, stats.dofs, stats.truncation_faces, stats.wavenumbers, stats.refine_time, stats.fe_time,
                     stats.boundary_time, stats.assemble_time, stats.preconditioner_time, stats.solve_time,
                     stats.electrode_boundary_time, stats.robin_evaluations, stats.robin_tables, stats.max_iterations,
                     stats.total_iterations, stats.n_solves, stats.boundary_points, stats.clipped_points, stats.electrode_points,
                     stats.electrode_clipped_points, stats.unbounded_points,
                     stats.electrode_points > 0 ? number(stats.ratio_min) : "null",
                     stats.electrode_points > 0 ? number(stats.ratio_max) : "null");
}

std::string rule_json(const std::string &key, const RuleReport &rule) {
  std::vector<double> ks, weights;
  for (const WavenumberNode &node : rule.nodes) {
    ks.push_back(node.k);
    weights.push_back(node.weight);
  }
  return fmt::format(R"(  "{0}": {{"error": {1}, "error_log": {2}, "nodes": [{3:.17g}], "weights": [{4:.17g}]}},)", key,
                     number(rule.error), number(rule.error_log), fmt::join(ks, ", "), fmt::join(weights, ", "));
}

} // namespace

int main(int argc, char *argv[]) {
  mfem::Mpi::Init(&argc, &argv);
  mfem::Hypre::Init();

  Settings settings;
  DCSolverOptions options;
  mfem::OptionsParser args(argc, argv);
  args.AddOption(&settings.mesh_file, "-m", "--mesh", "MFEM mesh; 3D (x, y, z) or 2D (y, z) for 2.5D, z down.");
  args.AddOption(&settings.property_file, "-p", "--properties", "Rows 'attribute resistivity [chargeability]'.");
  args.AddOption(&settings.survey_file, "-s", "--survey", "Electrodes and 'a b m n' measurements (-1 = pole).");
  args.AddOption(&settings.output_prefix, "-o", "--output", "Prefix of the outputs.");
  args.AddOption(&settings.background, "-bg", "--background", "Layered background: 'corner' or a file of 'top rho [eta]' rows.");
  args.AddOption(&settings.boundary, "-bc", "--boundary", "Robin coefficient: 'layered' or 'dey-morrison' (homogeneous).");
  args.AddOption(&settings.reference, "-ref", "--reference",
                 "Electrode treatment: 'centroid' or 'x,y,z' (shared reference), 'electrode' (own reference, shared "
                 "preconditioner), 'electrode-matrix' (own reference and matrix), 'electrode-data' (shared coefficient, "
                 "layered potential of each electrode as inhomogeneous Robin data).");
  args.AddOption(&options.order, "-order", "--order", "Polynomial order.");
  args.AddOption(&settings.solver, "-solver", "--solver", "Linear solver: 'cg' or 'mumps'.");
  args.AddOption(&settings.preconditioner, "-pc", "--preconditioner", "CG preconditioner: 'amg', 'lor' or 'direct'.");
  args.AddOption(&options.rtol, "-rtol", "--rtol", "Relative residual tolerance of CG.");
  args.AddOption(&options.max_iter, "-maxit", "--max-iterations", "Iteration limit of CG.");
  args.AddOption(&options.amg_strength_threshold, "-amg-theta", "--amg-strength", "BoomerAMG strength threshold.");
  args.AddOption(&options.mumps_blr_tolerance, "-blr-tol", "--mumps-blr-tolerance", "MUMPS block low-rank tolerance (0: off).");
  args.AddOption(&options.uniform_refinements, "-ru", "--uniform-refinements", "Uniform refinements.");
  args.AddOption(&settings.tx_refinements, "-rtx", "--tx-refinements", "Refinements around current electrodes.");
  args.AddOption(&settings.rx_refinements, "-rrx", "--rx-refinements", "Refinements around potential electrodes.");
  args.AddOption(&options.kx_min, "-kmin", "--kx-min", "2.5D: start of the logarithmic panels (0: 1e-3 / max distance).");
  args.AddOption(&options.kx_max, "-kmax", "--kx-max", "2.5D: largest wavenumber (0: 15 / min distance).");
  args.AddOption(&options.kx_points_per_decade, "-kppd", "--k-points-per-decade", "2.5D: Gauss points per decade.");
  args.AddOption(&settings.wavenumber_rule, "-kq", "--wavenumber-rule",
                 "2.5D weights: 'log' (Gauss in ln k), 'layered' (fitted to the DC background), 'layered-joint' (one fit "
                 "to the DC and instantaneous backgrounds) or 'layered-separate' (each state fitted to its own).");
  args.AddOption(&settings.decay_switch, "-kl-switch", "--kl-switch",
                 "2.5D: k L above which Robin coefficients use the source and its image only.");
  args.AddOption(&settings.surface_z, "-surface", "--surface-z", "Image plane of the geometric factor (default: background top).");
  args.AddOption(&settings.ip, "-ip", "--ip", "Instantaneous-conductivity pass: 'auto', 'yes' or 'no'.");
  args.AddOption(&settings.ip_coefficients, "-ip-coefficients", "--ip-coefficients",
                 "Robin coefficients of the instantaneous pass: 'own' (instantaneous background) or 'dc' (DC background).");
  args.AddOption(&settings.robin_table, "-robin-table", "--robin-table",
                 "Per-electrode layered values from tables: 'auto' (when cheaper), 'always' or 'never'.");
  args.AddOption(&settings.write_potentials, "-pot", "--write-potentials", "-no-pot", "--no-potentials",
                 "Write the unit-current potentials at the potential electrodes to <prefix>.pot.");
  args.AddOption(&settings.verbose, "-v", "--verbose", "-q", "--quiet", "Log every solve.");
  args.Parse();
  if (!args.Good()) {
    if (rank() == 0) {
      args.PrintUsage(mfem::out);
    }
    return 1;
  }
  log_level() = settings.verbose ? 2 : 1;
  log(1, "dcipfwd {}", DCIPFWD_VERSION);
  if (rank() == 0) {
    args.PrintOptions(mfem::out);
  }

  const std::string boundary = settings.boundary, solver = settings.solver, preconditioner = settings.preconditioner;
  const std::string reference = settings.reference, robin_table = settings.robin_table, rule = settings.wavenumber_rule;
  const std::string ip_coefficients = settings.ip_coefficients;
  MFEM_VERIFY(ip_coefficients == "own" || ip_coefficients == "dc", "Unknown -ip-coefficients value.");
  options.dc_coefficients = ip_coefficients == "dc";
  MFEM_VERIFY(boundary == "layered" || boundary == "dey-morrison", "Unknown boundary condition.");
  MFEM_VERIFY(solver == "cg" || solver == "mumps", "Unknown linear solver.");
  MFEM_VERIFY(preconditioner == "amg" || preconditioner == "lor" || preconditioner == "direct", "Unknown preconditioner.");
  MFEM_VERIFY(robin_table == "auto" || robin_table == "always" || robin_table == "never", "Unknown -robin-table value.");
  MFEM_VERIFY(rule == "log" || rule == "layered" || rule == "layered-joint" || rule == "layered-separate", "Unknown -kq value.");
  options.boundary_condition = boundary == "layered" ? DCBoundaryCondition::Layered : DCBoundaryCondition::DeyMorrison;
  options.solver = solver == "cg" ? DCSolverType::CG : DCSolverType::Mumps;
  options.preconditioner = preconditioner == "amg"   ? DCPreconditionerType::AMG
                           : preconditioner == "lor" ? DCPreconditionerType::LOR
                                                     : DCPreconditionerType::Direct;
  options.robin_table = robin_table == "auto"     ? DCRobinTable::Auto
                        : robin_table == "always" ? DCRobinTable::Always
                                                  : DCRobinTable::Never;

  const Model model(settings.mesh_file, settings.property_file);
  const Survey survey(settings.survey_file);
  const std::string background_name = settings.background;
  const std::vector<BackgroundLayer> background =
      background_name == "corner" ? model.corner_background() : read_background(background_name);
  if (std::isnan(settings.surface_z)) {
    settings.surface_z = background.front().top;
  }
  log(1, "Background ({} layers):", background.size());
  for (const BackgroundLayer &layer : background) {
    log(1, "  top {: .6g} m, rho {:.6g} ohm-m, eta {:.4g}", layer.top, 1.0 / layer.sigma, layer.eta);
  }

  const std::string ip = settings.ip;
  const bool chargeable = ip == "yes" || (ip == "auto" && model.has_chargeability());
  DCSolverOptions instantaneous_options = options;
  RuleReport rules[2]; // DC and instantaneous
  double rule_time = 0.0; // 2.5D: weight fits and a-priori errors, before the passes
  if (model.dim() == 2) {
    const double rule_start = MPI_Wtime();
    const auto [minimum, maximum] = survey.transverse_distance_range();
    MFEM_VERIFY(minimum > 0.0, "2.5D measurements need distinct current and potential electrodes.");
    if (options.kx_min == 0.0) {
      options.kx_min = 1.0e-3 / maximum;
    }
    if (options.kx_max == 0.0) {
      options.kx_max = 15.0 / minimum;
    }
    // The 2.5D potentials are those of the plane of the sources, so every electrode must lie on one x.
    double strike_min = survey.electrodes().front()[0], strike_max = strike_min;
    double depth_min = survey.electrodes().front()[2], depth_max = depth_min;
    for (const mfem::Vector &electrode : survey.electrodes()) {
      strike_min = std::min(strike_min, electrode[0]);
      strike_max = std::max(strike_max, electrode[0]);
      depth_min = std::min(depth_min, electrode[2]);
      depth_max = std::max(depth_max, electrode[2]);
    }
    MFEM_VERIFY(strike_max - strike_min <= 1.0e-9 * std::max(1.0, maximum),
                "2.5D modelling requires all electrodes at the same x.");
    log(1, "2.5D wavenumbers: k_min {:.4e} 1/m, k_max {:.4e} 1/m, {} points per decade", options.kx_min, options.kx_max,
        options.kx_points_per_decade);

    // A-priori errors against the layered background of each state and the fitted rules; all assume one electrode depth.
    const std::vector<WavenumberNode> log_rule =
        make_log_wavenumber_quadrature(options.kx_min, options.kx_max, options.kx_points_per_decade);
    rules[0].nodes = rules[1].nodes = log_rule;
    if (depth_max - depth_min <= 1.0e-9 * std::max(1.0, maximum)) {
      const WavenumberReference references[2] = { layered_wavenumber_reference(background, depth_min, false),
                                                  layered_wavenumber_reference(background, depth_min, true) };
      const std::vector<double> offsets =
          logarithmic_offsets(minimum, maximum, std::max(60, 4 * static_cast<int>(log_rule.size())));
      if (rule == "layered" || rule == "layered-separate") {
        rules[0].nodes = rules[1].nodes = fit_wavenumber_weights(log_rule, { references[0] }, offsets);
      }
      if (rule == "layered-separate") {
        rules[1].nodes = fit_wavenumber_weights(log_rule, { references[1] }, offsets);
      }
      if (rule == "layered-joint") {
        rules[0].nodes = rules[1].nodes = fit_wavenumber_weights(log_rule, { references[0], references[1] }, offsets);
      }
      for (int state = 0; state < 2; ++state) {
        rules[state].error = wavenumber_rule_error(rules[state].nodes, references[state], offsets);
        rules[state].error_log = wavenumber_rule_error(log_rule, references[state], offsets);
      }
      log(1, "2.5D rule '{}' with {} wavenumbers: a-priori error against the layered background {:.3e} (log rule {:.3e}), "
             "instantaneous {:.3e} ({:.3e})",
          rule, log_rule.size(), rules[0].error, rules[0].error_log, rules[1].error, rules[1].error_log);
    } else {
      MFEM_VERIFY(rule == "log", "Fitted wavenumber rules require all electrodes at one depth.");
    }
    options.wavenumbers = rules[0].nodes;
    instantaneous_options.wavenumbers = rules[1].nodes;
    instantaneous_options.kx_min = options.kx_min;
    instantaneous_options.kx_max = options.kx_max;
    rule_time = MPI_Wtime() - rule_start;
    MPI_Allreduce(MPI_IN_PLACE, &rule_time, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
  }

  std::vector<PassReport> reports(chargeable ? 2 : 1);
  reports[0].name = "dc";
  const PotentialTable dc = forward_pass(model, survey, options, settings, background, false, reports[0]);
  PotentialTable instantaneous;
  if (chargeable) {
    reports[1].name = "chargeable";
    instantaneous = forward_pass(model, survey, instantaneous_options, settings, background, true, reports[1]);
  }

  double memory = peak_memory_mib(), memory_sum = memory;
  MPI_Allreduce(MPI_IN_PLACE, &memory, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
  MPI_Allreduce(MPI_IN_PLACE, &memory_sum, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

  bool written = true;
  if (rank() == 0) {
    const std::string prefix = settings.output_prefix;
    std::ofstream data(prefix + ".dat");
    data << fmt::format("# dcipfwd {}: mesh {}, properties {}, survey {}\n", DCIPFWD_VERSION, settings.mesh_file,
                        settings.property_file, settings.survey_file);
    data << fmt::format("# boundary {}, reference {}, order {}, solver {}, pc {}\n", boundary, reference, options.order, solver,
                        preconditioner);
    data << "# a b m n dV(V/A) K(m) rho_a(ohm-m)" << (chargeable ? " dV_inst(V/A) m_a(V/V)" : "") << "\n";
    for (const Measurement &row : survey.measurements()) {
      const double voltage = potential_difference(row, [&](int source, int receiver) { return dc.at(source).at(receiver); });
      const double factor = survey.geometric_factor(row, settings.surface_z);
      data << fmt::format("{} {} {} {} {: .12e} {: .12e} {: .12e}", row.a, row.b, row.m, row.n, voltage, factor,
                          factor * voltage);
      if (chargeable) {
        const double instant =
            potential_difference(row, [&](int source, int receiver) { return instantaneous.at(source).at(receiver); });
        data << fmt::format(" {: .12e} {: .12e}", instant, 1.0 - instant / voltage);
      }
      data << "\n";
    }
    data.close();
    written = !data.fail();

    if (settings.write_potentials) {
      std::ofstream potentials(prefix + ".pot");
      potentials << "# current potential V(V/A)" << (chargeable ? " V_inst(V/A)" : "") << ", electrode indices of the survey\n";
      for (const auto &[source, values] : dc) {
        for (const auto &[receiver, value] : values) {
          if (receiver == source) {
            continue;
          }
          potentials << fmt::format("{} {} {: .15e}", source, receiver, value);
          if (chargeable) {
            potentials << fmt::format(" {: .15e}", instantaneous.at(source).at(receiver));
          }
          potentials << "\n";
        }
      }
      potentials.close();
      written = written && !potentials.fail();
    }

    std::ofstream json(prefix + ".json");
    json << "{\n";
    json << fmt::format(R"(  "version": "{}", "mesh": "{}", "properties": "{}", "survey": "{}", "background_file": "{}",)",
                        DCIPFWD_VERSION, settings.mesh_file, settings.property_file, settings.survey_file, background_name)
         << "\n";
    json << fmt::format(R"(  "dim": {}, "boundary": "{}", "reference": "{}", "order": {}, "solver": "{}", "preconditioner": "{}",)",
                        model.dim(), boundary, reference, options.order, solver, preconditioner)
         << "\n";
    json << fmt::format(R"(  "rtol": {}, "max_iterations": {}, "amg_strength_threshold": {}, "mumps_blr_tolerance": {},)",
                        options.rtol, options.max_iter, options.amg_strength_threshold, options.mumps_blr_tolerance)
         << "\n";
    json << fmt::format(R"(  "uniform_refinements": {}, "tx_refinements": {}, "rx_refinements": {}, "robin_table": "{}", )"
                        R"("ip_coefficients": "{}",)",
                        options.uniform_refinements, settings.tx_refinements, settings.rx_refinements, robin_table,
                        ip_coefficients)
         << "\n";
    if (model.dim() == 2) {
      json << fmt::format(R"(  "kx_min": {}, "kx_max": {}, "kx_points_per_decade": {}, "wavenumber_rule": "{}", "kl_switch": {}, )"
                          R"("rule_time": {:.6g},)",
                          options.kx_min, options.kx_max, options.kx_points_per_decade, rule, settings.decay_switch, rule_time)
           << "\n";
      json << rule_json("rule_dc", rules[0]) << "\n" << rule_json("rule_instantaneous", rules[1]) << "\n";
    }
    int ranks = 0;
    MPI_Comm_size(MPI_COMM_WORLD, &ranks);
    json << fmt::format(R"(  "ranks": {}, "peak_memory_mib_max": {:.1f}, "peak_memory_mib_sum": {:.1f},)", ranks, memory,
                        memory_sum)
         << "\n";
    json << "  \"background\": [";
    for (std::size_t layer = 0; layer < background.size(); ++layer) {
      json << fmt::format(R"({}{{"top": {}, "rho": {}, "eta": {}}})", layer ? ", " : "", background[layer].top,
                          1.0 / background[layer].sigma, background[layer].eta);
    }
    json << "],\n  \"passes\": [\n";
    for (std::size_t pass = 0; pass < reports.size(); ++pass) {
      json << fmt::format(R"(    {{"name": "{}", "wall_time": {:.6g}, "solvers": [)", reports[pass].name, reports[pass].wall_time);
      for (std::size_t index = 0; index < reports[pass].stats.size(); ++index) {
        json << (index ? ", " : "") << stats_json(reports[pass].stats[index]);
      }
      json << "]}" << (pass + 1 < reports.size() ? "," : "") << "\n";
    }
    json << "  ]\n}\n";
    json.close();
    written = written && !json.fail();
    if (written) {
      log(1, "Wrote {}.dat and {}.json; peak memory {:.1f} MiB per rank (max), {:.1f} MiB in total.", prefix, prefix, memory,
          memory_sum);
    }
  }
  MPI_Bcast(&written, 1, MPI_C_BOOL, 0, MPI_COMM_WORLD);
  if (!written) {
    log(0, "Error: cannot write the outputs with prefix {}.", settings.output_prefix);
    return 1;
  }
  return 0;
}
