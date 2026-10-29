#include "wavenumber_rule.h"

#include "utils.h"

#include <mfem.hpp>

#include <algorithm>
#include <cmath>
#include <memory>

namespace dcip {

namespace {

/// Solve the least-squares problem min |A x - b| for a dense row-major matrix with full column rank (Householder QR).
std::vector<double> least_squares(std::vector<double> a, std::vector<double> b, int rows, int columns) {
  const auto at = [&](int row, int column) -> double & { return a[static_cast<std::size_t>(row) * columns + column]; };
  for (int column = 0; column < columns; ++column) {
    double norm = 0.0;
    for (int row = column; row < rows; ++row) {
      norm += at(row, column) * at(row, column);
    }
    norm = std::sqrt(norm);
    MFEM_VERIFY(norm > 0.0, "The wavenumber fit is rank deficient.");
    const double alpha = at(column, column) > 0.0 ? -norm : norm;
    std::vector<double> v(rows - column);
    for (int row = column; row < rows; ++row) {
      v[row - column] = at(row, column);
    }
    v[0] -= alpha;
    double v_norm = 0.0;
    for (double entry : v) {
      v_norm += entry * entry;
    }
    if (v_norm == 0.0) {
      continue;
    }
    for (int target = column; target < columns; ++target) {
      double dot = 0.0;
      for (int row = column; row < rows; ++row) {
        dot += v[row - column] * at(row, target);
      }
      for (int row = column; row < rows; ++row) {
        at(row, target) -= 2.0 * dot / v_norm * v[row - column];
      }
    }
    double dot = 0.0;
    for (int row = column; row < rows; ++row) {
      dot += v[row - column] * b[row];
    }
    for (int row = column; row < rows; ++row) {
      b[row] -= 2.0 * dot / v_norm * v[row - column];
    }
  }
  std::vector<double> x(columns);
  for (int row = columns - 1; row >= 0; --row) {
    double sum = b[row];
    for (int column = row + 1; column < columns; ++column) {
      sum -= at(row, column) * x[column];
    }
    x[row] = sum / at(row, row);
  }
  return x;
}

} // namespace

WavenumberReference layered_wavenumber_reference(const std::vector<BackgroundLayer> &background, double depth,
                                                 bool chargeable) {
  MFEM_VERIFY(!background.empty(), "A wavenumber reference needs a background layer.");
  const double surface = background.front().top;
  std::vector<ConductiveLayer> layers;
  for (const BackgroundLayer &layer : background) {
    layers.push_back({ layer.top - surface, chargeable ? chargeable_conductivity(layer.sigma, layer.eta) : layer.sigma });
  }
  const double source_depth = std::max(0.0, depth - surface);
  auto layered = std::make_shared<LayeredPotential>(layers, source_depth);
  WavenumberReference reference;
  reference.potential = [layered, source_depth](double r) {
    double d_dr = 0.0, d_dz = 0.0;
    return layered->potential(r, source_depth, d_dr, d_dz);
  };
  // The layered transform holds half of the full cosine transform.
  reference.transformed = [layered, source_depth](double k, double r) {
    if (k * r > LayeredPotential::kDecaySwitch) {
      return 0.0;
    }
    double d_dr = 0.0, d_dz = 0.0;
    return 2.0 * layered->transformed_potential(k, r, source_depth, d_dr, d_dz);
  };
  return reference;
}

double wavenumber_rule_error(const std::vector<WavenumberNode> &nodes, const WavenumberReference &reference,
                             const std::vector<double> &offsets) {
  double worst = 0.0;
  for (double r : offsets) {
    double sum = 0.0;
    for (const WavenumberNode &node : nodes) {
      sum += node.weight * reference.transformed(node.k, r);
    }
    worst = std::max(worst, std::abs(sum / PI / reference.potential(r) - 1.0));
  }
  return worst;
}

std::vector<WavenumberNode> fit_wavenumber_weights(const std::vector<WavenumberNode> &nodes,
                                                   const std::vector<WavenumberReference> &references,
                                                   const std::vector<double> &offsets, double regularization) {
  const int n = static_cast<int>(nodes.size());
  const int m = static_cast<int>(offsets.size() * references.size());
  MFEM_VERIFY(!references.empty() && m >= n, "Fitting wavenumber weights needs at least as many offsets as nodes.");
  // Rows: relative residuals of every reference at every offset, then the regularization of every weight.
  std::vector<double> a(static_cast<std::size_t>(m + n) * n, 0.0), b(m + n, 0.0);
  int row = 0;
  for (const WavenumberReference &reference : references) {
    for (double r : offsets) {
      const double potential = reference.potential(r);
      for (int column = 0; column < n; ++column) {
        a[static_cast<std::size_t>(row) * n + column] = reference.transformed(nodes[column].k, r) / (PI * potential);
      }
      b[row++] = 1.0;
    }
  }
  for (int column = 0; column < n; ++column) {
    a[static_cast<std::size_t>(m + column) * n + column] = regularization / nodes[column].weight;
    b[m + column] = regularization;
  }
  const std::vector<double> weights = least_squares(a, b, m + n, n);
  std::vector<WavenumberNode> result = nodes;
  for (int column = 0; column < n; ++column) {
    result[column].weight = weights[column];
  }
  return result;
}

std::vector<double> logarithmic_offsets(double r_min, double r_max, int n) {
  MFEM_VERIFY(r_min > 0.0 && r_max >= r_min && n >= 2, "Logarithmic offsets need 0 < r_min <= r_max and n >= 2.");
  std::vector<double> offsets(n);
  for (int index = 0; index < n; ++index) {
    offsets[index] = r_min * std::pow(r_max / r_min, static_cast<double>(index) / (n - 1));
  }
  return offsets;
}

} // namespace dcip
