#include "dc_boundary.h"

#include "integral_transforms.h"
#include "utils.h"

#include <fmt/format.h>

#include <mfem.hpp>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <map>

namespace dcip {

namespace {

std::uint64_t bits(double value) {
  std::uint64_t result = 0;
  std::memcpy(&result, &value, sizeof(result));
  return result;
}

/// Nodes and weights of a quadrature rule.
struct QuadratureRule {
  std::vector<double> nodes;
  std::vector<double> weights;
};

/// Return the Gauss-Legendre rule of an order on [0, 1].
const QuadratureRule &unit_gauss_rule(int order) {
  static std::map<int, QuadratureRule> rules;
  auto found = rules.find(order);
  if (found == rules.end()) {
    mfem::IntegrationRule points(order);
    mfem::QuadratureFunctions1D::GaussLegendre(order, &points);
    QuadratureRule rule;
    for (int point = 0; point < points.GetNPoints(); ++point) {
      rule.nodes.push_back(points.IntPoint(point).x);
      rule.weights.push_back(points.IntPoint(point).weight);
    }
    found = rules.emplace(order, std::move(rule)).first;
  }
  return found->second;
}

/// Append a Gauss-Legendre panel of an order on [lower, upper].
void append_panel(QuadratureRule &rule, double lower, double upper, int order) {
  const QuadratureRule &unit = unit_gauss_rule(order);
  for (std::size_t point = 0; point < unit.nodes.size(); ++point) {
    rule.nodes.push_back(lower + (upper - lower) * unit.nodes[point]);
    rule.weights.push_back((upper - lower) * unit.weights[point]);
  }
}

/// Append a panel on [0, low] and panels of at most a decade from low up to upper.
void append_logarithmic_panels(QuadratureRule &rule, double low, double upper, int order = 16) {
  append_panel(rule, 0.0, low, order);
  for (double lower = low; lower < upper * (1.0 - 1.0e-12); lower *= 10.0) {
    append_panel(rule, lower, std::min(10.0 * lower, upper), order);
  }
}

/// Return a rule on [0, 60 / path] for kernels decaying as exp(-lambda path) with features down to low.
QuadratureRule decay_rule(double path, double low) {
  QuadratureRule rule;
  const double start = 1.0e-2 / path;
  if (low < start) {
    append_logarithmic_panels(rule, low, start);
  } else {
    append_panel(rule, 0.0, start, 16);
  }
  append_panel(rule, 1.0e-2 / path, 1.0e-1 / path, 16);
  append_panel(rule, 1.0e-1 / path, 1.0 / path, 24);
  append_panel(rule, 1.0 / path, 10.0 / path, 48);
  append_panel(rule, 10.0 / path, 30.0 / path, 48);
  append_panel(rule, 30.0 / path, 60.0 / path, 16);
  return rule;
}

/// Return the weight of the part above split of a kernel separated smoothly at split.
double upper_window(double x, double split) { return -std::expm1(-std::pow(x / split, 4)); }

/// Nodes x, weights and trigonometric factors of the Ooura-Mori formula for the integrals of f(x) cos x or f(x) sin x.
struct FourierRule {
  std::vector<double> nodes;
  std::vector<double> weights; // Weight times the cosine or sine at the node
};

/// Return the double exponential rule of Ooura and Mori (1999) for Fourier-type integrals over [0, infinity).
FourierRule make_fourier_rule(bool cosine) {
  constexpr int kM = 64;
  constexpr double kBeta = 0.25;
  const double h = PI / kM;
  const double alpha = kBeta / std::sqrt(1.0 + kM * std::log1p(kM) / (4.0 * PI));
  const double c1 = 2.0 + alpha + kBeta;
  FourierRule rule;
  for (int n = -6 * kM; n <= 6 * kM; ++n) {
    const double t = cosine ? (n - 0.5) * h : n * h;
    double phi = 0.0, d_phi = 0.0;
    if (std::abs(t) < 1.0e-8) {
      const double c2 = 0.5 * (kBeta - alpha);
      phi = 1.0 / c1;
      d_phi = (0.5 * c1 * c1 - c2) / (c1 * c1);
    } else {
      const double u = 2.0 * t + alpha * (1.0 - std::exp(-t)) + kBeta * (std::exp(t) - 1.0);
      const double d_u = 2.0 + alpha * std::exp(-t) + kBeta * std::exp(t);
      const double denominator = -std::expm1(-u);
      phi = t / denominator;
      d_phi = (denominator - t * std::exp(-u) * d_u) / (denominator * denominator);
    }
    const double weight = kM * h * d_phi * (cosine ? std::cos(kM * phi) : std::sin(kM * phi));
    if (std::isfinite(phi) && std::isfinite(weight) && phi > 0.0 && std::abs(weight) > 1.0e-30) {
      rule.nodes.push_back(kM * phi);
      rule.weights.push_back(weight);
    }
  }
  return rule;
}

const FourierRule &cosine_rule() {
  static const FourierRule rule = make_fourier_rule(true);
  return rule;
}

const FourierRule &sine_rule() {
  static const FourierRule rule = make_fourier_rule(false);
  return rule;
}

} // namespace

LayeredPotential::LayeredPotential(const std::vector<ConductiveLayer> &layers, double source_depth)
    : source_depth_(source_depth) {
  MFEM_VERIFY(!layers.empty() && layers.front().top == 0.0, "A layered earth starts with a layer at the surface.");
  MFEM_VERIFY(std::isfinite(source_depth) && source_depth >= 0.0, "The source must lie below the surface.");
  for (std::size_t layer = 0; layer < layers.size(); ++layer) {
    MFEM_VERIFY(std::isfinite(layers[layer].sigma) && layers[layer].sigma > 0.0, "Layer conductivities must be positive.");
    MFEM_VERIFY(layer == 0 || (std::isfinite(layers[layer].top) && layers[layer].top > layers[layer - 1].top),
                "Layer tops must increase with depth.");
    tops_.push_back(layers[layer].top);
    sigma_.push_back(layers[layer].sigma);
  }
  tops_.push_back(std::numeric_limits<double>::infinity());
  source_layer_ = layer_of(source_depth);
  double conductance = 0.0;
  for (std::size_t layer = 1; layer < sigma_.size(); ++layer) {
    conductance += sigma_[layer - 1] * (tops_[layer] - tops_[layer - 1]);
    low_scale_ = std::min(low_scale_, sigma_[layer] / conductance);
  }
}

int LayeredPotential::layer_of(double z) const {
  int layer = 0;
  while (layer + 1 < static_cast<int>(sigma_.size()) && z > tops_[layer + 1]) {
    ++layer;
  }
  return layer;
}

LayeredPotential::Kernel LayeredPotential::secondary_kernel(double lambda, double z, int receiver_layer) const {
  const int n = static_cast<int>(sigma_.size());
  const int s = source_layer_, r = receiver_layer;
  const double d = source_depth_;

  // Reflection coefficients of the decaying and the surface solutions at the bottom and top of every layer.
  std::vector<double> e(n, 0.0), down(n, 0.0), up(n, 1.0);
  for (int j = 0; j + 1 < n; ++j) {
    e[j] = std::exp(-2.0 * lambda * (tops_[j + 1] - tops_[j]));
  }
  for (int j = n - 2; j >= 0; --j) {
    const double t = (1.0 - down[j + 1] * e[j + 1]) / (1.0 + down[j + 1] * e[j + 1]);
    down[j] = (sigma_[j] - sigma_[j + 1] * t) / (sigma_[j] + sigma_[j + 1] * t);
  }
  for (int j = 1; j < n; ++j) {
    const double t = (1.0 - up[j - 1] * e[j - 1]) / (1.0 + up[j - 1] * e[j - 1]);
    up[j] = (sigma_[j] - sigma_[j - 1] * t) / (sigma_[j] + sigma_[j - 1] * t);
  }

  const bool bounded = s + 1 < n;
  const double a = bounded ? std::exp(-2.0 * lambda * (tops_[s + 1] - d)) : 0.0;
  const double b = std::exp(-2.0 * lambda * (d - tops_[s]));
  const double admittance =
      sigma_[s] * lambda * ((1.0 - up[s] * b) / (1.0 + up[s] * b) + (1.0 - down[s] * a) / (1.0 + down[s] * a));

  // Ratio of the solution at the receiver to its value at the source, and its z derivative.
  double ratio = 0.0, d_ratio = 0.0;
  if (r == s && z >= d) {
    const double direct = std::exp(-lambda * (z - d));
    const double reflected = bounded ? down[s] * std::exp(-lambda * (2.0 * tops_[s + 1] - z - d)) : 0.0;
    ratio = (direct + reflected) / (1.0 + down[s] * a);
    d_ratio = lambda * (reflected - direct) / (1.0 + down[s] * a);
  } else if (r == s) {
    const double direct = std::exp(-lambda * (d - z));
    const double reflected = up[s] * std::exp(-lambda * (z + d - 2.0 * tops_[s]));
    ratio = (direct + reflected) / (1.0 + up[s] * b);
    d_ratio = lambda * (direct - reflected) / (1.0 + up[s] * b);
  } else if (r > s) {
    double factor = std::exp(-lambda * (tops_[s + 1] - d)) * (1.0 + down[s]) / (1.0 + down[s] * a);
    for (int j = s + 1; j < r; ++j) {
      factor *= std::exp(-lambda * (tops_[j + 1] - tops_[j])) * (1.0 + down[j]) / (1.0 + down[j] * e[j]);
    }
    const double direct = std::exp(-lambda * (z - tops_[r]));
    const double reflected = r + 1 < n ? down[r] * std::exp(-lambda * (2.0 * tops_[r + 1] - tops_[r] - z)) : 0.0;
    ratio = factor * (direct + reflected) / (1.0 + down[r] * e[r]);
    d_ratio = factor * lambda * (reflected - direct) / (1.0 + down[r] * e[r]);
  } else {
    double factor = std::exp(-lambda * (d - tops_[s])) * (1.0 + up[s]) / (1.0 + up[s] * b);
    for (int j = s - 1; j > r; --j) {
      factor *= std::exp(-lambda * (tops_[j + 1] - tops_[j])) * (1.0 + up[j]) / (1.0 + up[j] * e[j]);
    }
    const double direct = std::exp(lambda * (z - tops_[r + 1]));
    const double reflected = up[r] * std::exp(-lambda * (z + tops_[r + 1] - 2.0 * tops_[r]));
    ratio = factor * (direct + reflected) / (1.0 + up[r] * e[r]);
    d_ratio = factor * lambda * (direct - reflected) / (1.0 + up[r] * e[r]);
  }

  Kernel kernel{ ratio / admittance, d_ratio / admittance };
  if (r == s) {
    // Remove the direct field and, below the surface layer's source, its surface image.
    const double direct = std::exp(-lambda * std::abs(z - d)) / (2.0 * sigma_[s] * lambda);
    kernel.value -= direct;
    kernel.d_dz += (z >= d ? 1.0 : -1.0) * lambda * direct;
    if (s == 0) {
      const double image = std::exp(-lambda * (z + d)) / (2.0 * sigma_[0] * lambda);
      kernel.value -= image;
      kernel.d_dz += lambda * image;
    }
  }
  return kernel;
}

double LayeredPotential::secondary_path(double z, int receiver_layer) const {
  if (receiver_layer != source_layer_) {
    return std::max(std::abs(z - source_depth_), 1.0e-12);
  }
  double path = source_layer_ > 0 ? z + source_depth_ : std::numeric_limits<double>::infinity();
  for (std::size_t interface = 1; interface + 1 < tops_.size(); ++interface) {
    path = std::min(path, std::abs(z - tops_[interface]) + std::abs(source_depth_ - tops_[interface]));
  }
  return std::isfinite(path) ? std::max(path, 1.0e-12) : path;
}

double LayeredPotential::potential(double r, double z, double &d_dr, double &d_dz) const {
  MFEM_VERIFY(r >= 0.0 && z >= 0.0, "Potentials are evaluated at a non-negative offset below the surface.");
  const int receiver_layer = layer_of(z);
  const int s = source_layer_;
  const double d = source_depth_;
  double value = 0.0;
  d_dr = 0.0;
  d_dz = 0.0;

  const auto add_point_source = [&](double scale, double vertical) {
    const double distance = std::hypot(r, vertical);
    value += scale / distance;
    d_dr -= scale * r / (distance * distance * distance);
    return scale * vertical / (distance * distance * distance);
  };
  if (receiver_layer == s) {
    d_dz -= add_point_source(1.0 / (4.0 * PI * sigma_[s]), z - d);
    if (s == 0) {
      d_dz -= add_point_source(1.0 / (4.0 * PI * sigma_[0]), z + d);
    }
  }

  const double path = secondary_path(z, receiver_layer);
  if (!std::isfinite(path)) {
    return value;
  }

  double sum_j0 = 0.0, sum_j1 = 0.0, sum_z = 0.0;
  const auto add_quadrature = [&](const QuadratureRule &rule, double split) {
    for (std::size_t node = 0; node < rule.nodes.size(); ++node) {
      const double lambda = rule.nodes[node];
      const double weight = rule.weights[node] * (split > 0.0 ? 1.0 - upper_window(lambda, split) : 1.0);
      const Kernel kernel = secondary_kernel(lambda, z, receiver_layer);
      const double j0 = bessel_j0(lambda * r), j1 = bessel_j1(lambda * r);
      sum_j0 += weight * kernel.value * lambda * j0;
      sum_j1 += weight * kernel.value * lambda * lambda * j1;
      sum_z += weight * kernel.d_dz * lambda * j0;
    }
  };
  if (r < path) {
    add_quadrature(decay_rule(path, 1.0e-2 * low_scale_), 0.0);
  } else {
    // The filter integrates the kernel above a hundred times its lowest abscissa and decade panels the part below.
    const double split = 100.0 * key_201_base.front() / r;
    for (std::size_t node = 0; node < key_201_base.size(); ++node) {
      const double lambda = key_201_base[node] / r;
      const double window = upper_window(lambda, split);
      const Kernel kernel = secondary_kernel(lambda, z, receiver_layer);
      sum_j0 += key_201_j0[node] * window * kernel.value * lambda / r;
      sum_j1 += key_201_j1[node] * window * kernel.value * lambda * lambda / r;
      sum_z += key_201_j0[node] * window * kernel.d_dz * lambda / r;
    }
    QuadratureRule low;
    append_logarithmic_panels(low, 1.0e-2 * std::min(low_scale_, split), 3.0 * split);
    add_quadrature(low, split);
  }
  value += sum_j0 / (2.0 * PI);
  d_dr -= sum_j1 / (2.0 * PI);
  d_dz += sum_z / (2.0 * PI);
  return value;
}

double LayeredPotential::transformed_potential(double k, double y, double z, double &d_dy, double &d_dz) const {
  MFEM_VERIFY(k > 0.0 && z >= 0.0, "Transformed potentials require a positive wavenumber below the surface.");
  const int receiver_layer = layer_of(z);
  const int s = source_layer_;
  const double d = source_depth_;
  MFEM_VERIFY(k * std::hypot(y, z - d) <= kLargestDecay,
              fmt::format("The transformed layered potential is evaluated only for k r <= {}.", kLargestDecay));
  double value = 0.0;
  d_dy = 0.0;
  d_dz = 0.0;

  const auto add_line_source = [&](double scale, double vertical) {
    const double distance = std::hypot(y, vertical);
    const double k1 = bessel_k1(k * distance);
    value += scale * bessel_k0(k * distance);
    d_dy -= scale * k * k1 * y / distance;
    return scale * k * k1 * vertical / distance;
  };
  if (receiver_layer == s) {
    d_dz -= add_line_source(1.0 / (4.0 * PI * sigma_[s]), z - d);
    if (s == 0) {
      d_dz -= add_line_source(1.0 / (4.0 * PI * sigma_[0]), z + d);
    }
  }

  const double path = secondary_path(z, receiver_layer);
  if (!std::isfinite(path)) {
    return value;
  }

  const auto kernel_at = [&](double t) { return secondary_kernel(std::hypot(k, t), z, receiver_layer); };

  double sum_cos = 0.0, sum_sin = 0.0, sum_z = 0.0;
  const double offset = std::abs(y);
  // The kernel changes shape where t reaches k and, below k, at the low-wavenumber scale of the layers.
  const double feature = std::min(low_scale_, k);
  const auto add_quadrature = [&](const QuadratureRule &rule, double split) {
    for (std::size_t node = 0; node < rule.nodes.size(); ++node) {
      const double t = rule.nodes[node];
      const double weight = rule.weights[node] * (split > 0.0 ? 1.0 - upper_window(t, split) : 1.0);
      const Kernel kernel = kernel_at(t);
      sum_cos += weight * kernel.value * std::cos(t * offset);
      sum_sin += weight * kernel.value * t * std::sin(t * offset);
      sum_z += weight * kernel.d_dz * std::cos(t * offset);
    }
  };
  if (offset < path) {
    add_quadrature(decay_rule(path, 1.0e-2 * feature), 0.0);
  } else {
    // The double exponential rule integrates the kernel above t = 10 / |y| and decade panels the part below.
    const double split = 10.0 / offset;
    const FourierRule &cosine = cosine_rule();
    for (std::size_t node = 0; node < cosine.nodes.size(); ++node) {
      const double t = cosine.nodes[node] / offset;
      const double weight = cosine.weights[node] * upper_window(t, split) / offset;
      const Kernel kernel = kernel_at(t);
      sum_cos += weight * kernel.value;
      sum_z += weight * kernel.d_dz;
    }
    const FourierRule &sine = sine_rule();
    for (std::size_t node = 0; node < sine.nodes.size(); ++node) {
      const double t = sine.nodes[node] / offset;
      sum_sin += sine.weights[node] * upper_window(t, split) * kernel_at(t).value * t / offset;
    }
    QuadratureRule low;
    append_logarithmic_panels(low, 1.0e-2 * std::min(feature, split), 3.0 * split, 48);
    add_quadrature(low, split);
  }
  value += sum_cos / (2.0 * PI);
  d_dy -= (y < 0.0 ? -1.0 : 1.0) * sum_sin / (2.0 * PI);
  d_dz += sum_z / (2.0 * PI);
  return value;
}

double chargeable_conductivity(double sigma, double eta) { return sigma / (1.0 - eta); }

DCBoundary::DCBoundary(int dim, const std::vector<BackgroundLayer> &column, const mfem::Vector &reference, double decay_switch)
    : dim_(dim), reference_(project_xyz_to_model(reference, dim)), decay_switch_(decay_switch) {
  MFEM_VERIFY(decay_switch_ > 0.0 && decay_switch_ <= LayeredPotential::kLargestDecay,
              fmt::format("The decay switch must lie in (0, {}].", LayeredPotential::kLargestDecay));
  for (const BackgroundLayer &layer : column) {
    MFEM_VERIFY(sigma_.empty() || layer.top > tops_.back(), "Background layer tops must increase with depth.");
    if (sigma_.empty() || layer.sigma != sigma_.back() || layer.eta != eta_.back()) {
      tops_.push_back(layer.top);
      sigma_.push_back(layer.sigma);
      eta_.push_back(layer.eta);
    }
  }
  MFEM_VERIFY(!tops_.empty(), "The background column has no ground layer.");
  surface_ = tops_.front();
}

std::vector<BackgroundLayer> DCBoundary::layers() const {
  std::vector<BackgroundLayer> result;
  for (std::size_t layer = 0; layer < tops_.size(); ++layer) {
    result.push_back({ tops_[layer], sigma_[layer], eta_[layer] });
  }
  return result;
}

const LayeredPotential &DCBoundary::potential(bool chargeable) const {
  std::unique_ptr<LayeredPotential> &result = potentials_[chargeable ? 1 : 0];
  if (!result) {
    std::vector<ConductiveLayer> layers;
    for (std::size_t layer = 0; layer < tops_.size(); ++layer) {
      layers.push_back(
          { tops_[layer] - surface_, chargeable ? chargeable_conductivity(sigma_[layer], eta_[layer]) : sigma_[layer] });
    }
    result = std::make_unique<LayeredPotential>(layers, std::max(0.0, reference_[dim_ - 1] - surface_));
  }
  return *result;
}

double DCBoundary::potential_gradient(double r, double z, double k, bool chargeable, double &d_dr, double &d_dz) const {
  const LayeredPotential &layered = potential(chargeable);
  const double depth = std::max(0.0, z - surface_);
  const double source_depth = std::max(0.0, reference_[dim_ - 1] - surface_);
  if (dim_ == 3) {
    return layered.potential(r, depth, d_dr, d_dz);
  }
  if (resolved(r, z, k)) {
    return layered.transformed_potential(k, r, depth, d_dr, d_dz);
  }
  // Beyond the switch only the source and its image contribute; values are scaled by exp(k |direct|).
  const double direct = std::hypot(r, depth - source_depth), image = std::hypot(r, depth + source_depth);
  d_dr = -k * r * (bessel_k1_scaled(k * direct) / direct + std::exp(k * (direct - image)) * bessel_k1_scaled(k * image) / image);
  d_dz = -k * ((depth - source_depth) * bessel_k1_scaled(k * direct) / direct +
               std::exp(k * (direct - image)) * (depth + source_depth) * bessel_k1_scaled(k * image) / image);
  return bessel_k0_scaled(k * direct) + std::exp(k * (direct - image)) * bessel_k0_scaled(k * image);
}

double DCBoundary::log_scale(double r, double z, double k, bool chargeable) const {
  if (dim_ == 3) {
    return 0.0;
  }
  if (resolved(r, z, k)) {
    return std::log(2.0); // transformed_potential holds half of the full transform
  }
  // Full transform of the source and its image, (K0(k d) + K0(k d')) / (2 pi sigma), in the layer of the source.
  const double direct = std::hypot(r, vertical_separation(z));
  return -k * direct - std::log(2.0 * PI * potential(chargeable).source_conductivity());
}

double DCBoundary::vertical_separation(double z) const {
  return std::abs(std::max(0.0, z - surface_) - std::max(0.0, reference_[dim_ - 1] - surface_));
}

bool DCBoundary::resolved(double r, double z, double k) const {
  return dim_ == 3 || k * std::hypot(r, vertical_separation(z)) <= decay_switch_;
}

double DCBoundary::robin_ratio(const mfem::Vector &position, const mfem::Vector &unit_normal, double k, bool chargeable,
                               double *log_value) const {
  double value = 0.0, normal_derivative = 0.0, d_dr = 0.0, d_dz = 0.0, r = 0.0;
  if (dim_ == 3) {
    const double dx = position[0] - reference_[0], dy = position[1] - reference_[1];
    r = std::hypot(dx, dy);
    value = potential_gradient(r, position[2], 0.0, chargeable, d_dr, d_dz);
    normal_derivative = d_dz * unit_normal[2] + (r > 0.0 ? d_dr * (dx * unit_normal[0] + dy * unit_normal[1]) / r : 0.0);
  } else {
    const double y = position[0] - reference_[0];
    r = std::abs(y);
    value = potential_gradient(r, position[1], k, chargeable, d_dr, d_dz);
    normal_derivative = (y < 0.0 ? -d_dr : d_dr) * unit_normal[0] + d_dz * unit_normal[1];
  }
  MFEM_VERIFY(value > 0.0 && std::isfinite(value), fmt::format("The layered boundary potential {} is not positive.", value));
  if (log_value) {
    *log_value = std::log(value) + log_scale(r, position[dim_ - 1], k, chargeable);
  }
  return -normal_derivative / value;
}

void RobinTable::grid(const DCBoundary &boundary, const DepthRange &range, double step, double &a, double &s_min,
                      int &nodes) {
  a = std::max(boundary.vertical_separation(range.z), 1.0e-6 * std::max(range.r_max, 1.0));
  s_min = std::asinh(std::max(range.r_min, 0.0) / a);
  const double s_max = std::asinh(range.r_max / a);
  nodes = std::max(4, static_cast<int>(std::ceil((s_max - s_min) / step)) + 1);
}

long long RobinTable::cost(const DCBoundary &boundary, const std::vector<DepthRange> &ranges, double step) {
  long long total = 0;
  for (const DepthRange &range : ranges) {
    double a = 0.0, s_min = 0.0;
    int nodes = 0;
    grid(boundary, range, step, a, s_min, nodes);
    total += nodes;
  }
  return total;
}

RobinTable::RobinTable(const DCBoundary &boundary, double k, bool chargeable, const std::vector<DepthRange> &ranges,
                       MPI_Comm comm, bool with_values, double step)
    : boundary_(boundary), k_(k), chargeable_(chargeable), with_values_(with_values), step_(step) {
  const int n_columns = static_cast<int>(ranges.size());
  std::vector<Column> columns(n_columns);
  std::vector<int> sizes(n_columns);
  long long total = 0;
  for (int index = 0; index < n_columns; ++index) {
    grid(boundary, ranges[index], step, columns[index].a, columns[index].s_min, sizes[index]);
    columns[index].z = ranges[index].z;
    total += sizes[index];
  }

  // Contiguous blocks of columns with about equal numbers of nodes per rank.
  int n_ranks = 1, rank = 0;
  MPI_Comm_size(comm, &n_ranks);
  MPI_Comm_rank(comm, &rank);
  std::vector<int> first(n_ranks + 1, n_columns);
  first[0] = 0;
  long long cumulative = 0;
  int owner = 1;
  for (int index = 0; index < n_columns && owner < n_ranks; ++index) {
    cumulative += sizes[index];
    while (owner < n_ranks && cumulative * n_ranks >= total * owner) {
      first[owner++] = index + 1;
    }
  }

  // Three values per node: L dG/dr / G, L dG/dz / G and ln G.
  std::vector<double> local;
  for (int index = first[rank]; index < first[rank + 1]; ++index) {
    const Column &column = columns[index];
    for (int node = 0; node < sizes[index]; ++node) {
      const double r = column.a * std::sinh(column.s_min + node * step);
      double d_dr = 0.0, d_dz = 0.0;
      const double value = boundary.potential_gradient(r, column.z, k, chargeable, d_dr, d_dz);
      MFEM_VERIFY(value > 0.0 && std::isfinite(value), "The tabulated layered potential is not positive.");
      const double distance = std::hypot(r, boundary.vertical_separation(column.z));
      local.insert(local.end(), { distance * d_dr / value, distance * d_dz / value,
                                  with_values ? std::log(value) + boundary.log_scale(r, column.z, k, chargeable) : 0.0 });
    }
  }

  std::vector<int> counts(n_ranks, 0), displacements(n_ranks, 0);
  for (int block = 0; block < n_ranks; ++block) {
    for (int index = first[block]; index < first[block + 1]; ++index) {
      counts[block] += 3 * sizes[index];
    }
    displacements[block] = block > 0 ? displacements[block - 1] + counts[block - 1] : 0;
  }
  std::vector<double> all(3 * total);
  MPI_Allgatherv(local.data(), static_cast<int>(local.size()), MPI_DOUBLE, all.data(), counts.data(), displacements.data(),
                 MPI_DOUBLE, comm);

  std::size_t offset = 0;
  for (int index = 0; index < n_columns; ++index) {
    Column &column = columns[index];
    while (column.resolved < sizes[index] &&
           boundary.resolved(column.a * std::sinh(column.s_min + column.resolved * step), column.z, k)) {
      ++column.resolved;
    }
    for (int node = 0; node < sizes[index]; ++node, offset += 3) {
      column.radial.push_back(all[offset]);
      column.vertical.push_back(all[offset + 1]);
      if (with_values) {
        column.log_value.push_back(all[offset + 2]);
      }
    }
    columns_.emplace(bits(column.z), std::move(column));
  }
}

double RobinTable::direct(double z, double r, double normal_r, double normal_z, double *log_value) const {
  ++direct_evaluations_;
  double d_dr = 0.0, d_dz = 0.0;
  const double value = boundary_.potential_gradient(r, z, k_, chargeable_, d_dr, d_dz);
  if (log_value) {
    *log_value = std::log(value) + boundary_.log_scale(r, z, k_, chargeable_);
  }
  return -(d_dr * normal_r + d_dz * normal_z) / value;
}

double RobinTable::ratio(double z, double r, double normal_r, double normal_z, double *log_value) const {
  MFEM_VERIFY(!log_value || with_values_, "This Robin table holds no potential values.");
  const auto found = columns_.find(bits(z));
  MFEM_VERIFY(found != columns_.end(), fmt::format("Depth {} is not tabulated.", z));
  const Column &column = found->second;
  const bool resolved = boundary_.resolved(r, z, k_);
  const int begin = resolved ? 0 : column.resolved;
  const int end = resolved ? column.resolved : static_cast<int>(column.radial.size());
  const double t = (std::asinh(r / column.a) - column.s_min) / step_;
  // Too few nodes on this side, or a point between the last node and the end of the side: evaluate directly.
  if (end - begin < 4 || t < begin || t > end - 1) {
    return direct(z, r, normal_r, normal_z, log_value);
  }
  const int first = std::clamp(static_cast<int>(std::floor(t)) - 1, begin, end - 4);
  double radial = 0.0, vertical = 0.0, value = 0.0;
  for (int i = 0; i < 4; ++i) {
    double weight = 1.0;
    for (int j = 0; j < 4; ++j) {
      if (j != i) {
        weight *= (t - (first + j)) / static_cast<double>(i - j);
      }
    }
    radial += weight * column.radial[first + i];
    vertical += weight * column.vertical[first + i];
    if (log_value) {
      value += weight * column.log_value[first + i];
    }
  }
  if (log_value) {
    *log_value = value;
  }
  return -(radial * normal_r + vertical * normal_z) / std::hypot(r, boundary_.vertical_separation(z));
}

} // namespace dcip
