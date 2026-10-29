#include "integral_transforms.h"

#include <mfem.hpp>

#include <algorithm>
#include <cmath>
#include <math.h>

namespace dcip {

namespace {

/// Return exp(x) K_nu(x) for nu = 0 or 1 from the trapezoidal rule for the integral of exp(-x (cosh t - 1)) cosh(nu t).
double scaled_bessel_k(int nu, double x) {
  const double h = std::min(0.2, 0.15 / std::sqrt(x));
  double sum = 0.5;
  for (int step = 1;; ++step) {
    const double t = step * h;
    const double half_sinh = std::sinh(0.5 * t);
    const double term = std::exp(-2.0 * x * half_sinh * half_sinh) * (nu == 0 ? 1.0 : std::cosh(t));
    sum += term;
    if (term < 1.0e-17 * sum) {
      break;
    }
  }
  return h * sum;
}

void verify_argument(double x) { MFEM_VERIFY(x > 0.0 && std::isfinite(x), "Bessel functions require a positive argument."); }

} // namespace

double bessel_k0(double x) {
  verify_argument(x);
  return scaled_bessel_k(0, x) * std::exp(-x);
}

double bessel_k1(double x) {
  verify_argument(x);
  return scaled_bessel_k(1, x) * std::exp(-x);
}

double bessel_k1_over_k0(double x) {
  verify_argument(x);
  return scaled_bessel_k(1, x) / scaled_bessel_k(0, x);
}

double bessel_k0_scaled(double x) {
  verify_argument(x);
  return scaled_bessel_k(0, x);
}

double bessel_k1_scaled(double x) {
  verify_argument(x);
  return scaled_bessel_k(1, x);
}

double bessel_j0(double x) {
#if defined(__APPLE__)
  return ::j0(x);
#else
  return std::cyl_bessel_j(0, x);
#endif
}

double bessel_j1(double x) {
#if defined(__APPLE__)
  return ::j1(x);
#else
  return std::cyl_bessel_j(1, x);
#endif
}

std::vector<WavenumberNode> make_log_wavenumber_quadrature(double k_min, double k_max, int points_per_decade) {
  MFEM_VERIFY(std::isfinite(k_min) && std::isfinite(k_max) && k_min > 0.0 && k_max > k_min,
              "A wavenumber quadrature requires 0 < k_min < k_max.");
  MFEM_VERIFY(points_per_decade >= 2 && points_per_decade <= 64, "Wavenumber points per decade must be in [2, 64].");

  // The integral of a + b ln k over [0, k_min] is k_min times its value at k_min / e.
  std::vector<WavenumberNode> nodes = { { k_min / std::exp(1.0), k_min } };

  mfem::IntegrationRule rule(points_per_decade);
  mfem::QuadratureFunctions1D::GaussLegendre(points_per_decade, &rule);
  const double log_min = std::log(k_min), log_range = std::log(k_max) - log_min;
  const int n_panels = std::max(1, static_cast<int>(std::ceil(log_range / std::log(10.0) - 1.0e-12)));
  for (int panel = 0; panel < n_panels; ++panel) {
    const double width = log_range / n_panels;
    for (int point = 0; point < rule.GetNPoints(); ++point) {
      const mfem::IntegrationPoint &ip = rule.IntPoint(point);
      const double k = std::exp(log_min + width * (panel + ip.x));
      nodes.push_back({ k, width * ip.weight * k });
    }
  }
  return nodes;
}

} // namespace dcip
