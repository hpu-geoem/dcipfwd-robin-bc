#include "test_common.h"
#include "dc_boundary.h"
#include "integral_transforms.h"

#include <algorithm>
#include <cmath>
#include <functional>
#include <limits>
#include <vector>

namespace {

/// Largest relative error of (2 / pi) int K0(k r) dk = 1 / r over distances from nearest to farthest.
double transform_error(const std::vector<WavenumberNode> &nodes, double nearest, double farthest) {
  double worst = 0.0;
  for (int sample = 0; sample <= 40; ++sample) {
    const double r = nearest * std::pow(farthest / nearest, sample / 40.0);
    double sum = 0.0;
    for (const WavenumberNode &node : nodes) {
      sum += node.weight * bessel_k0(node.k * r);
    }
    worst = std::max(worst, std::abs(2.0 / PI * sum * r - 1.0));
  }
  return worst;
}

} // namespace

TEST_CASE("modified Bessel functions of the second kind match reference values", "[bessel]") {
  // x, K0(x), K1(x)
  const double reference[][3] = {
    { 1.0e-6, 1.393144207362641e+01, 9.999999999927843e+05 }, { 1.0e-3, 7.023688800562383e+00, 9.999962381560855e+02 },
    { 0.1, 2.427069024702017e+00, 9.853844780870606e+00 },    { 1.0, 4.210244382407082e-01, 6.019072301972346e-01 },
    { 2.0, 1.138938727495334e-01, 1.398658818165225e-01 },    { 5.0, 3.691098334042594e-03, 4.044613445452163e-03 },
    { 20.0, 5.741237815336524e-10, 5.883057969557038e-10 },   { 100.0, 4.656628229175903e-45, 4.679853735636910e-45 }
  };
  for (const auto &row : reference) {
    CAPTURE(row[0]);
    CHECK(bessel_k0(row[0]) == test_common::approx(row[1], 1.0e-13, 0.0));
    CHECK(bessel_k1(row[0]) == test_common::approx(row[2], 1.0e-13, 0.0));
    CHECK(bessel_k1_over_k0(row[0]) == test_common::approx(row[2] / row[1], 1.0e-13, 0.0));
    CHECK(bessel_k0_scaled(row[0]) == test_common::approx(row[1] * std::exp(row[0]), 1.0e-13, 0.0));
    CHECK(bessel_k1_scaled(row[0]) == test_common::approx(row[2] * std::exp(row[0]), 1.0e-13, 0.0));
  }
}

TEST_CASE("Bessel functions of the first kind match reference values", "[bessel]") {
  // x, J0(x), J1(x)
  const double reference[][3] = { { 0.0, 1.0, 0.0 },
                                  { 0.5, 9.384698072408129e-01, 2.422684576748739e-01 },
                                  { 2.0, 2.238907791412357e-01, 5.767248077568734e-01 },
                                  { 3.0, -2.600519549019334e-01, 3.390589585259365e-01 },
                                  { 10.0, -2.459357644513483e-01, 4.347274616886144e-02 },
                                  { 100.0, 1.998585030422312e-02, -7.714535201411216e-02 } };
  for (const auto &row : reference) {
    CAPTURE(row[0]);
    CHECK(bessel_j0(row[0]) == test_common::approx(row[1], 0.0, 1.0e-14));
    CHECK(bessel_j1(row[0]) == test_common::approx(row[2], 0.0, 1.0e-14));
  }
}

TEST_CASE("ratio of the modified Bessel functions stays finite where they underflow", "[bessel]") {
  for (double x : { 1.0e3, 1.0e4, 1.0e8 }) {
    CAPTURE(x);
    CHECK(bessel_k0(x) == 0.0);
    // K1 / K0 = 1 + 1 / (2 x) - 1 / (8 x^2) + ...
    CHECK(bessel_k1_over_k0(x) == test_common::approx(1.0 + 0.5 / x - 0.125 / (x * x), 1.0e-6, 0.0));
  }
  // k K1(k r) / K0(k r) vanishes with k.
  CHECK(1.0e-12 * bessel_k1_over_k0(1.0e-12 * 50.0) < 1.0e-3);
}

TEST_CASE("wavenumber quadrature has one logarithmic node and Gauss points on equal panels", "[wavenumber]") {
  const double k_min = 2.0e-5, k_max = 3.0;
  const int points = 4;
  const std::vector<WavenumberNode> nodes = make_log_wavenumber_quadrature(k_min, k_max, points);

  // log10(k_max / k_min) = 5.2 needs six panels.
  REQUIRE(nodes.size() == 1 + 6 * points);
  for (std::size_t index = 0; index < nodes.size(); ++index) {
    CHECK(nodes[index].weight > 0.0);
    CHECK(nodes[index].k > (index == 0 ? 0.0 : nodes[index - 1].k));
  }
  CHECK(nodes[1].k > k_min);
  CHECK(nodes.back().k < k_max);

  // The first node integrates a + b ln k over [0, k_min].
  const double a = 0.7, b = -1.3;
  CHECK(nodes[0].weight * (a + b * std::log(nodes[0].k)) ==
        test_common::approx(k_min * (a + b * (std::log(k_min) - 1.0)), 1.0e-13));

  // The panels integrate 1 and 1 / k over [k_min, k_max].
  double length = 0.0, logarithm = 0.0;
  for (std::size_t index = 1; index < nodes.size(); ++index) {
    length += nodes[index].weight;
    logarithm += nodes[index].weight / nodes[index].k;
  }
  CHECK(logarithm == test_common::approx(std::log(k_max / k_min), 1.0e-13));
  CHECK(length == test_common::approx(k_max - k_min, 1.0e-5));

  // Bounds whose ratio overflows still give finite ordered nodes.
  const std::vector<WavenumberNode> wide = make_log_wavenumber_quadrature(1.0e-200, 1.0e200, 2);
  REQUIRE(wide.size() == 1 + 400 * 2);
  for (std::size_t index = 1; index < wide.size(); ++index) {
    REQUIRE(std::isfinite(wide[index].weight));
    REQUIRE(wide[index].k > wide[index - 1].k);
  }

  // A range of exactly two decades has two panels.
  CHECK(make_log_wavenumber_quadrature(1.0e-3, 1.0e-1, points).size() == 1 + 2 * points);
}

TEST_CASE("wavenumber quadrature converges for the cosine transform of K0", "[wavenumber]") {
  // Distances from 2 m to 5 km, wavenumbers from 1e-3 / farthest to 15 / nearest.
  const double nearest = 2.0, farthest = 5000.0;
  std::vector<double> errors;
  for (int points : { 3, 4, 5, 6 }) {
    errors.push_back(
        transform_error(make_log_wavenumber_quadrature(1.0e-3 / farthest, 15.0 / nearest, points), nearest, farthest));
  }
  CAPTURE(errors[0], errors[1], errors[2], errors[3]);
  CHECK(errors[0] < 3.0e-3);
  CHECK(errors[1] < 3.0e-4);
  CHECK(errors[2] < 3.0e-5);
  CHECK(errors[3] < 5.0e-6);
  for (int index = 1; index < 4; ++index) {
    CHECK(errors[index] < 0.3 * errors[index - 1]);
  }

  // A range cut short at either end loses accuracy.
  CHECK(transform_error(make_log_wavenumber_quadrature(1.0e-3 / farthest, 1.5 / nearest, 5), nearest, farthest) > 1.0e-2);
  CHECK(transform_error(make_log_wavenumber_quadrature(10.0 / farthest, 15.0 / nearest, 5), nearest, farthest) > 1.0e-2);
}

namespace {

using test_common::approx;

/// Potential of a unit surface current over a layer of conductivity sigma1 and thickness h on a half-space sigma2,
/// from its image series; kernel(a) is 1 / sqrt(r^2 + a^2) in 3D and K0(k sqrt(y^2 + a^2)) in 2.5D, negligible
/// beyond a = reach.
double two_layer_series(double sigma1, double sigma2, double h, double z, double reach,
                        const std::function<double(double)> &kernel) {
  const double reflection = (sigma1 - sigma2) / (sigma1 + sigma2);
  double sum = 0.0, factor = 1.0;
  if (z <= h) {
    sum = kernel(z);
    for (int image = 1; std::abs(factor) > 1.0e-16 && 2.0 * image * h - z < reach; ++image) {
      factor *= reflection;
      sum += factor * (kernel(2.0 * image * h - z) + kernel(2.0 * image * h + z));
    }
    return sum / (2.0 * PI * sigma1);
  }
  for (int image = 0; std::abs(factor) > 1.0e-16 && 2.0 * image * h + z < reach; ++image) {
    sum += factor * kernel(2.0 * image * h + z);
    factor *= reflection;
  }
  return (1.0 + reflection) * sum / (2.0 * PI * sigma1);
}

double series_3d(double sigma1, double sigma2, double h, double r, double z) {
  return two_layer_series(sigma1, sigma2, h, z, std::numeric_limits<double>::infinity(),
                          [r](double a) { return 1.0 / std::hypot(r, a); });
}

double series_25d(double sigma1, double sigma2, double h, double k, double y, double z) {
  return two_layer_series(sigma1, sigma2, h, z, 50.0 / k, [k, y](double a) { return bessel_k0(k * std::hypot(y, a)); });
}

/// Check the derivatives of a potential series(x, z) against central differences, away from the surface, the axis
/// and the interface at depth h, to a tolerance relative to the gradient magnitude.
void check_gradient(const std::function<double(double, double)> &series, double x, double z, double h, double d_dx, double d_dz,
                    double tolerance = 1.0e-5) {
  const double step = 1.0e-4 * std::max(1.0, std::hypot(x, z));
  const bool along_x = x > step, along_z = z > step && std::abs(z - h) > step;
  const double expected_dx = along_x ? (series(x + step, z) - series(x - step, z)) / (2.0 * step) : 0.0;
  const double expected_dz = along_z ? (series(x, z + step) - series(x, z - step)) / (2.0 * step) : 0.0;
  const double margin = tolerance * std::hypot(expected_dx, expected_dz);
  if (along_x) {
    CHECK(d_dx == approx(expected_dx, 0.0, margin));
  }
  if (along_z) {
    CHECK(d_dz == approx(expected_dz, 0.0, margin));
  }
}

} // namespace

TEST_CASE("layered potential of a homogeneous earth is the source and its surface image") {
  const double sigma = 0.02, depth = 35.0;
  const LayeredPotential layered({ { 0.0, sigma } }, depth);
  for (const auto &[r, z] : std::vector<std::pair<double, double>>{ { 10.0, 0.0 }, { 200.0, 80.0 }, { 0.0, 900.0 } }) {
    CAPTURE(r, z);
    double d_dr = 0.0, d_dz = 0.0;
    const double value = layered.potential(r, z, d_dr, d_dz);
    const double direct = std::hypot(r, z - depth), image = std::hypot(r, z + depth);
    CHECK(value == approx((1.0 / direct + 1.0 / image) / (4.0 * PI * sigma), 1.0e-12));
    CHECK(d_dr == approx(-r * (1.0 / std::pow(direct, 3) + 1.0 / std::pow(image, 3)) / (4.0 * PI * sigma), 1.0e-12, 1.0e-14));
    CHECK(d_dz ==
          approx(-((z - depth) / std::pow(direct, 3) + (z + depth) / std::pow(image, 3)) / (4.0 * PI * sigma), 1.0e-12, 1.0e-14));

    const double k = 0.004, y = r;
    double d_dy = 0.0;
    const double transformed = layered.transformed_potential(k, y, z, d_dy, d_dz);
    CHECK(transformed ==
          approx((bessel_k0(k * std::hypot(y, z - depth)) + bessel_k0(k * std::hypot(y, z + depth))) / (4.0 * PI * sigma),
                 1.0e-12));
  }
}

TEST_CASE("layered potential of a surface source on a two-layer earth matches the image series") {
  const double sigma1 = 0.01, h = 20.0;
  for (double sigma2 : { 0.001, 0.1 }) {
    const LayeredPotential layered({ { 0.0, sigma1 }, { h, sigma2 } }, 0.0);
    // Points in both layers, close to and far from the source, below it and on the far boundary.
    const std::vector<std::pair<double, double>> points = {
      { 5.0, 0.0 },     { 60.0, 12.0 }, { 300.0, 45.0 }, { 2000.0, 3.0 },
      { 3.0e4, 500.0 }, { 0.5, 400.0 }, { 0.0, 3000.0 }, { 4000.0, 3000.0 }
    };
    for (const auto &point : points) {
      const double r = point.first, z = point.second;
      CAPTURE(sigma2, r, z);
      double d_dr = 0.0, d_dz = 0.0;
      const double value = layered.potential(r, z, d_dr, d_dz);
      const double expected = series_3d(sigma1, sigma2, h, r, z);
      CHECK(value == approx(expected, 1.0e-6));
      check_gradient([&](double x, double depth) { return series_3d(sigma1, sigma2, h, x, depth); }, r, z, h, d_dr, d_dz);
    }

    for (const auto &point : points) {
      const double y = point.first, z = point.second;
      // Wavenumbers up to the largest resolved decay at the point.
      for (double k : { 1.0e-6, 1.0e-3, 0.05, 0.99 * LayeredPotential::kDecaySwitch / std::hypot(y, z) }) {
        if (k * std::hypot(y, z) > LayeredPotential::kDecaySwitch) {
          continue;
        }
        CAPTURE(sigma2, k, y, z);
        double d_dy = 0.0, d_dz = 0.0;
        const double value = layered.transformed_potential(k, y, z, d_dy, d_dz);
        CHECK(value == approx(series_25d(sigma1, sigma2, h, k, y, z), 1.0e-6));
        check_gradient([&](double x, double depth) { return series_25d(sigma1, sigma2, h, k, x, depth); }, y, z, h, d_dy, d_dz);
      }
    }
  }
}

TEST_CASE("layered potential over a highly resistive basement matches the image series") {
  const double sigma1 = 0.01, h = 20.0;
  for (double contrast : { 1.0e3, 1.0e5 }) {
    CAPTURE(contrast);
    const double sigma2 = sigma1 / contrast;
    const LayeredPotential layered({ { 0.0, sigma1 }, { h, sigma2 } }, 0.0);
    // Below the source, where the offset equals the secondary path, in the basement, and on far boundaries.
    const std::vector<std::pair<double, double>> points = { { 0.0, 40.0 },   { 20.001, 20.001 }, { 19.999, 20.001 },
                                                            { 100.0, 50.0 }, { 1000.0, 300.0 },  { 3000.0, 3000.0 } };
    // Wavenumbers on both sides of ten times the low-wavenumber scale of the layers.
    const double low_scale = sigma2 / (sigma1 * h);
    for (const auto &point : points) {
      const double r = point.first, z = point.second;
      CAPTURE(r, z);
      double d_dr = 0.0, d_dz = 0.0;
      const double value = layered.potential(r, z, d_dr, d_dz);
      CHECK(value == approx(series_3d(sigma1, sigma2, h, r, z), 1.0e-6));
      check_gradient([&](double x, double depth) { return series_3d(sigma1, sigma2, h, x, depth); }, r, z, h, d_dr, d_dz);
      for (double k : { 1.0e-6, 9.9999 * low_scale, 10.0001 * low_scale, 1.0e-4 }) {
        CAPTURE(k);
        double d_dy = 0.0;
        CHECK(layered.transformed_potential(k, r, z, d_dy, d_dz) == approx(series_25d(sigma1, sigma2, h, k, r, z), 1.0e-6));
        check_gradient([&](double x, double depth) { return series_25d(sigma1, sigma2, h, k, x, depth); }, r, z, h, d_dy, d_dz);
      }
    }
  }
}

TEST_CASE("layered potential over a highly conductive basement matches the image series") {
  const double sigma1 = 0.01, sigma2 = 10.0, h = 20.0;
  const LayeredPotential layered({ { 0.0, sigma1 }, { h, sigma2 } }, 0.0);
  // Far from the source the potential is about a thousandth of that of the top layer alone.
  const std::vector<std::pair<double, double>> points = {
    { 5.0, 0.0 }, { 300.0, 0.0 }, { 3000.0, 12.0 }, { 300.0, 20.5 }, { 1000.0, 3000.0 }
  };
  for (const auto &point : points) {
    const double r = point.first, z = point.second;
    CAPTURE(r, z);
    double d_dr = 0.0, d_dz = 0.0;
    CHECK(layered.potential(r, z, d_dr, d_dz) == approx(series_3d(sigma1, sigma2, h, r, z), 5.0e-6));
    check_gradient([&](double x, double depth) { return series_3d(sigma1, sigma2, h, x, depth); }, r, z, h, d_dr, d_dz);
    for (double k : { 1.0e-6, 1.0e-3, 0.99 * LayeredPotential::kDecaySwitch / std::hypot(r, z) }) {
      if (k * std::hypot(r, z) > LayeredPotential::kDecaySwitch) {
        continue;
      }
      CAPTURE(k);
      double d_dy = 0.0;
      CHECK(layered.transformed_potential(k, r, z, d_dy, d_dz) == approx(series_25d(sigma1, sigma2, h, k, r, z), 5.0e-5));
      check_gradient([&](double x, double depth) { return series_25d(sigma1, sigma2, h, k, x, depth); }, r, z, h, d_dy, d_dz,
                     1.0e-4);
    }
  }
}

TEST_CASE("layered potential of a buried source is reciprocal") {
  const std::vector<ConductiveLayer> layers = { { 0.0, 0.01 }, { 15.0, 0.2 }, { 60.0, 0.002 }, { 400.0, 0.05 } };
  const std::vector<std::pair<double, double>> depths = { { 5.0, 30.0 }, { 30.0, 250.0 }, { 8.0, 900.0 }, { 70.0, 70.0 } };
  for (const auto &[source, receiver] : depths) {
    const LayeredPotential forward(layers, source), backward(layers, receiver);
    for (double r : { 3.0, 80.0, 1500.0 }) {
      CAPTURE(source, receiver, r);
      double d_dr = 0.0, d_dz = 0.0;
      CHECK(forward.potential(r, receiver, d_dr, d_dz) == approx(backward.potential(r, source, d_dr, d_dz), 1.0e-7));
      for (double k : { 1.0e-4, 0.01 }) {
        if (k * std::hypot(r, receiver - source) > LayeredPotential::kDecaySwitch) {
          continue;
        }
        CHECK(forward.transformed_potential(k, r, receiver, d_dr, d_dz) ==
              approx(backward.transformed_potential(k, r, source, d_dr, d_dz), 1.0e-7));
      }
    }
  }
}

TEST_CASE("layered potential does not depend on interfaces between equal layers") {
  const LayeredPotential two({ { 0.0, 0.01 }, { 20.0, 0.001 } }, 3.0);
  const LayeredPotential three({ { 0.0, 0.01 }, { 20.0, 0.001 }, { 150.0, 0.001 } }, 3.0);
  for (const auto &[r, z] : std::vector<std::pair<double, double>>{ { 40.0, 5.0 }, { 400.0, 100.0 }, { 2000.0, 1000.0 } }) {
    CAPTURE(r, z);
    double d_dr2 = 0.0, d_dz2 = 0.0, d_dr3 = 0.0, d_dz3 = 0.0;
    CHECK(three.potential(r, z, d_dr3, d_dz3) == approx(two.potential(r, z, d_dr2, d_dz2), 1.0e-7));
    CHECK(d_dr3 == approx(d_dr2, 1.0e-6));
    CHECK(d_dz3 == approx(d_dz2, 1.0e-6, 1.0e-15));
  }
}
