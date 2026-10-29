// 2.5D wavenumber rules fitted to a layered background.

#include "test_common.h"
#include "wavenumber_rule.h"

#include <fmt/format.h>

#include <cmath>
#include <vector>

TEST_CASE("Layered wavenumber reference of a homogeneous earth is the line source and its image", "[wavenumber]") {
  const double sigma = 0.01;
  const WavenumberReference reference = layered_wavenumber_reference({ { 0.0, sigma, 0.0 } }, 0.0);
  for (double r : { 20.0, 150.0, 900.0 }) {
    CHECK(reference.potential(r) == test_common::approx(1.0 / (2.0 * PI * sigma * r), 1.0e-12));
    for (double k : { 1.0e-5, 1.0e-3, 0.01 }) {
      CHECK(reference.transformed(k, r) == test_common::approx(bessel_k0(k * r) / (PI * sigma), 1.0e-12));
    }
  }
}

TEST_CASE("Weights fitted to a resistive basement reduce the a-priori quadrature error", "[wavenumber]") {
  // 100 ohm-m, 50 m over 1000 ohm-m; surface offsets 20 m to 1 km, as in scripts/prototypes/wavenumber_fit.py.
  const WavenumberReference reference = layered_wavenumber_reference({ { 0.0, 0.01, 0.0 }, { 50.0, 0.001, 0.0 } }, 0.0);
  const std::vector<double> offsets = logarithmic_offsets(20.0, 1000.0, 80);
  for (int per_decade : { 3, 4 }) {
    const std::vector<WavenumberNode> log_rule = make_log_wavenumber_quadrature(1.0e-3 / 1000.0, 15.0 / 20.0, per_decade);
    const std::vector<WavenumberNode> fitted = fit_wavenumber_weights(log_rule, { reference }, offsets);
    const double log_error = wavenumber_rule_error(log_rule, reference, offsets);
    const double fitted_error = wavenumber_rule_error(fitted, reference, offsets);
    CAPTURE(per_decade, log_error, fitted_error);
    if (test_common::is_root()) {
      fmt::print("Wavenumber rule, {} per decade ({} nodes): log {:.3e}, layered fit {:.3e}\n", per_decade, log_rule.size(),
                 log_error, fitted_error);
    }
    CHECK(fitted_error < 0.5 * log_error);
  }
  // The independent image-series prototype gives 1.09e-3 for the log rule with 3 points per decade.
  const std::vector<WavenumberNode> rule = make_log_wavenumber_quadrature(1.0e-3 / 1000.0, 15.0 / 20.0, 3);
  CHECK(wavenumber_rule_error(rule, reference, offsets) == test_common::approx(1.09e-3, 0.1));
}

TEST_CASE("A joint fit serves the DC and the instantaneous backgrounds", "[wavenumber]") {
  // A chargeable cover changes the instantaneous background more than the basement.
  const std::vector<BackgroundLayer> background = { { 0.0, 0.01, 0.3 }, { 50.0, 0.001, 0.02 } };
  const WavenumberReference dc = layered_wavenumber_reference(background, 0.0, false);
  const WavenumberReference instantaneous = layered_wavenumber_reference(background, 0.0, true);
  const std::vector<double> offsets = logarithmic_offsets(20.0, 1000.0, 80);
  const std::vector<WavenumberNode> log_rule = make_log_wavenumber_quadrature(1.0e-3 / 1000.0, 15.0 / 20.0, 3);
  const auto dc_fit = fit_wavenumber_weights(log_rule, { dc }, offsets);
  const auto joint = fit_wavenumber_weights(log_rule, { dc, instantaneous }, offsets);
  const double errors[3][2] = { { wavenumber_rule_error(log_rule, dc, offsets), wavenumber_rule_error(log_rule, instantaneous, offsets) },
                                { wavenumber_rule_error(dc_fit, dc, offsets), wavenumber_rule_error(dc_fit, instantaneous, offsets) },
                                { wavenumber_rule_error(joint, dc, offsets), wavenumber_rule_error(joint, instantaneous, offsets) } };
  if (test_common::is_root()) {
    fmt::print("Wavenumber rule DC/instantaneous errors: log {:.2e}/{:.2e}, DC fit {:.2e}/{:.2e}, joint fit {:.2e}/{:.2e}\n",
               errors[0][0], errors[0][1], errors[1][0], errors[1][1], errors[2][0], errors[2][1]);
  }
  CHECK(errors[2][0] < errors[0][0]);
  CHECK(errors[2][1] < errors[0][1]);
}

TEST_CASE("Separate fits give each conductivity state its own weights", "[wavenumber]") {
  const std::vector<BackgroundLayer> background = { { 0.0, 0.01, 0.3 }, { 50.0, 0.001, 0.02 } };
  const WavenumberReference dc = layered_wavenumber_reference(background, 0.0, false);
  const WavenumberReference instantaneous = layered_wavenumber_reference(background, 0.0, true);
  const std::vector<double> offsets = logarithmic_offsets(20.0, 1000.0, 80);
  const std::vector<WavenumberNode> log_rule = make_log_wavenumber_quadrature(1.0e-3 / 1000.0, 15.0 / 20.0, 3);
  const auto dc_fit = fit_wavenumber_weights(log_rule, { dc }, offsets);
  const auto own_fit = fit_wavenumber_weights(log_rule, { instantaneous }, offsets);
  double change = 0.0;
  for (std::size_t i = 0; i < log_rule.size(); ++i) {
    change = std::max(change, std::abs(own_fit[i].weight / dc_fit[i].weight - 1.0));
  }
  CAPTURE(change);
  CHECK(change > 1.0e-4);
  CHECK(wavenumber_rule_error(own_fit, instantaneous, offsets) < wavenumber_rule_error(dc_fit, instantaneous, offsets));
  CHECK(wavenumber_rule_error(dc_fit, dc, offsets) < wavenumber_rule_error(own_fit, dc, offsets));
}
