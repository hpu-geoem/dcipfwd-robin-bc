// 2.5D wavenumber rules: a-priori error against a layered background and weights fitted to it.

#pragma once

#include "dc_boundary.h"
#include "integral_transforms.h"

#include <functional>
#include <vector>

namespace dcip {

/// Potential V(r) of a unit current and its full cosine transform Vt(k, r) along the strike, V(r) = (1/pi) int Vt dk.
struct WavenumberReference {
  std::function<double(double)> potential;
  std::function<double(double, double)> transformed;
};

/// Return the reference of a layered background (tops in model z, the first at the surface) with the DC or the
/// instantaneous conductivity, for a source and receivers at one depth; Vt is taken as zero where k r exceeds
/// LayeredPotential::kDecaySwitch.
WavenumberReference layered_wavenumber_reference(const std::vector<BackgroundLayer> &background, double depth,
                                                 bool chargeable = false);

/// Return the largest relative error of (1/pi) sum w Vt(k, r) against V(r) over the offsets.
double wavenumber_rule_error(const std::vector<WavenumberNode> &nodes, const WavenumberReference &reference,
                             const std::vector<double> &offsets);

/// Return the nodes with weights fitted by least squares so that the rule reproduces every reference at the offsets
/// (one block of residuals per reference); the relative deviation from the given weights is penalized with the
/// regularization factor.
std::vector<WavenumberNode> fit_wavenumber_weights(const std::vector<WavenumberNode> &nodes,
                                                   const std::vector<WavenumberReference> &references,
                                                   const std::vector<double> &offsets, double regularization = 1.0e-4);

/// Return n offsets spaced logarithmically from r_min to r_max.
std::vector<double> logarithmic_offsets(double r_min, double r_max, int n);

} // namespace dcip
