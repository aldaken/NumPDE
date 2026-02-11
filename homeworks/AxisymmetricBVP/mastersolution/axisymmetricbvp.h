/**
 * @file axisymmetricbvp.h
 * @brief NPDE homework AxisymmetricBVP code
 * @author Ralf Hiptmair
 * @date June 2025
 * @copyright Developed at SAM, ETH Zurich
 */

#ifndef AxisymmetricBVP_H_
#define AxisymmetricBVP_H_

// Include almost all parts of LehrFEM++; soem my not be needed
#include <lf/assemble/assemble.h>
#include <lf/fe/fe.h>
#include <lf/geometry/geometry_interface.h>
#include <lf/io/io.h>
#include <lf/mesh/hybrid2d/hybrid2d.h>
#include <lf/mesh/utils/utils.h>
#include <lf/uscalfe/uscalfe.h>

#include <Eigen/Core>
#include <Eigen/Sparse>
#include <initializer_list>
#include <iostream>

#define assertm(exp, msg) assert((void(msg), exp))

namespace AxisymmetricBVP {

/** @brief Data for a 1D quadrature rule */
/* SAM_LISTING_BEGIN_1 */
struct QuadRule1D {
  QuadRule1D(std::initializer_list<double> points,
             std::initializer_list<double> weights);
  Eigen::VectorXd points_;
  Eigen::VectorXd weights_;
};
/* SAM_LISTING_END_1 */

/** @brief Generate Gauss quadrature rules */
QuadRule1D make_GaussRule(unsigned int n_pts);

/** Computing element matrix */
/* SAM_LISTING_BEGIN_2 */
template <typename RHO_FUNCTOR, typename DRHO_FUNCTOR>
Eigen::Matrix4d computeElementMatrix(unsigned int i, unsigned int j,
                                     unsigned int n, RHO_FUNCTOR rho,
                                     DRHO_FUNCTOR d_rho) {
  assertm((i < n) && (j < n), "indices out of bound");
  // Meshwidth
  const double h = 1.0 / n;
  // Function r -> r on cell in **reference coordinates**
  auto r_coeff = [j, h](double r_ref) {
    return (static_cast<double>(j) + r_ref) * h;
  };
  // Fnction z -> z in reference coordinates
  auto z_coeff = [i, h](double z_ref) {
    return (static_cast<double>(i) + z_ref) * h;
  };
  // Gradients of local shape functions in reference coordinates
  std::vector<std::function<Eigen::Vector2d(Eigen::Vector2d)>> grad_lsf(
      {[h](Eigen::Vector2d x) -> Eigen::Vector2d {
         return Eigen::Vector2d(x[1] - 1.0, x[0] - 1.0) / h;
       },
       [h](Eigen::Vector2d x) -> Eigen::Vector2d {
         return Eigen::Vector2d(1.0 - x[1], -x[0]) / h;
       },
       [h](Eigen::Vector2d x) -> Eigen::Vector2d {
         return Eigen::Vector2d(x[1], x[0]) / h;
       },
       [h](Eigen::Vector2d x) -> Eigen::Vector2d {
         return Eigen::Vector2d(-x[1], 1.0 - x[0]) / h;
       }});
  // We need a Gauss quadrature rule that is exact for polynomials of
  // at least degree 7: Use 4-point Gauss quadrature rule.
  const unsigned int P = 4;
  QuadRule1D qr = make_GaussRule(P);
  assertm(qr.points_.size() == P, "Wrong number of quadrature points");

  Eigen::Matrix4d AK = Eigen::Matrix4d::Zero();
  // (Double) loop over quadrature points
  Eigen::Matrix4d A_add;
  for (int k = 0; k < P; ++k) {
    for (int m = 0; m < P; ++m) {
      const double r_ref = qr.points_[k];
      const double z_ref = qr.points_[m];
      const double r = r_coeff(r_ref);
      const double z = z_coeff(z_ref);
      const double z_star = 2 * (z - 0.5);
      const double rho_val = rho(z_star);
      const double drho_val = d_rho(z_star);
      Eigen::Matrix2d D =
          (Eigen::Matrix2d() << 1 + r * r * drho_val * drho_val,
           -0.5 * r * rho_val * drho_val, -0.5 * r * rho_val * drho_val, 0.25)
              .finished();
      // Compute contribution for a quadrature point
      for (int i = 0; i < 4; ++i) {
        for (int j = 0; j <= i; ++j) {
          A_add(j, i) = A_add(i, j) =
              2.0 * r *
              (grad_lsf[j](Eigen::Vector2d(r_ref, z_ref))
                   .dot(D * grad_lsf[i](Eigen::Vector2d(r_ref, z_ref))));
        }
      }
      AK += (h * h * qr.weights_[k] * qr.weights_[m]) * A_add;
    }
  }
  return AK;
}
/* SAM_LISTING_END_2 */

/** @brief Assembly of linear system of equations */

/* SAM_LISTING_BEGIN_3 */
template <typename RHO_FUNCTOR, typename DRHO_FUNCTOR, typename ELEMAT_FUNCTOR>
std::pair<Eigen::SparseMatrix<double>, Eigen::VectorXd> assembleLSE(
    unsigned int n, RHO_FUNCTOR rho, DRHO_FUNCTOR d_rho, ELEMAT_FUNCTOR emp) {
  unsigned int N = n * n - 1;  // Number of d.o.f.
  // A sparse matrix with at most 9 non-zero entries per row/column
  Eigen::SparseMatrix<double> A(N, N);
  A.reserve(Eigen::VectorXi::Constant(N, 10));
  // Right-hand side vector
  Eigen::VectorXd phi = Eigen::VectorXd::Zero(N);
  // Run through bottom row of cells
  for (int j = 0; j < n; ++j) {
    // Compute element matrix
    const Eigen::Matrix4d AK = emp(0, j, n, rho, d_rho);
    // Assemble intries into system matrix
    const unsigned int ne_idx = j;
    const unsigned int nw_idx = j + 1;
    A.coeffRef(ne_idx, ne_idx) += AK(3, 3);
    A.coeffRef(nw_idx, nw_idx) += AK(2, 2);
    A.coeffRef(nw_idx, ne_idx) += AK(2, 3);
    A.coeffRef(ne_idx, nw_idx) += AK(3, 2);
    // Assemble entries of the right hand side vector for Dirichlet boundary
    // values = 1
    const Eigen::Vector4d phi_loc = AK * Eigen::Vector4d(1.0, 1.0, 0.0, 0.0);
    phi[ne_idx] -= phi_loc[3];
    phi[nw_idx] -= phi_loc[2];
  }
  // Run through interior rows of cells
  for (int i = 1; i < n - 1; ++i) {
    for (int j = 0; j < n; ++j) {
      // Compute element matrix
      const Eigen::Matrix4d AK = emp(i, j, n, rho, d_rho);
      // Assemble intries into system matrix
      std::array<unsigned int, 4> v_idx = {
          (i - 1) * (n + 1) + j, (i - 1) * (n + 1) + j + 1, i * (n + 1) + j + 1,
          i * (n + 1) + j};
      for (int k = 0; k < 4; k++) {
        for (int m = 0; m < 4; m++) {
          A.coeffRef(v_idx[k], v_idx[m]) += AK(k, m);
        }
      }
    }
  }
  // Run through top row of cells
  const int i = n - 1;
  for (int j = 0; j < n; ++j) {
    // Compute element matrix
    const Eigen::Matrix4d AK = emp(i, j, n, rho, d_rho);
    // Assemble intries into system matrix
    const unsigned int se_idx = (i - 1) * (n + 1) + j;
    const unsigned int sw_idx = se_idx + 1;
    A.coeffRef(se_idx, se_idx) += AK(0, 0);
    A.coeffRef(sw_idx, sw_idx) += AK(1, 1);
    A.coeffRef(sw_idx, se_idx) += AK(0, 1);
    A.coeffRef(se_idx, sw_idx) += AK(1, 0);
  }
  return {A, phi};
}
/* SAM_LISTING_END_3 */

/* SAM_LISTING_BEGIN_4 */
template <typename RHO_FUNCTOR, typename DRHO_FUNCTOR>
Eigen::VectorXd solveHeatBolt(unsigned int n, RHO_FUNCTOR rho,
                              DRHO_FUNCTOR d_rho) {
  // Obtain linear system
  auto [A, rhs] = assembleLSE(
      n, rho, d_rho, computeElementMatrix<decltype(rho), decltype(d_rho)>);
  // Carry out direct Gaussian elimination
  Eigen::SparseLU<Eigen::SparseMatrix<double>> ALU;
  ALU.compute(A);
  assertm(ALU.info() == Eigen::Success, "LU decomposition failed");
  // Forward and backward substitution
  Eigen::VectorXd res = ALU.solve(rhs);
  assertm(ALU.info() == Eigen::Success, "Solution failed");
  return res;
}
/* SAM_LISTING_END_4 */

}  // namespace AxisymmetricBVP

#endif
