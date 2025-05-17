/**
 * @file leapfrogdissipativewave.h
 * @brief NPDE homework LeapfrogDissipativeWave code
 * @author R. Hiptmair
 * @date July 2022
 * @copyright Developed at SAM, ETH Zurich
 */

#ifndef LDW_H_H
#define LDW_H_H

#include <lf/assemble/assemble.h>
#include <lf/fe/fe.h>
#include <lf/io/io.h>
#include <lf/mesh/hybrid2d/hybrid2d.h>
#include <lf/mesh/utils/utils.h>
#include <lf/uscalfe/uscalfe.h>

#include <tuple>

namespace LeapfrogDissipativeWave {
/** @brief Dissipative leapfrod timestepping for u''=u'+u = 0
 *
 * @param M number of timesteps
 */
double timestepScalarTestProblem(unsigned int M);

/** @brief Tabulating temporal discretization errors */
void convergenceScalarTestProblem(void);

/** @brief Computation of Galerkin matrices in triplet format
 *
 * @param fe_space pointer to underlying quadratic Lagrangian FE space
 * @return tuple of the three Galerkin matrices A, B, and M
 */
extern std::tuple<lf::assemble::COOMatrix<double>,
                  lf::assemble::COOMatrix<double>,
                  lf::assemble::COOMatrix<double>>
computeGalerkinMatrices(
    std::shared_ptr<const lf::uscalfe::FeSpaceLagrangeO2<double>> fe_space_p);

/** @brief Disspative leapfrog timestepping for methof-of-lines ODE
 *
 * @param fe_space_p pointer to underlying FE space including local-to-global
 * index mapping and mesh
 * @param T final time
 * @param M number of timesteps
 * @param mu0 basis expansion coefficient vector for initial value for
 * mu-variable
 * @param nu0  basis expansion coefficient vector for initial value for
 * nu-variable
 * @return basis expansion coefficient vector for approximation of u(T)
 */
Eigen::VectorXd timestepDissipativeWaveEquation(
    std::shared_ptr<const lf::uscalfe::FeSpaceLagrangeO2<double>> fe_space_p,
    double T, unsigned int M, Eigen::VectorXd mu0, Eigen::VectorXd nu0);

/** @brief convergence test for dissipative leapfrog on unit square
 *
 * @param reflevels no of refinementt levels of finite-element mesh
 * @param T final time
 * @param M0 number of timesteps on coarsest level
 */
void convergenceDissipativeLeapfrog(unsigned int reflevels, double T = 1.0,
                                    unsigned int M0 = 10,
                                    unsigned int Mfac = 2);

void testDissipativeLeapfrog(void);
}  // namespace LeapfrogDissipativeWave

#endif
