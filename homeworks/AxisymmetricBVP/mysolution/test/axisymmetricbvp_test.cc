/**
 * @file AxisymmetricBVP_test.cc
 * @brief NPDE homework AxisymmetricBVP code
 * @author
 * @date
 * @copyright Developed at SAM, ETH Zurich
 */

#include "../axisymmetricbvp.h"

#include <Eigen/src/Core/Matrix.h>
#include <gtest/gtest.h>

#include <Eigen/Core>
#include <iostream>

/* Test in the google testing framework

  The following assertions are available, syntax
  EXPECT_XX( ....) << [anything that can be givne to std::cerr]

  EXPECT_EQ(val1, val2)
  EXPECT_NEAR(val1, val2, abs_error) -> should be used for numerical results!
  EXPECT_NE(val1, val2)
  EXPECT_TRUE(condition)
  EXPECT_FALSE(condition)
  EXPECT_GE(val1, val2)
  EXPECT_LE(val1, val2)
  EXPECT_GT(val1, val2)
  EXPECT_LT(val1, val2)
  EXPECT_STREQ(str1,str2)
  EXPECT_STRNE(str1,str2)
  EXPECT_STRCASEEQ(str1,str2)
  EXPECT_STRCASENE(str1,str2)

  "EXPECT" can be replaced with "ASSERT" when you want to program to terminate,
 if the assertion is violated.
 */

namespace AxisymmetricBVP::test {

TEST(AxisymmetricBVP, ElemMatTest) {
  auto rho = [](double /*z*/) -> double { return 1.0; };
  auto drho = [](double /*z*/) -> double { return 0.0; };
  // Reference value
  Eigen::Matrix4d A;
  A << 0.375, -0.291666666666667, -0.208333333333333, 0.125, -0.291666666666667,
      0.458333333333333, 0.0416666666666667, -0.208333333333333,
      -0.208333333333333, 0.0416666666666667, 0.458333333333333,
      -0.291666666666667, 0.125, -0.208333333333333, -0.291666666666667, 0.375;

  Eigen::Matrix4d AK =
      AxisymmetricBVP::computeElementMatrix(0, 0, 1, rho, drho);
  std::cout << "AK = \n " << AK << std::endl;
  EXPECT_NEAR((A - AK).norm(), 0.0, 10E-6);
}

// "Artificial" element matrix
template <typename RHO_FUNCTOR, typename DRHO_FUNCTOR>
Eigen::Matrix4d testEMP(unsigned int, unsigned int, unsigned int, RHO_FUNCTOR,
                        DRHO_FUNCTOR) {
  return (Eigen::Matrix4d() << 2, -1, 0, -1, -1, 2, -1, 0, 0, -1, 2, -1, -1, 0,
          -1, 2)
      .finished();
}

TEST(AxisymmetricBVP, LSETest) {
  // Test of assembly of linear system
  const unsigned int n = 4;
  auto rho = [](double /*z*/) -> double { return 1.0; };
  auto drho = [](double /*z*/) -> double { return 0.0; };

  auto [A, rhs] = AxisymmetricBVP::assembleLSE(
      n, rho, drho, testEMP<decltype(rho), decltype(drho)>);

  Eigen::MatrixXd A_dense{A};
  std::cout << "A = \n" << A_dense << std::endl;
  std::cout << "rhs = " << rhs.transpose() << std::endl;
  Eigen::MatrixXd A_ref(15, 15);
  // clang-format off
  A_ref << 
     4, -2,  0,  0,  0, -1,  0,  0,  0,  0,  0,  0,  0,  0,  0,
    -2,  8, -2,  0,  0,  0, -2,  0,  0,  0,  0,  0,  0,  0,  0,
     0, -2,  8, -2,  0,  0,  0, -2,  0,  0,  0,  0,  0,  0,  0,
     0,  0, -2,  8, -2,  0,  0,  0, -2,  0,  0,  0,  0,  0,  0,
     0,  0,  0, -2,  4,  0,  0,  0,  0, -1,  0,  0,  0,  0,  0,
    -1,  0,  0,  0,  0,  4, -2,  0,  0,  0, -1,  0,  0,  0,  0,
     0, -2,  0,  0,  0, -2,  8, -2,  0,  0,  0, -2,  0,  0,  0,
     0,  0, -2,  0,  0,  0, -2,  8, -2,  0,  0,  0, -2,  0,  0,
     0,  0,  0, -2,  0,  0,  0, -2,  8, -2,  0,  0,  0, -2,  0,
     0,  0,  0,  0, -1,  0,  0,  0, -2,  4,  0,  0,  0,  0, -1,
     0,  0,  0,  0,  0, -1,  0,  0,  0,  0,  4, -2,  0,  0,  0,
     0,  0,  0,  0,  0,  0, -2,  0,  0,  0, -2,  8, -2,  0,  0,
     0,  0,  0,  0,  0,  0,  0, -2,  0,  0,  0, -2,  8, -2,  0,
     0,  0,  0,  0,  0,  0,  0,  0, -2,  0,  0,  0, -2,  8, -2,
     0,  0,  0,  0,  0,  0,  0,  0,  0, -1,  0,  0,  0, -2,  4;
  // clang-format on
  Eigen::VectorXd rhs_ref(15);
  rhs_ref << 1, 2, 2, 2, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0;
  EXPECT_NEAR((A_dense - A_ref).norm(), 0.0, 1.0E-6);
  EXPECT_NEAR((rhs - rhs_ref).norm(), 0.0, 1.0E-6);
}

TEST(AxisymmetricBVP, LinSolTest) {
  // Solution must be linear for cylinder
  const unsigned int n = 10;
  // Build linear solution
  Eigen::MatrixXd u_lin_2D =
      Eigen::VectorXd::Constant(n + 1, 1.0) *
      Eigen::VectorXd::LinSpaced(n + 1, 1.0, 0.0).transpose();
  Eigen::Map<Eigen::VectorXd> u_lin(u_lin_2D.data(), (n + 1) * (n + 1), 1);

  Eigen::VectorXd sol = AxisymmetricBVP::solveHeatBolt(
      n, [](double /*z*/) -> double { return 1.0; },
      [](double /*z*/) -> double { return 0.0; });
  /*
  std::cout << "u_lin = " << u_lin.transpose() << std::endl;
  std::cout << "sol = " << sol.transpose() << std::endl;
  std::cout << "linearly decreasing temperature\n" << u_lin_2D << std::endl;
  std::cout << "solution =\n"
            << Eigen::Map<Eigen::MatrixXd>(sol.data(), n + 1, n - 1).transpose()
            << std::endl;
  */
  EXPECT_NEAR((u_lin.segment(n + 1, n * n - 1) - sol).norm(), 0.0, 10E-6);
}

TEST(AxisymmetricBVP, MaxPrincTest) {
  // Test whether discrete maximum principle is satisfied
  const unsigned int n = 10;
  Eigen::VectorXd sol = AxisymmetricBVP::solveHeatBolt(
      n, [](double z) -> double { return (1.0 + z * z); },
      [](double z) -> double { return 2 * z; });
  /*
  std::cout << "solution =\n"
            << Eigen::Map<Eigen::MatrixXd>(sol.data(), n + 1, n - 1).transpose()
            << std::endl;
  */
  EXPECT_TRUE(sol.maxCoeff() <= 1.0);
  EXPECT_TRUE(sol.minCoeff() > -0.0);
}

}  // namespace AxisymmetricBVP::test
