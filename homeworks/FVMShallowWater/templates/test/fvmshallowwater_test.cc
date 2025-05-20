/**
 * @file FVMShallowWater_test.cc
 * @brief NPDE homework FVMShallowWater code
 * @author
 * @date
 * @copyright Developed at SAM, ETH Zurich
 */

#include "../fvmshallowwater.h"

#include <gtest/gtest.h>

#include <Eigen/Core>
#include <Eigen/Dense>
#include <cstddef>

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

namespace FVMShallowWater::test {


TEST(FVMShallowWater, numfluxLFSWE1) {
  Eigen::Vector2d u, v, exact_flux;
  u << 1., 2.;
  v << 1., 2.;
  exact_flux << 2., 4.5;
  Eigen::Vector2d res = numfluxLFSWE(u, v);
  ASSERT_NEAR((res - exact_flux).norm(), 0.0, 1e-8);
}

TEST(FVMShallowWater, numfluxLFSWE2) {
  Eigen::Vector2d u, v, exact_flux;
  u << 1., 2.;
  v << 3., 5.;
  exact_flux << 2., 4.5;
  Eigen::Vector2d res = numfluxLFSWE(u, v);
  ASSERT_NEAR((res - exact_flux).norm(), 0.0, 1e-8);
}

TEST(FVMShallowWater, numfluxHLLESWE1) {
  Eigen::Vector2d u, v, exact_flux;
  u << 1., 3.;
  v << .5, 4.;
  exact_flux << 3., 9.5;
  Eigen::Vector2d res = numfluxHLLESWE(u, v);
  ASSERT_NEAR((res - exact_flux).norm(), 0.0, 1e-8);
}

TEST(FVMShallowWater, numfluxHLLESWE2) {
  Eigen::Vector2d u, v, exact_flux;
  u << 1., -4.;
  v << .5, -3.;
  exact_flux << -3., 18. + 1. / 8.;
  Eigen::Vector2d res = numfluxHLLESWE(u, v);
  ASSERT_NEAR((res - exact_flux).norm(), 0.0, 1e-8);
}

TEST(FVMShallowWater, numfluxHLLESWE3) {
  Eigen::Vector2d u, v, exact_flux;
  u << 1., -4.;
  v << .5, 3.;
  exact_flux << -0.128442, 0.149371;
  Eigen::Vector2d res = numfluxHLLESWE(u, v);
  ASSERT_NEAR((res - exact_flux).norm(), 0.0, 1e-5);
}



TEST(FVMShallowWater, B_isPhysicalTwoShockSolution1) {
  Eigen::Vector2d ul, us, ur;
  ul << 0.5, 0.4896896369201712;
  us << 1, 0.36700683814454793;
  ur << 0.5, -0.12268279877562327;

  bool res = FVMShallowWater::isPhysicalTwoShockSolution(ul, us, ur);
  ASSERT_TRUE(res);
}

TEST(FVMShallowWater, B_isPhysicalTwoShockSolution2) {
  Eigen::Vector2d ul, us, ur;
  ul << 0.5, 0.4896896369201712;
  us << 1, 0.36700683814454793;
  ur << 0.5, 0.4896896369201712;

  bool res = FVMShallowWater::isPhysicalTwoShockSolution(ul, us, ur);
  ASSERT_FALSE(res);
}

TEST(FVMShallowWater, solveSWE) {
  // Simple upwind flux for linear advection
  // Data
  double a = -1.0;
  double b = 2.0;
  double T = 0.5;
  size_t N = 300;
  // Initial values
  auto u0 = [](double x) -> Eigen::Vector2d {
    if (x > 0 && x < 1)
      return (Eigen::Vector2d() << 1. - x * (1. - x), 1.).finished();

    return (Eigen::Vector2d() << 1., 0.).finished();
  };
  // Recording intermediate approximations
  auto res =
      FVMShallowWater::solveSWE(T, N, &FVMShallowWater::numfluxLFSWE, u0);
  Eigen::Vector2d exact_at_200;
  exact_at_200 << 0.411047, 0.294142;
  ASSERT_NEAR((res[200] - exact_at_200).norm(), 0., 1e-4);
}

}  // namespace FVMShallowWater::test
