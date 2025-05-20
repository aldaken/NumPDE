/**
 * @file HodgeLaplacian2D_test.cc
 * @brief NPDE homework HodgeLaplacian2D code
 * @author Ralf Hiptmair
 * @date May 2025
 * @copyright Developed at SAM, ETH Zurich
 */

#include "../hodgelaplacian2d.h"

#include <gtest/gtest.h>

#include <Eigen/Core>

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

namespace HodgeLaplacian2D::test {
TEST(HodgeLaplacian, Dummy) { EXPECT_TRUE(1 == 1); }

}  // namespace HodgeLaplacian2D::test
