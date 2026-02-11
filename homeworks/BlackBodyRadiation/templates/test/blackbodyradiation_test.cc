/**
 * @file BlackBodyRadiation_test.cc
 * @brief NPDE homework BlackBodyRadiation code
 * @author Ralf Hiptmair
 * @date June 2025
 * @copyright Developed at SAM, ETH Zurich
 */

#include "../blackbodyradiation.h"

#include <gtest/gtest.h>
#include <lf/assemble/assembler.h>
#include <lf/assemble/dofhandler.h>
#include <lf/base/types.h>
#include <lf/fe/fe_tools.h>
#include <lf/mesh/test_utils/test_meshes.h>
#include <lf/mesh/utils/codim_mesh_data_set.h>
#include <lf/mesh/utils/special_entity_sets.h>

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

namespace BlackBodyRadiation::test {

TEST(BlackBodyRadiation, rhoTest) {
  // Function represented in finite element space
  auto u = [](Eigen::VectorXd x) -> double { return (x[0] + 2 * x[1]); };
  // Wrap into a MeshFunction
  lf::mesh::utils::MeshFunctionGlobal mf_u(u);
  // Coefficients
  const double sigma_c = 1.0;
  // Generate a small unstructured triangular mesh of the unit square
  const std::shared_ptr<lf::mesh::Mesh> mesh_p =
      lf::mesh::test_utils::GenerateHybrid2DTestMesh(3, 1.0 / 3.0);
  // Initialize Lagrangian finite element space
  std::shared_ptr<lf::uscalfe::FeSpaceLagrangeO1<double>> fes_p =
      std::make_shared<lf::uscalfe::FeSpaceLagrangeO1<double>>(mesh_p);
  const lf::assemble::DofHandler &dofh = fes_p->LocGlobMap();
  const lf::base::size_type N = dofh.NumDofs();
  // Obtain flag array for boundary edges (codimension = 1)
  lf::mesh::utils::CodimMeshDataSet<bool> bd_flags{
      lf::mesh::utils::flagEntitiesOnBoundary(mesh_p, 1)};
  // Interpolate u into FE space (exact represenation)
  const Eigen::VectorXd mu_vec = lf::fe::NodalProjection(*fes_p, mf_u);
  // Assemble $\vec{\rhobf}$
  BlackBodyRadiation::StefanBoltzmannElementVectorProvider sb_evp(dofh, mu_vec,
                                                                  bd_flags);
  const Eigen::VectorXd rho_vec =
      lf::assemble::AssembleVectorLocally<Eigen::VectorXd>(1, dofh, sb_evp);

  // Compute integral of $u_h^4$ over the boundasry of the unit square.
  const double itg_val = rho_vec.dot(Eigen::VectorXd::Constant(N, 1.0));
  std::cout << "itg_val = " << itg_val << std::endl;

  EXPECT_NEAR(itg_val, 349.0 / 5.0, 1.0E-6);
}

}  // namespace BlackBodyRadiation::test
