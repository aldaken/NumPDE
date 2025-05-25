/**
 * @file HodgeLaplacian2D_test.cc
 * @brief NPDE homework HodgeLaplacian2D code
 * @author Ralf Hiptmair
 * @date May 2025
 * @copyright Developed at SAM, ETH Zurich
 */

#include "../hodgelaplacian2d.h"

#include <gtest/gtest.h>
#include <lf/base/lf_assert.h>
#include <lf/mesh/entity.h>
#include <lf/mesh/test_utils/test_meshes.h>

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
// Build a 1-element mesh with a single reference triangle
std::shared_ptr<lf::mesh::Mesh> getRefTriaMesh() {
  // Short name for 2d coordinate vectors
  using coord_t = Eigen::Vector2d;
  // Corner coordinates for a triangle
  using tria_coord_t = Eigen::Matrix<double, 2, 3>;
  // Ftehc a mesh facvtory
  const std::shared_ptr<lf::mesh::hybrid2d::MeshFactory> mesh_factory_ptr =
      std::make_shared<lf::mesh::hybrid2d::MeshFactory>(2);
  // Add points
  mesh_factory_ptr->AddPoint(coord_t({0, 0}));  // point 0
  mesh_factory_ptr->AddPoint(coord_t({1, 0}));  // point 1
  mesh_factory_ptr->AddPoint(coord_t({0, 1}));  // point 2
  // Define triangular cell
  mesh_factory_ptr->AddEntity(lf::base::RefEl::kTria(),
                              std::vector<lf::base::size_type>({0, 1, 2}),
                              std::unique_ptr<lf::geometry::Geometry>(nullptr));
  // Ready to build the mesh data structure
  std::shared_ptr<lf::mesh::Mesh> mesh_p = mesh_factory_ptr->Build();
  LF_ASSERT_MSG(mesh_p->NumEntities(0) == 1, " Mesh should contain 1 cell");
  LF_ASSERT_MSG(mesh_p->NumEntities(1) == 3, " Mesh should contain 3 edges");
  LF_ASSERT_MSG(mesh_p->NumEntities(2) == 3, " Mesh should contain 3 nodes");
  return mesh_p;
}

TEST(HodgeLaplacian, RefTria) {
  // Compute element matrix for reference triangle and compare with precomputed
  // element matrix
  Eigen::MatrixXd M_ref(6, 6);
  // clang-format off
  M_ref << -1.0/12, -1.0/24, -1.0/24, -1.0/2,   0,     0.5,
         -1.0/24, -1.0/12, -1.0/24, 1.0/3, -1.0/6, -1.0/6,
         -1.0/24, -1.0/24, -1.0/12, 1.0/6, 1.0/6, -1.0/3,
         -0.5,   1.0/3,  1.0/6,     2,     2,     2,
          0,   -1.0/6, 1.0/6,      2,     2,     2,
         0.5,  -1.0/6, -1.0/3,      2,     2,     2;
  // clang-format on
  // Obtain "mesh" consisting of only the reference triangle
  std::shared_ptr<lf::mesh::Mesh> mesh_p = getRefTriaMesh();
  // Fetch unit triangle cell
  const lf::mesh::Entity* cell = mesh_p->EntityByIndex(0, 0);
  LF_ASSERT_MSG_CONSTEXPR(cell != nullptr, "Invalid cell");
  HodgeLaplacian2D::HodgeLaplacian2DElementMatrixProvider hlemp{};
  Eigen::MatrixXd Mk = hlemp.Eval(*cell);
  // std::cout << "M_ref = \n" << M_ref << std::endl;
  // std::cout << "MK = \n" << Mk << std::endl;
  // std::cout << "M_ref - MK = \n" << (M_ref - Mk) << std::endl;
  // Comparison without taking into account relative orientations.
  EXPECT_NEAR((M_ref.cwiseAbs() - Mk.cwiseAbs()).norm(), 0.0, 1E-6);
  // Correct for relative orientations
  auto relor = cell->RelativeOrientations();
  LF_ASSERT_MSG(relor.size() == 3, "Triangle should have 3 edges!?");
  for (int k = 0; k < 3; ++k) {
    if (relor[k] == lf::mesh::Orientation::negative) {
      std::cout << "Rel. Or. edge " << k << ": flipped\n";
      // Flip sign of 3+k-th rown and column
      M_ref.col(3 + k) *= -1.0;
      M_ref.row(3 + k) *= -1.0;
    } else {
      std::cout << "Rel. Or. edge " << k << ": straight\n";
    }
  }
  EXPECT_NEAR((M_ref - Mk).norm(), 0.0, 1E-6);
}

TEST(HodgeLaplacian, CmpEval) {
  // Obtain simple test mesh of unit square
  std::shared_ptr<const lf::mesh::Mesh> mesh_p =
      lf::mesh::test_utils::GenerateHybrid2DTestMesh(3, 1.0 / 3.0);
  HodgeLaplacian2D::HodgeLaplacian2DElementMatrixProvider hlemp{};
  // Loop over cells
  for (const lf::mesh::Entity* cell : mesh_p->Entities(0)) {
    EXPECT_EQ(cell->RefEl(), lf::base::RefEl::kTria());
    const Eigen::MatrixXd Mk_ref = hlemp.Eval_alt(*cell);
    const Eigen::MatrixXd Mk = hlemp.Eval(*cell);
    const double diff = (Mk - Mk_ref).norm();
    std::cout << "M_ref = \n" << Mk_ref << std::endl;
    std::cout << "MK = \n" << Mk << std::endl;
    std::cout << "M_ref - MK = \n" << (Mk_ref - Mk) << std::endl;
    EXPECT_NEAR(diff, 0.0, 1E-6);
  }
}

}  // namespace HodgeLaplacian2D::test
