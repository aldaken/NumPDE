/**
 * @file MaxwellEvlTM_test.cc
 * @brief NPDE homework MaxwellEvlTM code
 * @author Ralf Hoptmair
 * @date June 2025
 * @copyright Developed at SAM, ETH Zurich
 */

#include "../maxwellevltm.h"

#include <Eigen/src/SparseCore/SparseMatrix.h>
#include <gtest/gtest.h>
#include <lf/assemble/assembly_types.h>
#include <lf/assemble/dofhandler.h>
#include <lf/base/lf_assert.h>
#include <lf/mesh/entity.h>
#include <lf/mesh/test_utils/test_meshes.h>
#include <lf/mesh/utils/mesh_function_constant.h>
#include <lf/mesh/utils/mesh_function_global.h>
#include <lf/mesh/utils/print_info.h>

#include <Eigen/Core>
#include <ostream>

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

namespace MaxwellEvlTM::test {

// Generate "mesh" of unit square consisting of two triangles
// Partly copied from lecturedemodoc.cc
std::shared_ptr<lf::mesh::Mesh> getTwoTriagMesh() {
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
  mesh_factory_ptr->AddPoint(coord_t({1, 1}));  // point 3
  // Define triangular cells
  mesh_factory_ptr->AddEntity(lf::base::RefEl::kTria(),
                              std::vector<lf::base::size_type>({0, 1, 2}),
                              std::unique_ptr<lf::geometry::Geometry>(nullptr));
  mesh_factory_ptr->AddEntity(lf::base::RefEl::kTria(),
                              std::vector<lf::base::size_type>({3, 1, 2}),
                              std::unique_ptr<lf::geometry::Geometry>(nullptr));
  // Ready to build the mesh data structure
  std::shared_ptr<lf::mesh::Mesh> mesh_p = mesh_factory_ptr->Build();
  return mesh_p;
}

// Build node-edge incidence matrix
Eigen::SparseMatrix<double> buildD0(const lf::assemble::DofHandler &dofh_h) {
  // Obtain mesh
  std::shared_ptr<const lf::mesh::Mesh> mesh_p = dofh_h.Mesh();
  LF_ASSERT_MSG((dofh_h.NumDofs() == mesh_p->NumEntities(1)),
                "DofH must manage 1 dof per edge");
  // Build matrix in triplet (COO) format
  std::vector<Eigen::Triplet<double>> triplets{};
  // Run through the edges of the mesh
  for (const lf::mesh::Entity *edge : mesh_p->Entities(1)) {
    std::span<const lf::assemble::gdof_idx_t> ed_dof_idxs{
        dofh_h.GlobalDofIndices(*edge)};
    LF_ASSERT_MSG(ed_dof_idxs.size() == 1, "Edge may carry only one dof");
    const lf::assemble::gdof_idx_t ed_dof_idx = ed_dof_idxs[0];
    // Fetch endpoints of the current edge
    std::span<const lf::mesh::Entity *const> endpoints{edge->SubEntities(1)};
    LF_ASSERT_MSG(endpoints.size() == 2, "Edge must have two endpoints");
    const Eigen::Index p0_idx = mesh_p->Index(*endpoints[0]);
    const Eigen::Index p1_idx = mesh_p->Index(*endpoints[1]);
    // Edge is directed from enpoints 0 to endpoint 1
    triplets.emplace_back(ed_dof_idx, p0_idx, -1.0);
    triplets.emplace_back(ed_dof_idx, p1_idx, 1.0);
  }
  Eigen::SparseMatrix<double> D0(mesh_p->NumEntities(1),
                                 mesh_p->NumEntities(2));
  D0.setFromTriplets(triplets.begin(), triplets.end());
  return D0;
}

TEST(MaxwellEvlTM, BD0Test) {
  // Obtain simple test mesh of unit square
  std::shared_ptr<const lf::mesh::Mesh> mesh_p =
      lf::mesh::test_utils::GenerateHybrid2DTestMesh(3, 1.0 / 3.0);
  // Set up DofHandler for 1-form Whitney FEM (edge elements)
  lf::assemble::UniformFEDofHandler dofh_h(mesh_p,
                                           {{lf::base::RefEl::kPoint(), 0},
                                            {lf::base::RefEl::kSegment(), 1},
                                            {lf::base::RefEl::kTria(), 0},
                                            {lf::base::RefEl::kQuad(), 0}});
  // Set up DofHandler for 2-form Whitney FEM (p.w. constants)
  lf::assemble::UniformFEDofHandler dofh_e(mesh_p,
                                           {{lf::base::RefEl::kPoint(), 0},
                                            {lf::base::RefEl::kSegment(), 0},
                                            {lf::base::RefEl::kTria(), 1},
                                            {lf::base::RefEl::kQuad(), 1}});
  // Build matrix B
  // std::cout << "Building matrix B" << std::endl;
  Eigen::SparseMatrix<double> B = MaxwellEvlTM::buildB(dofh_e, dofh_h);
  // std::cout << "Matrix B built" << std::endl;
  // Build incidence matrix D0
  Eigen::SparseMatrix<double> D0 = buildD0(dofh_h);
  // std::cout << "Matrix D0 built\n";
  // Check, if B*D0 is zero
  EXPECT_NEAR((B.transpose() * D0).norm(), 0.0, 1E-6);
}

TEST(MaxwellEvlTM, BlfBtwoTriag) {
  // Obtain simple test mesh of unit square consisting of
  // two triangles.
  std::shared_ptr<const lf::mesh::Mesh> mesh_p = getTwoTriagMesh();
  // std::cout << "Mesh Information:" << std::endl;
  // lf::mesh::utils::PrintInfo(std::cout,*mesh_p,100);
  // Set up DofHandler for 1-form Whitney FEM (edge elements)
  lf::assemble::UniformFEDofHandler dofh_h(mesh_p,
                                           {{lf::base::RefEl::kPoint(), 0},
                                            {lf::base::RefEl::kSegment(), 1},
                                            {lf::base::RefEl::kTria(), 0},
                                            {lf::base::RefEl::kQuad(), 0}});
  // Set up DofHandler for 2-form Whitney FEM (p.w. constants)
  lf::assemble::UniformFEDofHandler dofh_e(mesh_p,
                                           {{lf::base::RefEl::kPoint(), 0},
                                            {lf::base::RefEl::kSegment(), 0},
                                            {lf::base::RefEl::kTria(), 1},
                                            {lf::base::RefEl::kQuad(), 1}});
  // Build matrix B
  Eigen::SparseMatrix<double> B = MaxwellEvlTM::buildB(dofh_e, dofh_h);

  // Test vetor field
  auto u = [](Eigen::Vector2d x) -> Eigen::Vector2d { return {-x[1], x[0]}; };
  lf::mesh::utils::MeshFunctionGlobal mf_u(u);
  // Constant scalar test function
  lf::mesh::utils::MeshFunctionConstant mf_c(0.5);
  // Interpolate into FE spaces
  Eigen::VectorXd dofvec_u = MaxwellEvlTM::nodalProjectionWF1(dofh_h, mf_u);
  Eigen::VectorXd dofvec_c = MaxwellEvlTM::nodalProjectionPWConst(dofh_e, mf_c);
  // Print degrees of freedom
  std::cout << "dofvec_u = " << dofvec_u.transpose() << std::endl;
  // Print matrix B
  const Eigen::MatrixXd B_dense = B;
  std::cout << "B = \n" << B_dense << std::endl;
  // Evaluate bilnear form
  const double blfval = dofvec_u.transpose() * B * dofvec_c;
  // std::cout << "Value b(c,u) = " << blfval << std::endl;
  EXPECT_NEAR(blfval, 1.0, 1.0E-6);
}

TEST(MaxwellEvlTM, BlfTest) {
  // Obtain simple test mesh of unit square
  std::shared_ptr<const lf::mesh::Mesh> mesh_p =
      lf::mesh::test_utils::GenerateHybrid2DTestMesh(3, 1.0 / 3.0);
  // Set up DofHandler for 1-form Whitney FEM (edge elements)
  lf::assemble::UniformFEDofHandler dofh_h(mesh_p,
                                           {{lf::base::RefEl::kPoint(), 0},
                                            {lf::base::RefEl::kSegment(), 1},
                                            {lf::base::RefEl::kTria(), 0},
                                            {lf::base::RefEl::kQuad(), 0}});
  // Set up DofHandler for 2-form Whitney FEM (p.w. constants)
  lf::assemble::UniformFEDofHandler dofh_e(mesh_p,
                                           {{lf::base::RefEl::kPoint(), 0},
                                            {lf::base::RefEl::kSegment(), 0},
                                            {lf::base::RefEl::kTria(), 1},
                                            {lf::base::RefEl::kQuad(), 1}});
  // Build matrix B
  Eigen::SparseMatrix<double> B = MaxwellEvlTM::buildB(dofh_e, dofh_h);
  // Build matrix Meps for eps = 1
  lf::mesh::utils::MeshFunctionConstant mf_eps(1.0);
  Eigen::SparseMatrix<double> Meps = MaxwellEvlTM::buildMeps(dofh_e, mf_eps);
  // Build matrix Mmu for mu = 1
  lf::mesh::utils::MeshFunctionConstant mf_mu(1.0);
  Eigen::SparseMatrix<double> Mmu = MaxwellEvlTM::buildMmu(dofh_h, mf_mu);

  // Test vetor field
  auto u = [](Eigen::Vector2d x) -> Eigen::Vector2d {
    // return {2.0 - x[1] / 3.0, 1.0 + x[0] / 3.0};
    return {2.0 - x[1], 1.0 + x[0]};
    // return {-x[1] , x[0] };
  };
  lf::mesh::utils::MeshFunctionGlobal mf_u(u);
  // Constant scalar test function
  lf::mesh::utils::MeshFunctionConstant mf_c(0.5);
  // Interpolate into FE spaces
  Eigen::VectorXd dofvec_u = MaxwellEvlTM::nodalProjectionWF1(dofh_h, mf_u);
  Eigen::VectorXd dofvec_c = MaxwellEvlTM::nodalProjectionPWConst(dofh_e, mf_c);
  // Evaluate bilnear form
  const double blfval = dofvec_u.transpose() * B * dofvec_c;
  // std::cout << "Value b(c,u) = " << blfval << std::endl;
  EXPECT_NEAR(blfval, 1.0, 1.0E-6);
  const double blfepsval = dofvec_c.transpose() * Meps * dofvec_c;
  // std::cout << "Value m_eps(c,c) = " << blfepsval << std::endl;
  EXPECT_NEAR(blfepsval, 0.25, 1E-6);
  const double blfmuval = dofvec_u.transpose() * Mmu * dofvec_u;
  // std::cout << "value m_mu(u,u) = " << blfmuval << std::endl;
  EXPECT_NEAR(blfmuval, 14.0 / 3.0, 1E-6);
}

}  // namespace MaxwellEvlTM::test
