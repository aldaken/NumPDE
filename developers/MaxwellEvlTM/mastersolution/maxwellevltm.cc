/**
 * @file maxwellevltm.cc
 * @brief NPDE homework MaxwellEvlTM code
 * @author Ralf Hiptmair
 * @date June 2025
 * @copyright Developed at SAM, ETH Zurich
 */

#include "maxwellevltm.h"

#include <lf/assemble/assembler.h>
#include <lf/assemble/assembly_types.h>
#include <lf/base/lf_assert.h>
#include <lf/base/ref_el.h>
#include <lf/geometry/geometry_interface.h>
#include <lf/mesh/test_utils/test_meshes.h>
#include <lf/mesh/utils/mesh_function_constant.h>
#include <lf/mesh/utils/mesh_function_global.h>

namespace MaxwellEvlTM {
/* SAM_LISTING_BEGIN_1 */
B_EMP::ElemMat B_EMP::Eval(const lf::mesh::Entity &cell) {
  LF_VERIFY_MSG(cell.RefEl() == lf::base::RefEl::kTria(),
                "Unsupported cell type " << cell.RefEl());
  // Element matrix based on induced orientation of edges
  // is the matrix $[1,1,1]^{\top}$.
  // We have to correct for orientation mismatch
  auto relor = cell.RelativeOrientations();
  LF_ASSERT_MSG(relor.size() == 3, "Triangle should have 3 edges!?");
  for (int k = 0; k < 3; ++k) {
    if (relor[k] == lf::mesh::Orientation::negative) {
      // Flip sign of matrix entry
      BK_(k, 0) = -1.0;
    } else {
      BK_(k, 0) = 1.0;
    }
  }
  // We also have to correct for the orientation of the cell
  // We check whether the spanning vectors form a right-handed coordinate system
  const Eigen::MatrixXd verts{lf::geometry::Corners(*cell.Geometry())};
  Eigen::Matrix2d X;
  X.col(0) = verts.col(1) - verts.col(0);
  X.col(1) = verts.col(2) - verts.col(0);
  if (X.determinant() < 0) {
    BK_ *= -1.0;
  }
  return BK_;
}
/* SAM_LISTING_END_1 */

Eigen::SparseMatrix<double> buildB(const lf::assemble::DofHandler &dofh_e,
                                   const lf::assemble::DofHandler &dofh_h) {
  // Obtain mesh
  std::shared_ptr<const lf::mesh::Mesh> mesh_p = dofh_h.Mesh();
  LF_ASSERT_MSG(mesh_p == dofh_e.Mesh(),
                "DofHandlers must be based on same mesh");
  const lf::mesh::Mesh &mesh = *mesh_p;
  LF_ASSERT_MSG((dofh_h.NumDofs() == mesh.NumEntities(1)),
                "DofH must manage 1 dof per edge");
  LF_ASSERT_MSG((dofh_e.NumDofs() == mesh.NumEntities(0)),
                "DofH must manage 1 dof per cell");
  // Temporary matrix in triplet format
  const lf::assemble::size_type N1 = dofh_h.NumDofs();  // dim test space
  const lf::assemble::size_type N2 = dofh_e.NumDofs();  // dim trial space
  lf::assemble::COOMatrix<double> B_COO(N1, N2);
  // ENTITY_MATRIX_PROVIDER object
  B_EMP B_emp;
  // std::cout << "buildB: building " << N1 << " x " << N2 << "-matrix" <<
  // std::endl;
  lf::assemble::AssembleMatrixLocally(0, dofh_e, dofh_h, B_emp, B_COO);
  // std::cout << "buildB: triplets initialized" << std::endl;
  return B_COO.makeSparse();
}

std::vector<double> MeshFunctionPWConst::operator()(
    const lf::mesh::Entity &cell, const Eigen::MatrixXd &local) const {
  LF_VERIFY_MSG(cell.RefEl() == lf::base::RefEl::kTria() ||
                    cell.RefEl() == lf::base::RefEl::kQuad(),
                "Unsupported entity type " << cell.RefEl());
  // Number of "evaluastion points"
  const size_t no_pts = local.cols();
  // Obtain dof index for the cell
  std::span<const lf::assemble::gdof_idx_t> locdofs{
      dofh_.GlobalDofIndices(cell)};
  LF_ASSERT_MSG(locdofs.size() == 1, "One dof must belong to every cell");
  // Copy cell value into result vector
  const std::vector<double> res(no_pts, coeffs_[locdofs[0]]);
  return res;
}

std::vector<Eigen::Vector2d> MeshFunctionWF1::operator()(
    const lf::mesh::Entity &cell, const Eigen::MatrixXd &local) const {
  LF_VERIFY_MSG(cell.RefEl() == lf::base::RefEl::kTria(),
                "Unsupported entity type " << cell.RefEl());
  LF_VERIFY_MSG(cell.Geometry()->isAffine(),
                "Triangle must have straight edges");
  // Compute gradients of barycentric coordinate functions and store them in the
  // columns of the $2\times 3$-matrix G
  // clang-format off
  const Eigen::MatrixXd dpt = (Eigen::MatrixXd(2, 1) << 0.0, 0.0).finished();
  const Eigen::Matrix<double, 2, 3> G =
      cell.Geometry()->JacobianInverseGramian(dpt).block(0, 0, 2, 2) *
    (Eigen::Matrix<double, 2, 3>(2,3) << -1, 1, 0,
                                         -1, 0, 1).finished();
  // clang-format on

  // Lambda functions in terms of reference coordinates for local Whitney
  // 1-forms, see \prbcref{eq:lsf1}.
  const std::array<std::function<Eigen::Vector2d(Eigen::Vector2d)>, 3> beta{
      [&G](Eigen::Vector2d c) -> Eigen::Vector2d {
        return (1.0 - c[0] - c[1]) * G.col(1) - c[0] * G.col(0);
      },
      [&G](Eigen::Vector2d c) -> Eigen::Vector2d {
        return c[0] * G.col(2) - c[1] * G.col(1);
      },
      [&G](Eigen::Vector2d c) -> Eigen::Vector2d {
        return c[1] * G.col(0) - (1.0 - c[0] - c[1]) * G.col(2);
      }};
  //  Obtain local d.o.f.s
  std::span<const lf::assemble::gdof_idx_t> locdofs{
      dofh_.GlobalDofIndices(cell)};
  LF_ASSERT_MSG(locdofs.size() == 3, "Three dofs must belong to every cell");
  // Whitney 1-forms (edge basis functions)
  Eigen::Vector3d wf1ldofs(coeffs_[locdofs[0]], coeffs_[locdofs[1]],
                           coeffs_[locdofs[2]]);
  // Correct for orientation mismatch
  auto relor = cell.RelativeOrientations();
  LF_ASSERT_MSG(relor.size() == 3, "Triangle should have 3 edges!?");
  for (int k = 0; k < 3; ++k) {
    if (relor[k] == lf::mesh::Orientation::negative) {
      // Flip sign of coefficient
      wf1ldofs[k] *= -1;
    }
  }
  std::vector<Eigen::Vector2d> res;
  // Run through evaluation points (columns of argument matrix)
  const size_t no_pts = local.cols();
  for (int j = 0; j < no_pts; ++j) {
    res.emplace_back(wf1ldofs[0] * beta[0](local.col(j)) +
                     wf1ldofs[1] * beta[1](local.col(j)) +
                     wf1ldofs[2] * beta[2](local.col(j)));
  }
  return res;
}

void testCvgMOLLeapfrog(unsigned int refsteps) {
  using namespace std::numbers;
  // ********** Part I: Manufactured solution **********
  // Domain: unit square
  const double eps = sqrt2 * pi;
  const double mu = sqrt2 * pi;
  // Solution for x3-component of the electric field
  auto e = [](Eigen::Vector2d x, double t) -> double {
    return std::sin(pi * x[0]) * std::sin(pi * x[1]) * std::sin(t);
  };
  // Solution for transversal component of magnetic field
  auto h = [mu](Eigen::Vector2d x, double t) -> Eigen::Vector2d {
    return -pi / mu * std::cos(t) *
           Eigen::Vector2d(std::sin(pi * x[0]) * std::cos(pi * x[1]),
                           -std::cos(pi * x[0]) * std::sin(pi * x[1]));
  };
  // Note: right hand source function is zero
  // Mesh functions for constant coefficients
  lf::mesh::utils::MeshFunctionConstant mf_eps(eps);
  lf::mesh::utils::MeshFunctionConstant mf_mu(mu);

  // ********** Part II: Loop over sequence of meshes **********
  // Generate a small unstructured triangular mesh
  const std::shared_ptr<lf::mesh::Mesh> mesh_ptr =
      lf::mesh::test_utils::GenerateHybrid2DTestMesh(3, 1.0 / 3.0);
  const std::shared_ptr<lf::refinement::MeshHierarchy> multi_mesh_p =
      lf::refinement::GenerateMeshHierarchyByUniformRefinemnt(mesh_ptr,
                                                              refsteps);
  lf::refinement::MeshHierarchy &multi_mesh{*multi_mesh_p};
  // Ouput summary information about hierarchy of nested meshes
  std::cout << "\t Sequence of nested meshes created\n";
  multi_mesh.PrintInfo(std::cout);

  // Number of levels
  const int L = multi_mesh.NumLevels();
  // Table of error norms
  std::vector<std::tuple<size_t, double, double>> L2errs;
  for (int level = 0; level < L; ++level) {
    const std::shared_ptr<const lf::mesh::Mesh> mesh_p =
        multi_mesh.getMesh(level);
    const lf::mesh::Mesh &mesh = *mesh_p;
    // Initialization of DofHandler objects
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
    // Dimensions of finite element spaces
    const size_t N2 = dofh_e.NumDofs();
    const size_t N1 = dofh_h.NumDofs();
    // Build matrix B
    Eigen::SparseMatrix<double> B = MaxwellEvlTM::buildB(dofh_e, dofh_h);
    // Build matrix $\VM_{\epsilon}$ for constant coefficient
    Eigen::SparseMatrix<double> Meps = MaxwellEvlTM::buildMeps(dofh_e, mf_eps);
    // Build matrix $\VM_{\mu}$ for constant coefficient $\mu=\mu(\Bx)$
    Eigen::SparseMatrix<double> Mmu = MaxwellEvlTM::buildMmu(dofh_h, mf_mu);
    // Dummy/zero right hand side functor
    auto j_zero = [N2](double t) -> Eigen::VectorXd {
      return Eigen::VectorXd::Constant(N2, 0.0);
    };
    // Interpolate initial values
    const Eigen::VectorXd e0_vec = nodalProjectionPWConst(
        dofh_e, lf::mesh::utils::MeshFunctionGlobal(
                    [&e](Eigen::Vector2d x) -> double { return e(x, 0.0); }));
    const Eigen::VectorXd h0_vec = nodalProjectionWF1(
        dofh_h,
        lf::mesh::utils::MeshFunctionGlobal(
            [&h](Eigen::Vector2d x) -> Eigen::Vector2d { return h(x, 0.0); }));
    // Leapfrog timestepping, M timesteps until some final time
    const double T_fin = 10.0;
    const unsigned int M = 100 * std::pow(2, level);
    // Record total energy in each timestep
    std::vector<double> E_tot{};
    auto rec = [&Meps, &Mmu, &E_tot](const Eigen::VectorXd &e_vec,
                                     const Eigen::VectorXd &h_vec) -> void {
      const double E_el = 0.5 * e_vec.transpose() * Meps * e_vec;
      const double E_mag = 0.5 * h_vec.transpose() * Mmu * h_vec;
      E_tot.push_back(E_el + E_mag);
    };
    auto [e_sol, h_sol] =
        leapfrogETM(Meps, Mmu, B, j_zero, e0_vec, h0_vec, M, T_fin, rec);
    //
  }
}

}  // namespace MaxwellEvlTM
