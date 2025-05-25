/**
 * @file hodgelaplacian2d.h
 * @brief NPDE homework HodgeLaplacian2D code
 * @author
 * @date
 * @copyright Developed at SAM, ETH Zurich
 */

#ifndef HodgeLaplacian2D_H_
#define HodgeLaplacian2D_H_

// Include almost all parts of LehrFEM++; soem my not be needed
#include <lf/assemble/assemble.h>
#include <lf/assemble/assembler.h>
#include <lf/assemble/dofhandler.h>
#include <lf/base/lf_assert.h>
#include <lf/fe/fe.h>
#include <lf/geometry/geometry_interface.h>
#include <lf/io/io.h>
#include <lf/mesh/entity.h>
#include <lf/mesh/hybrid2d/hybrid2d.h>
#include <lf/mesh/utils/utils.h>
#include <lf/uscalfe/uscalfe.h>

#include <Eigen/Core>
#include <Eigen/Sparse>

namespace HodgeLaplacian2D {
/**
 * @brief Element matrix provider for monolithic Whitney finite element
 * discretization of the Hodge Laplacian generalized saddle point problem.
 *
 * This class fits the concept of ELEMENT_MATRIX_PROVIDER
 */

/* SAM_LISTING_BEGIN_1 */
class HodgeLaplacian2DElementMatrixProvider {
 public:
  // The size of the element matrix is $6\times 6$.
  using ElemMat = Eigen::Matrix<double, 6, 6>;
  HodgeLaplacian2DElementMatrixProvider(
      const HodgeLaplacian2DElementMatrixProvider &) = delete;
  HodgeLaplacian2DElementMatrixProvider(
      HodgeLaplacian2DElementMatrixProvider &&) noexcept = default;
  HodgeLaplacian2DElementMatrixProvider &operator=(
      const HodgeLaplacian2DElementMatrixProvider &) = delete;
  HodgeLaplacian2DElementMatrixProvider &operator=(
      HodgeLaplacian2DElementMatrixProvider &&) = delete;
  HodgeLaplacian2DElementMatrixProvider() = default;
  virtual ~HodgeLaplacian2DElementMatrixProvider() = default;
  [[nodiscard]] bool isActive(const lf::mesh::Entity & /*cell*/) {
    return true;
  }
  // Computation of element matrix $\VM_K$: two versions
  [[nodiscard]] ElemMat Eval(const lf::mesh::Entity &cell);
  [[nodiscard]] ElemMat Eval_ref(const lf::mesh::Entity &cell);

 private:
  ElemMat MK_;
};
/* SAM_LISTING_END_1 */

/**
 * @brief ENTITY_VECTOR_PROVIDER class for source terms
 * in Whitney FEM for Hodge-Laplacian mixed variational problem
 *
 */
/* SAM_LISTING_BEGIN_7 */
template <lf::mesh::utils::MeshFunction MESH_FUNCTION>
class HodgeLaplacian2DElementVectorProvider {
 public:
  using ElemVec = Eigen::Matrix<double, 6, 1>;
  HodgeLaplacian2DElementVectorProvider(
      const HodgeLaplacian2DElementVectorProvider &) = delete;
  HodgeLaplacian2DElementVectorProvider(
      HodgeLaplacian2DElementVectorProvider &&) noexcept = default;
  HodgeLaplacian2DElementVectorProvider &operator=(
      const HodgeLaplacian2DElementVectorProvider &) = delete;
  HodgeLaplacian2DElementVectorProvider &operator=(
      HodgeLaplacian2DElementVectorProvider &&) = delete;
  virtual ~HodgeLaplacian2DElementVectorProvider() = default;

  HodgeLaplacian2DElementVectorProvider(MESH_FUNCTION f) : f_(f) {}
  [[nodiscard]] bool isActive(const lf::mesh::Entity & /*cell*/) {
    return true;
  }
  // Computation of element matrix $\VM_K$: two versions
  [[nodiscard]] ElemVec Eval(const lf::mesh::Entity &cell);

 private:
  ElemVec phiK_;
  MESH_FUNCTION f_;
};
/* SAM_LISTING_END_7 */

/* SAM_LISTING_BEGIN_8 */
template <lf::mesh::utils::MeshFunction MESH_FUNCTION>
HodgeLaplacian2DElementVectorProvider<MESH_FUNCTION>::ElemVec
HodgeLaplacian2DElementVectorProvider<MESH_FUNCTION>::Eval(
    const lf::mesh::Entity &cell) {
  LF_VERIFY_MSG(cell.RefEl() == lf::base::RefEl::kTria(),
                "Unsupported cell type " << cell.RefEl());
  // Area of the triangle
  double area = lf::geometry::Volume(*cell.Geometry());
  // Compute gradients of barycentric coordinate functions, see
  // \lref{cpp:gradbarycoords}. Get vertices of the triangle
  auto endpoints = lf::geometry::Corners(*(cell.Geometry()));
  Eigen::Matrix<double, 3, 3> X;  // temporary matrix
  X.block<3, 1>(0, 0) = Eigen::Vector3d::Ones();
  X.block<3, 2>(0, 1) = endpoints.transpose();
  // This matrix contains $\cob{\grad \lambda_i}$ in its columns.
  // Note that $\grad\lambda_i$ is accessed as G.col(i-1).
  const auto G{X.inverse().block<2, 3>(1, 0)};
#if SOLUTION
  // Matrix collecting the \textbf{reference coordinates} of the midpoints of
  // the edges of the triangle
  // clang-format off
    const Eigen::MatrixXd mp{(Eigen::MatrixXd(2, 3) <<
			      0.5, 0.5, 0.0,
			      0.0, 0.5, 0.5).finished()};
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
  // Obtain values of $\Vf$ at midpoints of edges
  const std::vector<Eigen::Vector2d> f_vals{f_(cell, mp)};
  LF_VERIFY_MSG(f_vals.size() == 3, "Too few f values");
  // Fill entries of element vetor. Note that the first three components remain
  // set to zero, because the r.h.s. functional vanishes for the 0-form
  // component of the test function
  phiK_.setZero();
  for (int i = 0; i < 3; ++i) {
    // Iterate over quadrature nodes (midpoints of edges)
    for (int k = 0; k < 3; ++k) {
      phiK_[i + 3] += beta[i](mp.col(k)).dot(f_vals[k]);
    }
  }
  // Multiply with quadrature weight
  phiK_ *= (area / 3.0);
  // Correct for orientation mismatch
  auto relor = cell.RelativeOrientations();
  LF_ASSERT_MSG(relor.size() == 3, "Triangle should have 3 edges!?");
  for (int k = 0; k < 3; ++k) {
    if (relor[k] == lf::mesh::Orientation::negative) {
      // Flip sign of coefficient
      phiK_[k + 3] *= -1;
    }
  }
#else
  /* **********************************************************************
     Your code here
     ********************************************************************** */
#endif
  return phiK_;
}
/* SAM_LISTING_END_8 */

/**
 * @brief Compute right-hand side vector for Hodge-Laplacian
 * Whitney FEM
 *
 * @param dofh DofHandler for monolithic Whitney finite element space for
 * Hodge-Laplacian mixed variational problem
 * @param f MeshFunction providing source vector field
 */
template <lf::mesh::utils::MeshFunction MESH_FUNCTION>
Eigen::VectorXd computeHodgeLaplaceRhsVector(
    const lf::assemble::DofHandler &dofh, MESH_FUNCTION f) {
  // Total number of FE d.o.f.s
  lf::assemble::size_type N = dofh.NumDofs();
  // Right-hand side vector
  Eigen::VectorXd rhs(N);
  // Object in charge of computing the element vectors
  HodgeLaplacian2DElementVectorProvider<MESH_FUNCTION> hlevp(f);
  // Carry out assembly
  rhs.setZero();
  lf::assemble::AssembleVectorLocally(0, dofh, hlevp, rhs);
  return rhs;
}

/**
 * @brief Assembly of full Galerkin matrix in triplet format
 *
 * @param dofh DofHandler object  for all FE spaces
 */
lf::assemble::COOMatrix<double> buildHodgeLaplacianGalerkinMatrix(
    const lf::assemble::DofHandler &dofh);

/**
 * @brief Compute Whitney FEM solution of Hodge Laplace BVP
 *
 * @param dofh DofHandler for monolithic Whitney finite element space for
 * Hodge-Laplacian mixed variational problem
 * @param f MeshFunction providing source vector field
 */
template <lf::mesh::utils::MeshFunction MESH_FUNCTION>
Eigen::VectorXd solveHodgeLaplaceBVP(const lf::assemble::DofHandler &dofh,
                                     MESH_FUNCTION f) {
  // Galerkin matrix in COO format
  lf::assemble::COOMatrix<double> M_COO{
      buildHodgeLaplacianGalerkinMatrix(dofh)};
  const Eigen::SparseMatrix<double> M_crs = M_COO.makeSparse();
  // Right-hand side vector
  const Eigen::VectorXd phi{
      computeHodgeLaplaceRhsVector<MESH_FUNCTION>(dofh, f)};

  // Solve linear system using Eigen's sparse direct elimination
  // Examine return status of solver in case the matrix is singular
  Eigen::SparseLU<Eigen::SparseMatrix<double>> solver;
  solver.compute(M_crs);
  LF_VERIFY_MSG(solver.info() == Eigen::Success, "LU decomposition failed");
  const Eigen::VectorXd dofvec = solver.solve(phi);
  LF_VERIFY_MSG(solver.info() == Eigen::Success, "Solving LSE failed");

  return dofvec;
}

/** @brief Mesh function providing a proxy vector field for Whitney 1-forms in
 * 2D
 *
 */
class MeshFunctionWF1 {
 public:
  MeshFunctionWF1(lf::assemble::DofHandler &dofh, Eigen::VectorXd coeffs)
      : dofh_(dofh), coeffs_(std::move(coeffs)) {
    LF_ASSERT_MSG(dofh_.NumDofs() == coeffs_.size(),
                  "Size mismatch for coeff vector");
  }

  // Evaluation operator: returns the values of the vectorfield in the space of
  // Whitney 1-forms at a number of points inside a cell
  std::vector<Eigen::Vector2d> operator()(const lf::mesh::Entity &cell,
                                          const Eigen::MatrixXd &local) const;

 private:
  lf::assemble::DofHandler &dofh_;
  Eigen::VectorXd coeffs_;
};

}  // namespace HodgeLaplacian2D

#endif
