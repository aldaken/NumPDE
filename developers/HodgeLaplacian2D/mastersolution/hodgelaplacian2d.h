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
  [[nodiscard]] ElemMat Eval_alt(const lf::mesh::Entity &cell);

 private:
  ElemMat MK_;
};
/* SAM_LISTING_END_1 */

/**
 * @brief ENTITY_VECTOR_PROVIDER class for source terms
 * in Whitney FEM for Hodge-Laplacian mixed variational problem
 *
 */
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
  [[nodiscard]] bool isActive(const lf::mesh::Entity & /*cell*/) {
    return true;
  }
  // Computation of element matrix $\VM_K$: two versions
  [[nodiscard]] ElemVec Eval(const lf::mesh::Entity &cell);

 private:
  ElemVec phiK_;
  MESH_FUNCTION f_;
};

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
  // This matrix contains $\cob{\grad \lambda_i}$ in its columns
  const auto G{X.inverse().block<2, 3>(1, 0)};
#if SOLUTION
  // Dummy lambda functions for barycentric coordinates
  const std::array<std::function<double(Eigen::Vector3d)>, 3> lambda{
      [](Eigen::Vector3d c) -> double { return c[0]; },
      [](Eigen::Vector3d c) -> double { return c[1]; },
      [](Eigen::Vector3d c) -> double { return c[2]; }};
  // Lambda functions for local Whitney 1-forms, see \prbcref{eq:lsf1}.
  // Note that $\grad\lambda_i$ is accessed as G.col(i-1).
  const std::array<std::function<Eigen::Vector2d(Eigen::Vector3d)>, 3> beta{
      [&G](Eigen::Vector3d c) -> Eigen::Vector2d {
        return c[0] * G.col(1) - c[1] * G.col(0);
      },
      [&G](Eigen::Vector3d c) -> Eigen::Vector2d {
        return c[1] * G.col(2) - c[2] * G.col(1);
      },
      [&G](Eigen::Vector3d c) -> Eigen::Vector2d {
        return c[2] * G.col(0) - c[0] * G.col(2);
      }};
  // Barycentric coordinates of the midpoints of the edges for
  // use with the 3-point edge midpoint quadrature rule \prbeqref{eq:MPR}
  const std::array<Eigen::Vector3d, 3> mp = {Eigen::Vector3d({0.5, 0.5, 0}),
                                             Eigen::Vector3d({0, 0.5, 0.5}),
                                             Eigen::Vector3d({0.5, 0, 0.5})};
#else
/* **********************************************************************
   Your code here
   ********************************************************************** */
#endif
  return phiK_;
}

/**
 * @brief Compute right-hand side vector for Hodge-Laplacian
 * Whitney FEM
 *
 */
Eigen::VectorXd computeHodgeLaplaceRhsVector(
    const lf::assemble::DofHandler &dofh);

/**
 * @brief Assembly of full Galerkin matrix in triplet format
 *
 * @param dofh DofHandler object  for all FE spaces
 */
lf::assemble::COOMatrix<double> buildHodgeLaplacianGalerkinMatrix(
    const lf::assemble::DofHandler &dofh);

}  // namespace HodgeLaplacian2D

#endif
