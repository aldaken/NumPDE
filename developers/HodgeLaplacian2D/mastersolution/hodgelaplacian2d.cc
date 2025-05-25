/**
 * @file hodgelaplacian2d.cc
 * @brief NPDE homework HodgeLaplacian2D code
 * @author Ralf Hoptmair
 * @date May 2025
 * @copyright Developed at SAM, ETH Zurich
 */

#include "hodgelaplacian2d.h"

namespace HodgeLaplacian2D {
/* SAM_LISTING_BEGIN_1 */
HodgeLaplacian2DElementMatrixProvider::ElemMat
HodgeLaplacian2DElementMatrixProvider::Eval(const lf::mesh::Entity& cell) {
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
  // Compute the element matrix  for $-\Delta$ and $\cob{\Cs^0_1}$.
  // See also \lref{mc:ElementMatrixLaplLFE}.
  // Eigen::Matrix<double, 3, 3> L = area * G.transpose() * G;
  // Fill the four 3x3 blocks of $\VM_K$
  const double A_ent = 3.0 / (area * area);
  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 3; ++j) {
      // Left upper block $-\VC_K$:
      MK_(i, j) = -(lambda[i](mp[0]) * lambda[j](mp[0]) +
                    lambda[i](mp[1]) * lambda[j](mp[1]) +
                    lambda[i](mp[2]) * lambda[j](mp[2]));
      // Upper right block $\VB_K$ and lower left block $\VB_K^{\top}$:
      const double val =
          ((beta[j](mp[0])) + (beta[j](mp[1])) + (beta[j](mp[2])))
              .dot(G.col(i));
      MK_(i, j + 3) = val;
      MK_(j + 3, i) = val;
      // Lower right block
      MK_(i + 3, j + 3) = A_ent;
    }
  }
  // Correct for orientation mismatch, cf. \prbcref{sp:H}.
  auto relor = cell.RelativeOrientations();
  LF_ASSERT_MSG(relor.size() == 3, "Triangle should have 3 edges!?");
  for (int k = 0; k < 3; ++k) {
    if (relor[k] == lf::mesh::Orientation::negative) {
      // Flip sign of 3+k-th rown and column
      MK_.col(3 + k) *= -1.0;
      MK_.row(3 + k) *= -1.0;
    }
  }
  // Finally multiply with with the quadrature weight.
  MK_ *= area / 3.0;
#else
/* **********************************************************************
   Your code here
   ********************************************************************** */
#endif
  return MK_;
}
/* SAM_LISTING_END_1 */

/* SAM_LISTING_BEGIN_2 */
HodgeLaplacian2DElementMatrixProvider::ElemMat
HodgeLaplacian2DElementMatrixProvider::Eval_alt(const lf::mesh::Entity& cell) {
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
  // Compute the element matrix  for $-\Delta$ and $\cob{\Cs^0_1}$.
  // See also \lref{mc:ElementMatrixLaplLFE}.
  Eigen::Matrix<double, 3, 3> L = area * G.transpose() * G;

  // Fill left upper block $\VC_K$, \prbcref{eq:Mk}
  const Eigen::Matrix3d C =
      (Eigen::Matrix3d() << 2, 1, 1, 1, 2, 1, 1, 1, 2).finished();
  MK_.block(0, 0, 3, 3) = -(area / 12.0) * C;
  // Initialize right upper block $\VB_K$ and left lower block $\VB^\top$,
  // see \prbcref{eq:BKdef}.
  Eigen::Matrix3d B;
  B.col(0) = (L.col(1) - L.col(0)) / 3.0;
  B.col(1) = (L.col(2) - L.col(1)) / 3.0;
  B.col(2) = (L.col(0) - L.col(2)) / 3.0;
  MK_.block(0, 3, 3, 3) = B;
  MK_.block(3, 0, 3, 3) = B.transpose();
  // Set lower right block $\VA_K$, see \prbcref{ae:Akrefc}
  MK_.block(3, 3, 3, 3) = Eigen::Matrix3d::Constant(1.0 / area);
  // Correct for orientation mismatch, cf. \prbcref{sp:H}.
  auto relor = cell.RelativeOrientations();
  LF_ASSERT_MSG(relor.size() == 3, "Triangle should have 3 edges!?");
  for (int k = 0; k < 3; ++k) {
    if (relor[k] == lf::mesh::Orientation::negative) {
      // Flip sign of 3+k-th rown and column
      MK_.col(3 + k) *= -1.0;
      MK_.row(3 + k) *= -1.0;
    }
  }
#else
/* **********************************************************************
   Your code here
   ********************************************************************** */
#endif
  return MK_;
}
/* SAM_LISTING_END_2 */

Eigen::VectorXd computeHodgeLaplaceRhsVector(
    const lf::assemble::DofHandler& dofh) {
  // Total number of FE d.o.f.s
  lf::assemble::size_type N = dofh.NumDofs();
  // Right-hand side vector
  Eigen::VectorXd rhs(N);

  return rhs;
}

lf::assemble::COOMatrix<double> buildHodgeLaplacianGalerkinMatrix(
    const lf::assemble::DofHandler& dofh) {
  // Total number of FE d.o.f.s
  lf::assemble::size_type N = dofh.NumDofs();
  // Full Galerkin matrix in triplet format
  lf::assemble::COOMatrix<double> A(N, N);
#if SOLUTION
  // Set up computation of element matrix
  HodgeLaplacian2DElementMatrixProvider hlemp{};
  // Assemble \cor{full} Galerkin matrix for Whitney FEM for
  // Hodge-Laplacian mixed variational problem
  lf::assemble::AssembleMatrixLocally(0, dofh, dofh, hlemp, A);
#else
  /* **********************************************************************
     Your code here
     ********************************************************************** */
#endif
  return A;
}

}  // namespace HodgeLaplacian2D
