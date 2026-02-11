/**
 * @file maxwellevltm.h
 * @brief NPDE homework MaxwellEvlTM code
 * @author
 * @date
 * @copyright Developed at SAM, ETH Zurich
 */

#ifndef MaxwellEvlTM_H_
#define MaxwellEvlTM_H_

// Include almost all parts of LehrFEM++; soem my not be needed
#include <lf/assemble/assemble.h>
#include <lf/assemble/assembler.h>
#include <lf/assemble/assembly_types.h>
#include <lf/assemble/coomatrix.h>
#include <lf/assemble/dofhandler.h>
#include <lf/base/lf_assert.h>
#include <lf/fe/fe.h>
#include <lf/geometry/geometry_interface.h>
#include <lf/io/io.h>
#include <lf/mesh/hybrid2d/hybrid2d.h>
#include <lf/mesh/utils/utils.h>
#include <lf/uscalfe/uscalfe.h>

#include <Eigen/Core>
#include <Eigen/Sparse>
#include <iostream>

namespace MaxwellEvlTM {
/**
 * @brief Element matrix provider for the matrix block B
 *
 * This class fits the concept of ELEMENT_MATRIX_PROVIDER
 */
/* SAM_LISTING_BEGIN_1 */
class B_EMP {
 public:
  // The size of the element matrix is $3\times 1$.
  using ElemMat = Eigen::Matrix<double, 3, 1>;
  B_EMP(const B_EMP &) = delete;
  B_EMP(B_EMP &&) noexcept = default;
  B_EMP &operator=(const B_EMP &) = delete;
  B_EMP &operator=(B_EMP &&) = delete;
  B_EMP() = default;
  virtual ~B_EMP() = default;
  [[nodiscard]] bool isActive(const lf::mesh::Entity & /*cell*/) {
    return true;
  }
  // Computation of element matrix $\VM_K$: two versions
  [[nodiscard]] ElemMat Eval(const lf::mesh::Entity &cell);

 private:
  ElemMat BK_;
};
/* SAM_LISTING_END_1 */

/**
 * @brief Element matrix provider for the matrix block M_eps
 *
 * This class fits the concept of ELEMENT_MATRIX_PROVIDER
 */
/* SAM_LISTING_BEGIN_2 */
template <lf::mesh::utils::MeshFunction MESH_FUNCTION>
class Meps_EMP {
 public:
  // The size of the element ``matrix'' is $1\times 1$.
  using ElemMat = Eigen::Matrix<double, 1, 1>;
  Meps_EMP(const Meps_EMP &) = delete;
  Meps_EMP(Meps_EMP &&) noexcept = default;
  Meps_EMP &operator=(const Meps_EMP &) = delete;
  Meps_EMP &operator=(Meps_EMP &&) = delete;
  Meps_EMP() = default;
  virtual ~Meps_EMP() = default;

  Meps_EMP(MESH_FUNCTION epsilon) : epsilon_(epsilon) {}
  [[nodiscard]] bool isActive(const lf::mesh::Entity & /*cell*/) {
    return true;
  }
  // Computation of element matrix $\VM_K$: two versions
  [[nodiscard]] ElemMat Eval(const lf::mesh::Entity &cell);

 private:
  ElemMat MK_;
  MESH_FUNCTION epsilon_;
};
/* SAM_LISTING_END_2 */

template <lf::mesh::utils::MeshFunction MESH_FUNCTION>
typename Meps_EMP<MESH_FUNCTION>::ElemMat Meps_EMP<MESH_FUNCTION>::Eval(
    const lf::mesh::Entity &cell) {
  LF_VERIFY_MSG(cell.RefEl() == lf::base::RefEl::kTria(),
                "Unsupported cell type " << cell.RefEl());
  // Area of the triangle
  double area = lf::geometry::Volume(*cell.Geometry());
  // Reference coordinates for the center of gravity
  const Eigen::MatrixXd center{
      (Eigen::MatrixXd(2, 1) << 1.0 / 3.0, 1.0 / 3.0).finished()};
  // Obtain value of coefficient
  const std::vector<double> eps_val_center = epsilon_(cell, center);
  // $1\times 1$ element matrix: area * coefficient value
  MK_(0, 0) = area * eps_val_center[0];
  return MK_;
}

/**
 * @brief Element matrix provider for the matrix block M_mu
 *
 * This class fits the concept of ELEMENT_MATRIX_PROVIDER
 */
/* SAM_LISTING_BEGIN_3 */
template <lf::mesh::utils::MeshFunction MESH_FUNCTION>
class Mmu_EMP {
 public:
  // The size of the element matrix is $3\times 3$.
  using ElemMat = Eigen::Matrix<double, 3, 3>;
  Mmu_EMP(const Mmu_EMP &) = delete;
  Mmu_EMP(Mmu_EMP &&) noexcept = default;
  Mmu_EMP &operator=(const Mmu_EMP &) = delete;
  Mmu_EMP &operator=(Mmu_EMP &&) = delete;
  Mmu_EMP() = default;
  virtual ~Mmu_EMP() = default;

  Mmu_EMP(MESH_FUNCTION mu) : mu_(mu) {}
  [[nodiscard]] bool isActive(const lf::mesh::Entity & /*cell*/) {
    return true;
  }
  // Computation of element matrix $\VM_K$: two versions
  [[nodiscard]] ElemMat Eval(const lf::mesh::Entity &cell);

 private:
  ElemMat MK_;
  MESH_FUNCTION mu_;
};
/* SAM_LISTING_END_3 */

/* SAM_LISTING_BEGIN_4 */
template <lf::mesh::utils::MeshFunction MESH_FUNCTION>
typename Mmu_EMP<MESH_FUNCTION>::ElemMat Mmu_EMP<MESH_FUNCTION>::Eval(
    const lf::mesh::Entity &cell) {
  LF_VERIFY_MSG(cell.RefEl() == lf::base::RefEl::kTria(),
                "Unsupported cell type " << cell.RefEl());
  LF_VERIFY_MSG(cell.Geometry()->isAffine(),
                "Triangle must have straight edges");
  // Area of the triangle
  double area = lf::geometry::Volume(*cell.Geometry());
  // Compute gradients of barycentric coordinate functions and store them in the
  // columns of the $2\times 3$-matrix G
  // clang-format off
    const Eigen::MatrixXd dpt = (Eigen::MatrixXd(2, 1) << 0.0, 0.0).finished();
    const Eigen::Matrix<double, 2, 3> G =
      cell.Geometry()->JacobianInverseGramian(dpt).block(0, 0, 2, 2) *
      (Eigen::Matrix<double, 2, 3>(2,3) << -1, 1, 0,
       -1, 0, 1).finished();
  // clang-format on
  // This function uses the 3-point local quadrature rule whose nodes coincide
  // with the midpoints of the edges of the triangle. Values (*2) of local
  // Whitney 1-forms in midpoints of edges
  std::array<Eigen::Matrix<double, 2, 3>, 3> betavals;
  // betavals[index of midpoints].col(index of basis form)
  betavals[0].col(0) = 0.5 * (G.col(1) - G.col(0));
  betavals[0].col(1) = 0.5 * (G.col(2));
  betavals[0].col(2) = 0.5 * (-G.col(2));
  betavals[1].col(0) = 0.5 * (-G.col(0));
  betavals[1].col(1) = 0.5 * (G.col(2) - G.col(1));
  betavals[1].col(2) = 0.5 * (G.col(0));
  betavals[2].col(0) = 0.5 * (G.col(1));
  betavals[2].col(1) = 0.5 * (-G.col(1));
  betavals[2].col(2) = 0.5 * (G.col(0) - G.col(2));
  // Reference coordinates for midpoints of edges
  // clang-format off
    const Eigen::MatrixXd mpc{
      (Eigen::MatrixXd(2, 3) << 0.5, 0.5, 0.0,
       0.0, 0.5, 0.5).finished()};
  // clang-format on
  std::vector<double> mu_vals = mu_(cell, mpc);
  // Run through midpoints of edges
  MK_.setZero();
  for (unsigned int k = 0; k < 3; ++k) {
    MK_ += mu_vals[k] * betavals[k].transpose() * betavals[k];
  }
  // Correct for orientation mismatch
  auto relor = cell.RelativeOrientations();
  LF_ASSERT_MSG(relor.size() == 3, "Triangle should have 3 edges!?");
  for (int k = 0; k < 3; ++k) {
    if (relor[k] == lf::mesh::Orientation::negative) {
      // Flip sign of 3+k-th rown and column
      MK_.col(k) *= -1.0;
      MK_.row(k) *= -1.0;
    }
  }
  // Finally multiply with quadracdture weight
  return 1.0 / 3.0 * area * MK_;
}
/* SAM_LISTING_END_4 */

/**
 * @brief ENTITY_VECTOR_PROVIDER class for source term and p.w. constant FE
 * space
 */
/* SAM_LISTING_BEGIN_7 */
template <lf::mesh::utils::MeshFunction MESH_FUNCTION>
class phi_EVP {
 public:
  using ElemVec = Eigen::Matrix<double, 1, 1>;
  phi_EVP(const phi_EVP &) = delete;
  phi_EVP(phi_EVP &&) noexcept = default;
  phi_EVP &operator=(const phi_EVP &) = delete;
  phi_EVP &operator=(phi_EVP &&) = delete;
  virtual ~phi_EVP() = default;

  phi_EVP(MESH_FUNCTION f) : f_(f) {}
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

template <lf::mesh::utils::MeshFunction MESH_FUNCTION>
typename phi_EVP<MESH_FUNCTION>::ElemVec phi_EVP<MESH_FUNCTION>::Eval(
    const lf::mesh::Entity &cell) {
  LF_VERIFY_MSG(cell.RefEl() == lf::base::RefEl::kTria(),
                "Unsupported cell type " << cell.RefEl());
  // Area of the triangle
  double area = lf::geometry::Volume(*cell.Geometry());
  // Reference coordinates for the center of gravity
  const Eigen::MatrixXd center{
      (Eigen::MatrixXd(2, 1) << 1.0 / 3.0, 1.0 / 3.0).finished()};
  // Obtain value of source current $j$ and center of triangle
  const std::vector<double> j_val_center = f_(cell, center);
  // $1\times 1$ element matrix: area * coefficient value
  phiK_(0, 0) = area * j_val_center[0];
  return phiK_;
}

/** @brief Building the matrix $\VM_{\epsilon}$
 */
template <lf::mesh::utils::MeshFunction MESH_FUNCTION>
Eigen::SparseMatrix<double> buildMeps(const lf::assemble::DofHandler &dofh_e,
                                      MESH_FUNCTION epsilon) {
  // Obtain mesh
  std::shared_ptr<const lf::mesh::Mesh> mesh_p = dofh_e.Mesh();
  const lf::mesh::Mesh &mesh = *mesh_p;
  LF_ASSERT_MSG((dofh_e.NumDofs() == mesh.NumEntities(0)),
                "DofH must manage 1 dof per cell");
  // Temporary triplet matrix
  const lf::assemble::size_type N = dofh_e.NumDofs();
  lf::assemble::COOMatrix<double> Meps_COO(N, N);
  // ENTITY_MATRIX_PROVIDER object
  Meps_EMP<MESH_FUNCTION> Meps_emp{epsilon};
  // Actual cell oriented assembly
  lf::assemble::AssembleMatrixLocally(0, dofh_e, dofh_e, Meps_emp, Meps_COO);
  return Meps_COO.makeSparse();
}

/** @brief Building the matrix $\VM_{\mu}$
 */
template <lf::mesh::utils::MeshFunction MESH_FUNCTION>
Eigen::SparseMatrix<double> buildMmu(const lf::assemble::DofHandler &dofh_h,
                                     MESH_FUNCTION mu) {
  // Obtain mesh
  std::shared_ptr<const lf::mesh::Mesh> mesh_p = dofh_h.Mesh();
  const lf::mesh::Mesh &mesh = *mesh_p;
  LF_ASSERT_MSG((dofh_h.NumDofs() == mesh.NumEntities(1)),
                "DofH must manage 1 dof per edge");
  // Temporary triplet matrix
  const lf::assemble::size_type N = dofh_h.NumDofs();
  lf::assemble::COOMatrix<double> Mmu_COO(N, N);
  // ENTITY_MATRIX_PROVIDER object
  Mmu_EMP<MESH_FUNCTION> Mmu_emp{mu};
  // Actual cell oriented assembly
  lf::assemble::AssembleMatrixLocally(0, dofh_h, dofh_h, Mmu_emp, Mmu_COO);
  return Mmu_COO.makeSparse();
}

/** @brief Building matrix $\VB$
 */
Eigen::SparseMatrix<double> buildB(const lf::assemble::DofHandler &dofh_e,
                                   const lf::assemble::DofHandler &dofh_h);
/** @brief Setting up right-hand side vector
 */
template <lf::mesh::utils::MeshFunction MESH_FUNCTION>
Eigen::VectorXd buildPhi(const lf::assemble::DofHandler &dofh_e,
                         MESH_FUNCTION j) {
  // Obtain mesh
  std::shared_ptr<const lf::mesh::Mesh> mesh_p = dofh_e.Mesh();
  const lf::mesh::Mesh &mesh = *mesh_p;
  LF_ASSERT_MSG((dofh_e.NumDofs() == mesh.NumEntities(0)),
                "DofH must manage 1 dof per cell");
  // Right-hand-side vector
  Eigen::VectorXd phi_vec(dofh_e.NumDofs());
  phi_vec.setZero();
  // ENTITY_VECTOR_PROVIDER object
  phi_EVP<MESH_FUNCTION> phi_evp{j};
  // Building the right-hand-side vector
  lf::assemble::AssembleVectorLocally(0, dofh_e, phi_evp, phi_vec);
  return phi_vec;
}

/** @brief Leapfrog timestepping
 */
/* SAM_LISTING_BEGIN_9 */
template <typename RHSVECFUNCTOR,
          typename RECORDER = std::function<void(const Eigen::VectorXd &,
                                                 const Eigen::VectorXd &)>>
std::pair<Eigen::VectorXd, Eigen::VectorXd> leapfrogETM(
    const Eigen::SparseMatrix<double> &Meps,
    const Eigen::SparseMatrix<double> &Mmu,
    const Eigen::SparseMatrix<double> &B, RHSVECFUNCTOR rhs,
    const Eigen::VectorXd &e0, const Eigen::VectorXd &h0, unsigned int M,
    double T_final,
    RECORDER rec = [](const Eigen::VectorXd &e_vec,
                      const Eigen::VectorXd &h_vec) -> void {}) {
  const size_t N2 = Meps.rows();
  const size_t N1 = Mmu.rows();
  LF_ASSERT_MSG(Meps.cols() == Meps.rows(), "Meps must be square");
  LF_ASSERT_MSG(Mmu.cols() == Mmu.rows(), "Mmu must be square");
  LF_ASSERT_MSG(Meps.cols() == B.cols(), "Size mismatch B and Mesp");
  LF_ASSERT_MSG(Mmu.rows() == B.rows(), "Size mismatch B and Mmu");
  LF_ASSERT_MSG(e0.size() == N2, "E0.size must be equal to number of cells");
  LF_ASSERT_MSG(h0.size() == N1, "h0.size must be equal to number of edges");
  // For the sake of efficiency performs LU decomposition of
  // $\VM_{\epsilon}$ and $\VM_{\mu}$ before entering the timestepping loop!
  Eigen::SparseLU<Eigen::SparseMatrix<double>> solver_Meps;
  solver_Meps.compute(Meps);
  Eigen::SparseLU<Eigen::SparseMatrix<double>> solver_Mmu;
  solver_Mmu.compute(Mmu);
  // Auxiliary vectors holding states during timestepping
  rec(e0, h0);
  Eigen::VectorXd h_old = h0;
  Eigen::VectorXd h_new(N1);
  Eigen::VectorXd e_old(N2);
  Eigen::VectorXd e_new = e0;
  Eigen::VectorXd h_j(N1);
  const double tau = T_final / M;
  // Initial step
  h_new = 0.5 * tau * solver_Mmu.solve(-B * e0) + h0;
  // Main leapfrog timestepping loop
  double tj = 0.5 * tau;
  for (int j = 1; j <= M; ++j, tj += tau) {
    e_old = e_new;
    h_old = h_new;
    const Eigen::VectorXd rhs_vec = rhs(tj);
    LF_ASSERT_MSG(rhs_vec.size() == N2, "Size mismatch for r.h.s. vector");
    e_new = tau * solver_Meps.solve(rhs_vec + B.transpose() * h_old) + e_old;
    h_new = tau * solver_Mmu.solve(-B * e_new) + h_old;
    h_j = 0.5 * (h_new + h_old);
    rec(e_new, h_j);
  }
  // Final step
  return {e_new, h_j};
}
/* SAM_LISTING_END_9 */

/** @brief Mesh function providing a proxy vector field for Whitney 1-forms in
 * 2D.
 *
 * The DofHandler is supposed to manage solely edge-based d.o.f.s
 *
 */
class MeshFunctionPWConst {
 public:
  MeshFunctionPWConst(const lf::assemble::DofHandler &dofh,
                      Eigen::VectorXd coeffs)
      : dofh_(dofh), coeffs_(std::move(coeffs)) {
    const lf::mesh::Mesh &mesh = *dofh.Mesh();
    LF_ASSERT_MSG(dofh_.NumDofs() == coeffs_.size(),
                  "Size mismatch for coeff vector");
    LF_ASSERT_MSG(dofh.NumDofs() == mesh.NumEntities(2),
                  "DofH must manage 1 dof/cell");
  }

  // Evaluation operator: emulates p.w. constant function
  std::vector<double> operator()(const lf::mesh::Entity &cell,
                                 const Eigen::MatrixXd &local) const;

 private:
  const lf::assemble::DofHandler &dofh_;
  Eigen::VectorXd coeffs_;
};

/** @brief Mesh function providing a proxy vector field for Whitney 1-forms in
 * 2D.
 *
 * The DofHandler is supposed to manage solely edge-based d.o.f.s
 *
 */
class MeshFunctionWF1 {
 public:
  MeshFunctionWF1(const lf::assemble::DofHandler &dofh, Eigen::VectorXd coeffs)
      : dofh_(dofh), coeffs_(std::move(coeffs)) {
    const lf::mesh::Mesh &mesh = *dofh.Mesh();
    LF_ASSERT_MSG(dofh_.NumDofs() == coeffs_.size(),
                  "Size mismatch for coeff vector");
    LF_ASSERT_MSG(dofh.NumDofs() == mesh.NumEntities(1),
                  "DofH must manage 1 dof/edge");
  }

  // Evaluation operator: returns the values of the vectorfield in the space of
  // Whitney 1-forms at a number of points inside a cell
  std::vector<Eigen::Vector2d> operator()(const lf::mesh::Entity &cell,
                                          const Eigen::MatrixXd &local) const;

 private:
  const lf::assemble::DofHandler &dofh_;
  Eigen::VectorXd coeffs_;
};

/** @brief Approximate "nodal interpolation" of a vectorfield into the
 * Whitney finite element space of 1-forms using Option I 2D Euclidean
 * vector proxies.
 *
 * Implementation for DofHandler managing edge-based dofs.
 *
 * @param DofHandler for edge finite element space
 */
template <lf::mesh::utils::MeshFunction MESH_FUNCTION>
Eigen::VectorXd nodalProjectionWF1(const lf::assemble::DofHandler &dofh,
                                   MESH_FUNCTION vf) {
  const lf::mesh::Mesh &mesh = *dofh.Mesh();
  const Eigen::Index N = dofh.NumDofs();
  LF_ASSERT_MSG(N == mesh.NumEntities(1), "DofH must manage 1 dof/edge");
  Eigen::VectorXd np_coeffs(N);
  np_coeffs.setZero();
  // Reference coordinates of midpoint of edge
  Eigen::MatrixXd mpc(1, 1);
  mpc(0, 0) = 0.5;
  for (const lf::mesh::Entity *edge : mesh.Entities(1)) {
    // Obtain number of global d.o.f. assciated with edge
    std::span<const lf::assemble::gdof_idx_t> edofs{
        dofh.InteriorGlobalDofIndices(*edge)};
    LF_ASSERT_MSG(edofs.size() == 1, "Only one d.o.f. per edge is allowed");
    const Eigen::Matrix2d endpt = lf::geometry::Corners(*edge->Geometry());
    const Eigen::Vector2d edvec = endpt.col(1) - endpt.col(0);
    // Midpoint quadrature rule:
    const auto vf_vals{vf(*edge, mpc)};
    np_coeffs[edofs[0]] = edvec.dot(vf_vals[0]);
  }
  return np_coeffs;
}

/** @brief Interpolation of a function into space of p.w. constants
 *
 * Implementation for DofHandler managing cell-based dofs.
 *
 * @param DofHandler for p.w. constant FE space
 */
template <lf::mesh::utils::MeshFunction MESH_FUNCTION>
Eigen::VectorXd nodalProjectionPWConst(const lf::assemble::DofHandler &dofh,
                                       MESH_FUNCTION vf) {
  const lf::mesh::Mesh &mesh = *dofh.Mesh();
  const Eigen::Index N = dofh.NumDofs();
  LF_ASSERT_MSG(N == mesh.NumEntities(0), "DofH must manage 1 dof/cell");
  Eigen::VectorXd np_coeffs(N);
  np_coeffs.setZero();
  // Reference coordinates of center of cell
  Eigen::MatrixXd mpc =
      (Eigen::MatrixXd(2, 1) << 1.0 / 3.0, 1.0 / 3.0).finished();
  for (const lf::mesh::Entity *cell : mesh.Entities(0)) {
    // Obtain number of global d.o.f. assciated with a cell
    std::span<const lf::assemble::gdof_idx_t> cdofs{
        dofh.InteriorGlobalDofIndices(*cell)};
    LF_ASSERT_MSG(cdofs.size() == 1, "Only one d.o.f. per cell is allowed");
    // Sample function at the center of the cell
    const auto vf_vals{vf(*cell, mpc)};
    np_coeffs[cdofs[0]] = vf_vals[0];
  }
  return np_coeffs;
}

/** @brief Convergence test with manufactured solution
 *
 * Computes L2 norm of error of the MOL solution at final time
 * for s sequence of meshes generated by regular refinement.
 */
void testCvgMOLLeapfrog(unsigned int refsteps);

}  // namespace MaxwellEvlTM

#endif
