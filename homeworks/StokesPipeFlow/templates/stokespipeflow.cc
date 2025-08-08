/**
 * @file stokespipeflow.cc
 * @brief NPDE homework StokesPipeFlow code
 * @ author Wouter Tonnon
 * @ date May 2024
 * @ copyright Developed at SAM, ETH Zurich
 */

#include "stokespipeflow.h"

#include <Eigen/src/Core/Matrix.h>
#include <Eigen/src/Core/util/Meta.h>
#include <lf/assemble/assembler.h>
#include <lf/assemble/assembly_types.h>
#include <lf/assemble/coomatrix.h>
#include <lf/assemble/dofhandler.h>
#include <lf/base/lf_assert.h>
#include <lf/base/ref_el.h>
#include <lf/fe/fe_tools.h>
#include <lf/fe/mesh_function_grad_fe.h>
#include <lf/geometry/geometry_interface.h>
#include <lf/mesh/entity.h>
#include <lf/mesh/test_utils/test_meshes.h>
#include <lf/mesh/utils/mesh_function_global.h>
#include <lf/mesh/utils/mesh_function_unary.h>

#include <cstddef>
#include <iomanip>

namespace StokesPipeFlow {
/* SAM_LISTING_BEGIN_1 */
TaylorHoodElementMatrixProvider::ElemMat TaylorHoodElementMatrixProvider::Eval(
    const lf::mesh::Entity& cell) {
  LF_VERIFY_MSG(cell.RefEl() == lf::base::RefEl::kTria(),
                "Unsupported cell type " << cell.RefEl());
  // Area of the triangle
  double area = lf::geometry::Volume(*cell.Geometry());
  // Compute gradients of barycentric coordinate functions, see
  // \lref{cpp:gradbarycoords} Get vertices of the triangle
  auto endpoints = lf::geometry::Corners(*(cell.Geometry()));
  Eigen::Matrix<double, 3, 3> X;  // temporary matrix
  X.block<3, 1>(0, 0) = Eigen::Vector3d::Ones();
  X.block<3, 2>(0, 1) = endpoints.transpose();
  // This matrix contains $\cob{\grad \lambda_i}$ in its columns
  const auto G{X.inverse().block<2, 3>(1, 0)};
/* **********************************************************************
   Your code here
   ********************************************************************** */
  // !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
  // Workaround for a side-effect in LehrFEM++
  // Local shape functions associated with edges have to be swapped, if the edge
  // has "negative" orientation with respect to the triangle. This is meant to
  // offset such a swapping done by LehrFEM++'s UniformFEDofHandler.
  auto edge_orientations = cell.RelativeOrientations();
  LF_ASSERT_MSG(edge_orientations.size() == 3,
                "Triangle should have 3 edges!?");
  Eigen::VectorXd tmp(15);
  for (int k = 0; k < 3; ++k) {
    const lf::assemble::ldof_idx_t ed_dofx = 9 + 2 * k;
    const lf::assemble::ldof_idx_t ed_dofy = 10 + 2 * k;
    if (edge_orientations[k] == lf::mesh::Orientation::negative) {
      // The rows and columns of the element matrix with numbers 9+2*k and
      // 9+2*k+1 have to be swapped.
      tmp = AK_.col(ed_dofx);
      AK_.col(ed_dofx) = AK_.col(ed_dofy);
      AK_.col(ed_dofy) = tmp;
      tmp = AK_.row(ed_dofx);
      AK_.row(ed_dofx) = AK_.row(ed_dofy);
      AK_.row(ed_dofy) = tmp;
    }
  }
  // !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
  // Finally multiply with the quadrature weight
  return AK_;
}
/* SAM_LISTING_END_1 */

/* SAM_LISTING_BEGIN_3 */
lf::assemble::COOMatrix<double> buildTaylorHoodGalerkinMatrix(
    const lf::assemble::DofHandler& dofh) {
  // Total number of FE d.o.f.s without Lagrangian multiplier
  lf::assemble::size_type n = dofh.NumDofs();
  // Full Galerkin matrix in triplet format taking into account the zero mean
  // constraint on the pressure.
  lf::assemble::COOMatrix<double> A(n + 1, n + 1);
  // Set up computation of element matrix
  TaylorHoodElementMatrixProvider themp{};
  // Assemble \cor{full} Galerkin matrix for Taylor-Hood FEM
  lf::assemble::AssembleMatrixLocally(0, dofh, dofh, themp, A);

// Add bottom row and right column corresponding to Lagrange multiplier
// You cannot use AssembleMatrixLocally() because the DofHandler does
// not know about this extra unknown.
// Do cell-oriented assembly "manually"
/* **********************************************************************
   Your code here
   ********************************************************************** */
  // Rely on return value optimization
  return A;
}
/* SAM_LISTING_END_3 */

/**
 * @brief Compute dissipated power for a Talor-Hood FE solution
 *
 * @param dofh DofHandler object for monolithic Taylor-Hood FEM
 * @param mu_vec basis expansion coefficient vector
 */
/* SAM_LISTING_BEGIN_8 */
double compDissPowVolume(const lf::assemble::DofHandler& dofh,
                         const Eigen::VectorXd& mu_vec) {
  // First fetch underlying mesh
  const lf::mesh::Mesh& mesh{*dofh.Mesh()};
  // Summation variable
  double p_diss{0.0};
  // Loop over all cells and add up contributions of local integrals to the
  // dissipated power.
  for (const lf::mesh::Entity* cell : mesh.Entities(0)) {
    LF_ASSERT_MSG(cell->RefEl() == lf::base::RefEl::kTria(),
                  "Only implemented for triangles");
    // Obtain coefficients for local shape functions
    // Local indices for LSFs fpr velocity components
    std::array<Eigen::Index, 6> vx_idx{0, 3, 6, 9, 11, 13};
    std::array<Eigen::Index, 6> vy_idx{1, 4, 7, 10, 12, 14};
    // !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    // Workaround for a side-effect in LehrFEM++
    // Local shape functions associated with edges have to be swapped, if the
    // edge has "negative" orientation with respect to the triangle. This is
    // meant to offset such a swapping done by LehrFEM++'s UniformFEDofHandler.
    auto edge_orientations = cell->RelativeOrientations();
    LF_ASSERT_MSG(edge_orientations.size() == 3,
                  "Triangle should have 3 edges!?");
    for (int k = 0; k < 3; ++k) {
      if (edge_orientations[k] == lf::mesh::Orientation::negative) {
        // The rows and columns of the element matrix with numbers 9+2*k and
        // 9+2*k+1 have to be swapped.
        std::swap(vx_idx[3 + k], vy_idx[3 + k]);
      }
    }
    // !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    std::span<const lf::assemble::gdof_idx_t> dof_idx{
        dofh.GlobalDofIndices(*cell)};
    LF_ASSERT_MSG(dof_idx.size() == 15, "Taylor-Hood FEM involves 15 LSFs!");

    // Copied from Eval(); Found no good way of moving this to a function
    // Area of the triangle
    double area = lf::geometry::Volume(*cell->Geometry());
    // Compute gradients of barycentric coordinate functions, see
    // \lref{cpp:gradbarycoords} Get vertices of the triangle
    auto endpoints = lf::geometry::Corners(*(cell->Geometry()));
    Eigen::Matrix<double, 3, 3> X;  // temporary matrix
    X.block<3, 1>(0, 0) = Eigen::Vector3d::Ones();
    X.block<3, 2>(0, 1) = endpoints.transpose();
    // This matrix contains $\cob{\grad \lambda_i}$ in its columns
    const auto G{X.inverse().block<2, 3>(1, 0)};
    // Gradients of local shape functions of quadratic Lagrangian finite element
    // space as lambda functions, see \prbeqref{eq:quadlsf}
    std::array<std::function<Eigen::Vector2d(Eigen::Vector3d)>, 6> gradq{
        [&G](Eigen::Vector3d c) -> Eigen::Vector2d {
          return (4 * c[0] - 1) * G.col(0);
        },
        [&G](Eigen::Vector3d c) -> Eigen::Vector2d {
          return (4 * c[1] - 1) * G.col(1);
        },
        [&G](Eigen::Vector3d c) -> Eigen::Vector2d {
          return (4 * c[2] - 1) * G.col(2);
        },
        [&G](Eigen::Vector3d c) -> Eigen::Vector2d {
          return 4 * (c[0] * G.col(1) + c[1] * G.col(0));
        },
        [&G](Eigen::Vector3d c) -> Eigen::Vector2d {
          return 4 * (c[1] * G.col(2) + c[2] * G.col(1));
        },
        [&G](Eigen::Vector3d c) -> Eigen::Vector2d {
          return 4 * (c[2] * G.col(0) + c[0] * G.col(2));
        }};
    // Barycentric coordinates of the midpoints of the edges for
    // use with the 3-point edge midpoint quadrature rule \prbeqref{eq:MPR}
    // Note that we only integrate a quadratic polynomial so that the edge
    // midpoint quadrature rule evaluates the integral exactly.
    const std::array<Eigen::Vector3d, 3> mp = {Eigen::Vector3d({0.5, 0.5, 0}),
                                               Eigen::Vector3d({0, 0.5, 0.5}),
                                               Eigen::Vector3d({0.5, 0, 0.5})};
/* **********************************************************************
   Your code here
   ********************************************************************** */
  }
  return p_diss;
}
/* SAM_LISTING_END_8 */

/**
 * @brief Computes dissipated power by boundary-based formula
 *
 * @param dofh DofHandler object for monolithic Taylor-Hood FEM
 * @param mu_vec basis expansion coefficient vector
 *
 * This implementation is valid only in the special geometric setting of the
 * pipe flow model for the homework project StokesPipeFLow.
 */
/* SAM_LISTING_BEGIN_9 */
double compDissPowBd(const lf::assemble::DofHandler& dofh,
                     const Eigen::VectorXd& muvec, bool print) {
  // First fetch underlying mesh
  const lf::mesh::Mesh& mesh{*dofh.Mesh()};
  // Summation variable
  double p_diss{0.0};
// Loop over the edges and check whether they are located on the inlet $x_1=0$
// or outlet $x_1=1$.
/* **********************************************************************
   Your code here
   ********************************************************************** */
  return p_diss;
}
/* SAM_LISTING_END_9 */

double allPipeFlow(PowerFlag powerflag, bool producevtk, const char* meshfile,
                   const char* outfile) {
  LF_VERIFY_MSG((meshfile != nullptr), "Must provide a mesh file");
  const std::string meshfile_str = std::string(meshfile);

  /* SAM_LISTING_BEGIN_5 */
  auto mesh_factory = std::make_unique<lf::mesh::hybrid2d::MeshFactory>(2);
  lf::io::GmshReader reader(std::move(mesh_factory), meshfile_str);

  const std::shared_ptr<const lf::mesh::Mesh> mesh_ptr = reader.mesh();
  const lf::mesh::Mesh& mesh{*mesh_ptr};

  // Initialize dof handler for Taylor-Hood FEM
  lf::assemble::UniformFEDofHandler dofh(mesh_ptr,
                                         {{lf::base::RefEl::kPoint(), 3},
                                          {lf::base::RefEl::kSegment(), 2},
                                          {lf::base::RefEl::kTria(), 0},
                                          {lf::base::RefEl::kQuad(), 0}});
// We define function providing the boundary values for the velocity
  /* **********************************************************************
     Replace this placeholder code
     ********************************************************************** */
  auto g = [](Eigen::Vector2d /*x*/) -> Eigen::Vector2d { return {1.0, 1.0}; };
  // Solve the system
  Eigen::VectorXd res = StokesPipeFlow::solvePipeFlow(dofh, g);

  double p_diss = 0.0;
  switch (powerflag) {
    case VOLUME: {
      p_diss = compDissPowVolume(dofh, res);
      break;
    }
    case BOUNDARY: {
      p_diss = compDissPowBd(dofh, res);
      break;
    }
    default: {
      std::cout << "Dissipated power not computed\n";
    }
  }

  std::cout << "Dissipated power: " << p_diss << std::endl;

  if (producevtk) {
    LF_VERIFY_MSG(outfile != nullptr,
                  "Filename for .vtk files has to be provided");
    // Define first- and second-order Lagrangian FE spaces for the piecewise
    // linear Taylor-Hood pressure approximation and the piecewise quadratic
    auto fes_o1_ptr =
        std::make_shared<lf::uscalfe::FeSpaceLagrangeO1<double>>(mesh_ptr);
    auto fes_o2_ptr =
        std::make_shared<lf::uscalfe::FeSpaceLagrangeO2<double>>(mesh_ptr);
    // Fetch dof handler for the components of the velocity
    const lf::assemble::DofHandler& dofh_u = fes_o2_ptr->LocGlobMap();
    //  Fetch dof handler for the pressure p
    const lf::assemble::DofHandler& dofh_p = fes_o1_ptr->LocGlobMap();

    // Coefficient vectors for the first and second component of the velocity
    Eigen::VectorXd coeff_vec_u1(dofh_u.NumDofs());
    Eigen::VectorXd coeff_vec_u2(dofh_u.NumDofs());
    // Coefficient vector for the pressure
    Eigen::VectorXd coeff_vec_p(dofh_p.NumDofs());

    // Loop over vertices and edges to remap the basis expansion coefficients
    for (int codim = 2; codim >= 1; codim--) {
      for (auto e : mesh.Entities(codim)) {
        // Global indices for u1, u2 for the respective vertex or edge
        auto glob_idxs = dofh.GlobalDofIndices(*e);
        auto glob_idx_o2 = dofh_u.GlobalDofIndices(*e)[0];

        // Extract the correct elements for the coefficient matrix of the
        // components of u
        coeff_vec_u1(glob_idx_o2) = res(glob_idxs[0]);
        coeff_vec_u2(glob_idx_o2) = res(glob_idxs[1]);

        // Global indices for p for the respective vertex
        lf::assemble::gdof_idx_t glob_idx_o1 = -1;
        // The pressure is only defined on vertices
        if (codim == 2) {
          glob_idx_o1 = dofh_p.GlobalDofIndices(*e)[0];
          coeff_vec_p(glob_idx_o1) = res(glob_idxs[2]);
        }
      }
    }

    // Define finite-element mesh functions
    lf::fe::MeshFunctionFE<double, double> mf_o2_u1(fes_o2_ptr, coeff_vec_u1);
    lf::fe::MeshFunctionFE<double, double> mf_o2_u2(fes_o2_ptr, coeff_vec_u2);
    lf::fe::MeshFunctionFE<double, double> mf_o1_p(fes_o1_ptr, coeff_vec_p);

    const std::string outfile_str = std::string(outfile);
    std::cout << outfile_str << std::endl;
    lf::io::VtkWriter vtk_writer(mesh_ptr, outfile_str);
    vtk_writer.WritePointData("u1", mf_o2_u1);
    vtk_writer.WritePointData("u2", mf_o2_u2);
    vtk_writer.WritePointData("p", mf_o1_p);
  }
  /* SAM_LISTING_END_5 */
  return p_diss;
}

void visualizeTHPipeFlow(const char* meshfile, const char* outfile) {
  (void)allPipeFlow(NOCMOP, true, meshfile, outfile);
}

double computeDissipatedPower(const char* meshfile) {
  return allPipeFlow(VOLUME, false, meshfile);
}

double computeDissipatedPowerBd(const char* meshfile) {
  return allPipeFlow(BOUNDARY, false, meshfile);
}

/* SAM_LISTING_BEGIN_7 */
void testCvgTaylorHood(unsigned int refsteps) {
  // Analytic solution for velocity and pressure
  Eigen::Vector2d d{2.0, 1.0};
  auto v_ex = [&d](Eigen::Vector2d x) -> Eigen::Vector2d {
    x += Eigen::Vector2d(0.5, 0.5);
    const double nx = x.norm();
    return 0.25 * (-std::log(nx) * d + x * (x.dot(d)) / (nx * nx));
  };
  auto p_ex = [&d](Eigen::Vector2d x) -> double {
    x += Eigen::Vector2d(0.5, 0.5);
    return -0.5 * x.dot(d) / x.squaredNorm() + (d[0] + d[1]) * 0.5 * 0.502128;
  };
  auto grad_v1 = [&d](Eigen::Vector2d x) -> Eigen::Vector2d {
    x += Eigen::Vector2d(0.5, 0.5);
    const double x2 = x[0] * x[0];
    const double x3 = x2 * x[0];
    const double y2 = x[1] * x[1];
    const double den = (x2 + y2) * (x2 + y2);
    return 0.25 * d[0] *
               Eigen::Vector2d{(-x3 + x[0] * y2) / den,
                               -x[1] * (3 * x2 + y2) / den} +
           0.25 * d[1] *
               Eigen::Vector2d{x[1] * (-x2 + y2) / den, x[0] * (x2 - y2) / den};
  };
  auto grad_v2 = [&d](Eigen::Vector2d x) -> Eigen::Vector2d {
    x += Eigen::Vector2d(0.5, 0.5);
    const double x2 = x[0] * x[0];
    const double x3 = x2 * x[0];
    const double y2 = x[1] * x[1];
    const double den = (x2 + y2) * (x2 + y2);
    return 0.25 * d[1] *
               Eigen::Vector2d{-x[0] * (x2 + 3 * y2) / den,
                               x[1] * (x2 - y2) / den} +
           0.25 * d[0] *
               Eigen::Vector2d{x[1] * (-x2 + y2) / den, x[0] * (x2 - y2) / den};
  };

  // Generate a small unstructured triangular mesh of the unit square
  const std::shared_ptr<lf::mesh::Mesh> mesh_ptr =
      lf::mesh::test_utils::GenerateHybrid2DTestMesh(3, 1.0 / 3.0);
  // Generate sequence of meshes by uniform regular refinement
  const std::shared_ptr<lf::refinement::MeshHierarchy> multi_mesh_p =
      lf::refinement::GenerateMeshHierarchyByUniformRefinemnt(mesh_ptr,
                                                              refsteps);
  lf::refinement::MeshHierarchy& multi_mesh{*multi_mesh_p};
  // Ouput summary information about hierarchy of nested meshes
  std::cout << "\t Sequence of nested meshes created\n";
  multi_mesh.PrintInfo(std::cout);

  // Number of levels
  const int L = multi_mesh.NumLevels();

  // Table of various error norms
  std::vector<std::tuple<size_t, double, double, double, double, double>> errs;
  // Loop over the levels
  for (int level = 0; level < L; ++level) {
    const std::shared_ptr<const lf::mesh::Mesh> lev_mesh_p =
        multi_mesh.getMesh(level);
    // Initialize dof handler for Taylor-Hood FEM
    lf::assemble::UniformFEDofHandler dofh(lev_mesh_p,
                                           {{lf::base::RefEl::kPoint(), 3},
                                            {lf::base::RefEl::kSegment(), 2},
                                            {lf::base::RefEl::kTria(), 0},
                                            {lf::base::RefEl::kQuad(), 0}});
    // Define first- and second-order Lagrangian FE spaces for the piecewise
    // linear Taylor-Hood pressure approximation and the piecewise quadratic
    auto fes_o1_ptr =
        std::make_shared<lf::uscalfe::FeSpaceLagrangeO1<double>>(lev_mesh_p);
    auto fes_o2_ptr =
        std::make_shared<lf::uscalfe::FeSpaceLagrangeO2<double>>(lev_mesh_p);
    // Fetch dof handler for the components of the velocity
    const lf::assemble::DofHandler& dofh_u = fes_o2_ptr->LocGlobMap();
    //  Fetch dof handler for the pressure p
    const lf::assemble::DofHandler& dofh_p = fes_o1_ptr->LocGlobMap();
    LF_ASSERT_MSG(dofh.NumDofs() == 2 * dofh_u.NumDofs() + dofh_p.NumDofs(),
                  "No dof mismatch");
    // Solve the system with trace of the exact velocity solution as Dirichlet
    // data
    Eigen::VectorXd res = StokesPipeFlow::solvePipeFlow(dofh, v_ex);
    // Coefficient vectors for the first and second component of the velocity
    Eigen::VectorXd coeff_vec_u1 = Eigen::VectorXd::Zero(dofh_u.NumDofs());
    Eigen::VectorXd coeff_vec_u2 = Eigen::VectorXd::Zero(dofh_u.NumDofs());
    // Coefficient vector for the pressure
    Eigen::VectorXd coeff_vec_p = Eigen::VectorXd::Zero(dofh_p.NumDofs());

    // Remapping dofs: We first loop over vertices, then over edges
    for (int codim = 2; codim >= 1; codim--) {
      for (auto e : lev_mesh_p->Entities(codim)) {
        // Global indices for u1, u2 for the respective vertex or edge
        auto glob_idxs = dofh.InteriorGlobalDofIndices(*e);
        auto glob_idx_o2 = dofh_u.InteriorGlobalDofIndices(*e)[0];
        // Extract the correct elements for the coefficient matrix of the
        // components of u
        coeff_vec_u1[glob_idx_o2] = res[glob_idxs[0]];
        coeff_vec_u2[glob_idx_o2] = res[glob_idxs[1]];
        // The pressure is only defined on vertices
        if (codim == 2) {
          // Global indices for p for the respective vertex
          lf::assemble::gdof_idx_t glob_idx_o1 = dofh_p.GlobalDofIndices(*e)[0];
          coeff_vec_p(glob_idx_o1) = res[glob_idxs[2]];
        }
      }
    }
    // Variables for storing the error norms
    double L2err_u1, L2err_u2, H1err_u1, H1err_u2, L2err_p;
    /* **********************************************************************
       Your code here
       **********************************************************************
     */
    errs.emplace_back(dofh.NumDofs(), L2err_u1, L2err_u2, H1err_u1, H1err_u2,
                      L2err_p);
  }
  // Output table of errors to file and terminal
  std::ofstream out_file("errors.txt");
  std::cout.precision(3);
  std::cout << std::endl
            << std::left << std::setw(10) << "N" << std::right << std::setw(16)
            << "L2 err(v1)" << std::setw(16) << "L2 err(v2)" << std::setw(16)
            << "H1 err(v1)" << std::setw(16) << "H1 err(v2)" << std::setw(16)
            << "L2 err(p)" << '\n';
  std::cout << "---------------------------------------------" << '\n';
  for (const auto& err : errs) {
    auto [N, L2err_u1, L2err_u2, H1err_u1, H1err_u2, L2err_p] = err;
    out_file << std::left << std::setw(10) << N << std::left << std::setw(16)
             << L2err_u1 << std::setw(16) << L2err_u2 << std::setw(16)
             << H1err_u1 << std::setw(16) << H1err_u2 << std::setw(16)
             << L2err_p << '\n';
    std::cout << std::left << std::setw(10) << N << std::left << std::setw(16)
              << L2err_u1 << std::setw(16) << L2err_u2 << std::setw(16)
              << H1err_u1 << std::setw(16) << H1err_u2 << std::setw(16)
              << L2err_p << '\n';
  }
}
/* SAM_LISTING_END_7 */

}  // namespace StokesPipeFlow
