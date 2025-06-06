/**
 * @ file hodgelaplacian2d_main.cc
 * @ brief NPDE homework HodgeLaplacian2D
 * @ author Ralf Hiptmair
 * @ date May 2025
 * @ copyright Developed at SAM, ETH Zurich
 */

#include <lf/fe/fe_tools.h>
#include <lf/io/vtk_writer.h>
#include <lf/mesh/utils/mesh_function_unary.h>

#include <iostream>
#include <numbers>

#include "hodgelaplacian2d.h"

int main(int /*argc*/, char** /*argv*/) {
  std::cout << "NumPDE homework problem HodgeLaplacian2D\n";
  std::cout << "Created by R. Hiptmair, May 2025\n";
  lf::base::LehrFemInfo::PrintInfo(std::cout);

  // Empiric convergence study
  HodgeLaplacian2D::testCvgHLWhitneyFEM(4);

  // VTK output of solution
  const char no = '1';
  std::string meshfile = std::string("meshes/unitsquare") + no + ".msh";
  std::cout << "Reading mesh from file " << meshfile << std::endl;

  auto mesh_factory = std::make_unique<lf::mesh::hybrid2d::MeshFactory>(2);
  lf::io::GmshReader reader(std::move(mesh_factory), meshfile);

  const std::shared_ptr<const lf::mesh::Mesh> mesh_ptr = reader.mesh();
  const lf::mesh::Mesh& mesh{*mesh_ptr};

  // Initialize dof handler for monolithic Whitney FEM
  lf::assemble::UniformFEDofHandler dofh(mesh_ptr,
                                         {{lf::base::RefEl::kPoint(), 1},
                                          {lf::base::RefEl::kSegment(), 1},
                                          {lf::base::RefEl::kTria(), 0},
                                          {lf::base::RefEl::kQuad(), 0}});
  // Define source vectorfield
  auto f = [](Eigen::Vector2d x) -> Eigen::Vector2d {
    return Eigen::Vector2d(1.0, 2.0);
  };
  lf::mesh::utils::MeshFunctionGlobal mf_f(f);
  // Compute basis expansion coefficient vector of solution
  Eigen::VectorXd sol = HodgeLaplacian2D::solveHodgeLaplaceBVP(dofh, mf_f);
  std::cout << "Solution computed\n";
  // Compute L2-norm of 1-form component of solution
  HodgeLaplacian2D::MeshFunctionWF1 mf_sol(dofh, sol);
  double L2norm = std::sqrt(lf::fe::IntegrateMeshFunction(
      mesh, lf::mesh::utils::squaredNorm(mf_sol), 2));
  std::cout << "L2-norm of 1-form solution = " << L2norm << std::endl;
  auto [p_dat, u_dat] = HodgeLaplacian2D::reconstructNodalFields(dofh, sol);
  lf::io::VtkWriter vtk_writer(mesh_ptr, "2dhl.vtk");
  vtk_writer.WritePointData("p", p_dat);
  vtk_writer.WritePointData("u", u_dat);
  std::cout << "VTK file written\n";
  return 0;
}
