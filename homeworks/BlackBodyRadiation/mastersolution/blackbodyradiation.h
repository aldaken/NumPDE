/**
 * @file blackbodyradiation.h
 * @brief NPDE homework BlackBodyRadiation code
 * @author
 * @date
 * @copyright Developed at SAM, ETH Zurich
 */

#ifndef BlackBodyRadiation_H_
#define BlackBodyRadiation_H_

// Include almost all parts of LehrFEM++; soem my not be needed
#include <lf/assemble/assemble.h>
#include <lf/assemble/dofhandler.h>
#include <lf/fe/fe.h>
#include <lf/geometry/geometry_interface.h>
#include <lf/io/io.h>
#include <lf/mesh/hybrid2d/hybrid2d.h>
#include <lf/mesh/utils/codim_mesh_data_set.h>
#include <lf/mesh/utils/utils.h>
#include <lf/uscalfe/uscalfe.h>

#include <Eigen/Core>
#include <Eigen/Sparse>
#include <iostream>

namespace BlackBodyRadiation {

/**
 * @brief ENTITY_VECTOR_PROVIDER class for Stefan-Boltzmann boundary flux
 * discretized by means of piecewise linear finite elements
 *
 */
/* SAM_LISTING_BEGIN_7 */
class StefanBoltzmannElementVectorProvider {
 public:
  using ElemVec = Eigen::Matrix<double, 2, 1>;
  StefanBoltzmannElementVectorProvider(
      const StefanBoltzmannElementVectorProvider &) = delete;
  StefanBoltzmannElementVectorProvider(
      StefanBoltzmannElementVectorProvider &&) noexcept = default;
  StefanBoltzmannElementVectorProvider &operator=(
      const StefanBoltzmannElementVectorProvider &) = delete;
  StefanBoltzmannElementVectorProvider &operator=(
      StefanBoltzmannElementVectorProvider &&) = delete;
  virtual ~StefanBoltzmannElementVectorProvider() = default;

  StefanBoltzmannElementVectorProvider(
      const lf::assemble::DofHandler &dofh, const Eigen::VectorXd &mu_vec,
      lf::mesh::utils::CodimMeshDataSet<bool> bd_flags)
      : dofh_(dofh), mu_vec_(mu_vec), bd_flags_(bd_flags) {}
  [[nodiscard]] bool isActive(const lf::mesh::Entity &edge) {
    return bd_flags_(edge);
  }
  [[nodiscard]] ElemVec Eval(const lf::mesh::Entity &edge);

 private:
  const lf::assemble::DofHandler &dofh_;
  const lf::mesh::utils::CodimMeshDataSet<bool> bd_flags_;
  const Eigen::VectorXd mu_vec_;
};
/* SAM_LISTING_END_7 */

}  // namespace BlackBodyRadiation

#endif
