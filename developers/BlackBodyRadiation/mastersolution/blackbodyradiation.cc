/**
 * @file blackbodyradiation.cc
 * @brief NPDE homework BlackBodyRadiation code
 * @author Ralf Hiptmair
 * @date June 2025
 * @copyright Developed at SAM, ETH Zurich
 */

#include "blackbodyradiation.h"

#include <Eigen/src/Core/Matrix.h>
#include <lf/assemble/assembly_types.h>
#include <lf/base/lf_assert.h>
#include <lf/base/ref_el.h>
#include <lf/base/types.h>
#include <lf/geometry/geometry_interface.h>
#include <lf/mesh/entity.h>
#include <lf/quad/quad.h>
#include <lf/quad/quad_rule.h>

namespace BlackBodyRadiation {
/* SAM_LISTING_BEGIN_1 */
StefanBoltzmannElementVectorProvider::ElemVec
StefanBoltzmannElementVectorProvider::Eval(const lf::mesh::Entity &edge) {
  LF_ASSERT_MSG(edge.RefEl() == lf::base::RefEl::kSegment(),
                "Only definied for edges");
  // To be filled with entries of the element vector
  ElemVec loc_vec = ElemVec::Zero();
  // Length of edge
  const double ed_len = lf::geometry::Volume(*edge.Geometry());
  // Obtain coefficients for local shape function expansion of $u_h$
  std::span<const lf::assemble::gdof_idx_t> loc_idx{
      dofh_.GlobalDofIndices(edge)};
  LF_ASSERT_MSG(loc_idx.size() == 2, "Every edge must bear two d.o.f.");
  const Eigen::Vector2d uh_coeff(mu_vec_[loc_idx[0]], mu_vec_[loc_idx[1]]);
#if SOLUTION
  // Retrieve quadrature rule of degree of exactness = 5
  lf::quad::QuadRule qr =
      lf::quad::make_QuadRule(lf::base::RefEl::kSegment(), 5);
  // Note that qr.Points is just a row vector with the reference coordinates of
  // the quadrature nodes, which agree with the values of the "right"
  // barycentric coordinate function in the quadature node
  const Eigen::MatrixXd &qp = qr.Points();
  const Eigen::VectorXd &qw = qr.Weights();
  LF_ASSERT_MSG(qp.rows() == 1, "1D reference coordinates for edge!");
  const lf::base::size_type P = qr.NumPoints();
  // Loop over quadrature points and sum contributions
  for (unsigned int j = 0; j < P; ++j) {
    // Evaluate barycentric coordinate functions in quadrature nodes
    const double l0_val = 1 - qp(0, j);
    const double l1_val = qp(0, j);
    // Evaluate $u_h$ in quadrature nodes
    const double uh_val = l0_val * uh_coeff[0] + l1_val * uh_coeff[1];
    // Update according to \prbcref{eq:vecpl}
    loc_vec += ed_len * qw[j] * std::pow(uh_val, 4) * ElemVec(l0_val, l1_val);
  }
#else
  /* **********************************************************************
     Your code here
     ********************************************************************** */
#endif
  return loc_vec;
}
/* SAM_LISTING_END_1 */

}  // namespace BlackBodyRadiation
