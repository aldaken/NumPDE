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
  /* **********************************************************************
     Your code here
     ********************************************************************** */
  return loc_vec;
}
/* SAM_LISTING_END_1 */

}  // namespace BlackBodyRadiation
