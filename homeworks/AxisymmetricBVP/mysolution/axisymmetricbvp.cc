/**
 * @file axisymmetricbvp.cc
 * @brief NPDE homework AxisymmetricBVP code
 * @author Ralf Hiptmair
 * @date June 2025
 * @copyright Developed at SAM, ETH Zurich
 */

#include "axisymmetricbvp.h"

namespace AxisymmetricBVP {

QuadRule1D::QuadRule1D(std::initializer_list<double> points,
                       std::initializer_list<double> weights)
    : points_(points.size()), weights_(weights.size()) {
  assertm(weights.size() == points.size(), "Size mismatch");
  int j = 0;
  for (double w : weights) {
    weights_[j++] = w;
  }
  j = 0;
  for (double zeta : points) {
    points_[j++] = zeta;
    ;
  }
}

QuadRule1D make_GaussRule(unsigned int n_pts) {
  switch (n_pts) {
    case 1: {
      return QuadRule1D({0.5}, {1.0});
    }
    case 2: {
      return QuadRule1D({0.21132486540518713, 0.78867513459481287}, {0.5, 0.5});
    }
    case 3: {
      return QuadRule1D(
          {0.11270166537925830, 0.50000000000000000, 0.88729833462074170},
          {0.27777777777777785, 0.44444444444444442, 0.27777777777777785});
    }
    case 4: {
      return QuadRule1D({0.06943184420297371, 0.33000947820757187,
                         0.66999052179242813, 0.93056815579702623},
                        {0.17392742256872684, 0.32607257743127310,
                         0.32607257743127310, 0.17392742256872684});
    }
    case 5: {
      return QuadRule1D(
          {0.04691007703066802, 0.23076534494715845, 0.50000000000000000,
           0.76923465505284150, 0.95308992296933193},
          {0.11846344252809471, 0.23931433524968310, 0.28444444444444450,
           0.23931433524968310, 0.11846344252809471});
    }
    case 6: {
      return QuadRule1D(
          {0.03376524289842397, 0.16939530676686776, 0.38069040695840151,
           0.61930959304159849, 0.83060469323313224, 0.96623475710157603},
          {0.08566224618958487, 0.18038078652406947, 0.23395696728634569,
           0.23395696728634569, 0.18038078652406947, 0.08566224618958487});
    }
    default: {
      assertm(false, "Quadrature rule not available");
      break;
    }
  };
  return QuadRule1D({}, {});
}

}  // namespace AxisymmetricBVP
