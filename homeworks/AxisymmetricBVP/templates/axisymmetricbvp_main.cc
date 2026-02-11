/**
 * @ file axisymmetricbvp_main.cc
 * @ brief NPDE homework TEMPLATE MAIN FILE
 * @ author Ralf Hiptmair
 * @ date June 2025
 * @ copyright Developed at SAM, ETH Zurich
 */

#include <iostream>

#include "axisymmetricbvp.h"

int main(int /*argc*/, char** /*argv*/) {
  std::cout << "NumPDE homework problem AxisymmetricBVP\n";

  std::cout << "Solving heat conduction problem on 50x50 grid\n";
  const unsigned int n = 50;
  Eigen::VectorXd sol = AxisymmetricBVP::solveHeatBolt(
      n, [](double z) -> double { return (1.0 + z * z); },
      [](double z) -> double { return 2 * z; });
  std::cout << "Mean temperature = " << sol.sum() / (n * n - 1) << std::endl;

  return 0;
}
