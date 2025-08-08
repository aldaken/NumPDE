// Demonstration code for course Numerical Methods for Partial Differential
// Equations Author: R. Hiptmair, SAM, ETH Zurich Date: July 2025
// Related to Experiment 9.3.4.20 (Energy conservation for leapfrog)

#include "lfen.h"

int main(int /*argc*/, char** /*argv*/) {
  std::cout << "Energy tracking for leapfrog timestepping\n";
  const int n = 100;  // e.g. 10×10 interior grid
  const int m = 200;
  /*
  Eigen::MatrixXd A = LeapfrogWave::buildLaplacian2D(n);
  std::cout << "Poisson matrix = \n" << A/((n+1)*(n+1)) << std::endl;
  */
  LeapfrogWave::tabulate_energies(n, m, "lf_energies");
  return 0;
}
