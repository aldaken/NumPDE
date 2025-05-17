/**
 * @ file stokesminielement_main.cc
 * @ brief NPDE homework TEMPLATE MAIN FILE
 * @ author
 * @ date
 * @ copyright Developed at SAM, ETH Zurich
 */

#include <iostream>

#include "stokesminielement.h"

int main(int /*argc*/, char** /*argv*/) {
  std::cout << "NumPDE homework problem StokesMINIElement\n";
  lf::base::LehrFemInfo::PrintInfo(std::cout);

  std::cout << "test1\n";
  // Test of convergence of simple FEM on the unit square with manufactured
  // solution
  StokesMINIElement::testCvgSimpleFEM(4);
  std::cout << "test2\n";
  // Test of convergence of MINI FEM on the unit square with manufactured
  // solution
  StokesMINIElement::testCvgMINIFEM(4);
  std::cout << "test3\n";

  return 0;
}
