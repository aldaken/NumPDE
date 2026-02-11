/**
 * @ file maxwellevltm_main.cc
 * @ brief NPDE homework TEMPLATE MAIN FILE
 * @ author Ralf Hiptmair
 * @ date June 2025
 * @ copyright Developed at SAM, ETH Zurich
 */

#include <iostream>

#include "maxwellevltm.h"

int main(int /*argc*/, char** /*argv*/) {
  std::cout << "NumPDE homework problem MaxwellEvlTM\n";
  MaxwellEvlTM::testCvgMOLLeapfrog(3);
  return 0;
}
