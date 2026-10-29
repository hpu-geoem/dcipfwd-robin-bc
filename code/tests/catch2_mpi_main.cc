#define CATCH_CONFIG_RUNNER

#include <catch2/catch_session.hpp>

#include <mfem.hpp>

#include "log.h"

int main(int argc, char **argv) {
  mfem::Mpi::Init(&argc, &argv);
  mfem::Hypre::Init();
  dcip::log_level() = 0;
  Catch::Session session;
  session.configData().runOrder = Catch::TestRunOrder::Declared;
  int result = session.applyCommandLine(argc, argv);
  if (result == 0) {
    result = session.run();
  }
  int finalized = 0;
  MPI_Finalized(&finalized);
  if (!finalized) {
    MPI_Finalize();
  }
  return result;
}
