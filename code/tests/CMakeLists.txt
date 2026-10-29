set(DCIPFWD_TEST_TIMEOUT 900 CACHE STRING "Timeout in seconds for MPI unit tests.")

# Build a Catch2 test with the MPI main and register runs on the given rank counts.
function(add_dcipfwd_test name source)
  add_executable(${name} ${source} catch2_mpi_main.cc)
  target_link_libraries(${name} PRIVATE dcipfwd_core Catch2::Catch2)
  foreach(ranks ${ARGN})
    add_test(NAME ${name}_${ranks}
             COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} ${ranks} ${MPIEXEC_PREFLAGS} $<TARGET_FILE:${name}>
                     ${MPIEXEC_POSTFLAGS})
    set_tests_properties(${name}_${ranks} PROPERTIES PROCESSORS ${ranks} TIMEOUT "${DCIPFWD_TEST_TIMEOUT}"
                                                    ENVIRONMENT "OMP_NUM_THREADS=1;OPENBLAS_NUM_THREADS=1")
  endforeach()
endfunction()

add_dcipfwd_test(test_integral_transforms integral_transforms_test.cc 1)
add_dcipfwd_test(test_dc_3d dc_3d_test.cc 2)
add_dcipfwd_test(test_dc_25d dc_25d_test.cc 2)
add_dcipfwd_test(test_ip ip_test.cc 2)
add_dcipfwd_test(test_model_survey model_survey_test.cc 1)
add_dcipfwd_test(test_dc_reference dc_reference_test.cc 2)
add_dcipfwd_test(test_wavenumber_rule wavenumber_rule_test.cc 1)
add_dcipfwd_test(test_octree_model octree_model_test.cc 2)
add_dcipfwd_test(test_spectral_bound spectral_bound_test.cc 2)
