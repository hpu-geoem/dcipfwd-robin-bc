cmake_minimum_required(VERSION 3.20)

if(POLICY CMP0135)
  cmake_policy(SET CMP0135 NEW)
endif()

project(dcipfwd VERSION 1.0.0 LANGUAGES C CXX)

option(DCIPFWD_BUILD_TESTS "Build the unit tests (bundled Catch2)." ON)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_CONFIGURATION_TYPES AND NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

# MFEM built with MPI, hypre, MUMPS and GSLIB; its CMake package brings the libraries it was built with.
find_package(MPI REQUIRED COMPONENTS C CXX)
# MFEM_PREFIX (or MFEM_DIR, or the MFEM_DIR environment variable) is the MFEM installation prefix.
set(MFEM_PREFIX "" CACHE PATH "MFEM installation prefix")
if(NOT MFEM_PREFIX)
  if(MFEM_DIR)
    set(MFEM_PREFIX "${MFEM_DIR}")
  elseif(DEFINED ENV{MFEM_DIR})
    set(MFEM_PREFIX "$ENV{MFEM_DIR}")
  endif()
endif()
find_package(MFEM CONFIG QUIET HINTS "${MFEM_PREFIX}" "${MFEM_PREFIX}/lib/cmake/mfem")
if(NOT MFEM_FOUND)
  include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/MFEMConfigMk.cmake)
endif()
foreach(feature MFEM_USE_MPI MFEM_USE_MUMPS MFEM_USE_GSLIB)
  if(NOT ${feature})
    message(FATAL_ERROR "MFEM must be built with ${feature}.")
  endif()
endforeach()

include(FetchContent)
FetchContent_Declare(fmt URL "${CMAKE_CURRENT_SOURCE_DIR}/third_party/fmt-12.1.0.tar.gz")
FetchContent_MakeAvailable(fmt)

add_library(dcipfwd_core
  src/integral_transforms.cc
  src/dc_boundary.cc
  src/mesh_finder.cc
  src/model.cc
  src/survey.cc
  src/dc_solver.cc
  src/wavenumber_rule.cc
  src/octree_model.cc
)
configure_file(cmake/version.h.in ${CMAKE_CURRENT_BINARY_DIR}/include/version.h @ONLY)
target_include_directories(dcipfwd_core PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include ${CMAKE_CURRENT_BINARY_DIR}/include)
target_link_libraries(dcipfwd_core PUBLIC mfem fmt::fmt MPI::MPI_CXX)

add_executable(dcipfwd apps/dcipfwd.cc)
target_link_libraries(dcipfwd PRIVATE dcipfwd_core)

add_executable(octree_mesh apps/octree_mesh.cc)
target_link_libraries(octree_mesh PRIVATE dcipfwd_core)

if(DCIPFWD_BUILD_TESTS)
  enable_testing()
  FetchContent_Declare(catch2 URL "${CMAKE_CURRENT_SOURCE_DIR}/third_party/catch2-3.14.0.tar.gz")
  FetchContent_MakeAvailable(catch2)
  add_subdirectory(tests)
endif()
