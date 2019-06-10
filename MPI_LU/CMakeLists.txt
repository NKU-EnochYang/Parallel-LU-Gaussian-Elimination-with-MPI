cmake_minimum_required(VERSION 3.14)
project(MPI_LU)

add_executable(MPI_LU main.cpp MPI_LU_Pipeline.cpp MPI_LU_Pipeline.h Matrix.cpp Matrix.h MPI_LU_Block.cpp MPI_LU_Block.h MPI_LU_Recycle.cpp MPI_LU_Recycle.h)

find_package(MPI)
include_directories(${MPI_INCLUDE_PATH})

SET(CMAKE_CXX_COMPILER "/usr/local/mpich-3.3/bin/mpicxx")
SET(CMAKE_C_COMPILER "/usr/local/mpich-3.3/bin/mpicc")
