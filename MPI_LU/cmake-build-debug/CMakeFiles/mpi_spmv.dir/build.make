# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.14

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:


#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:


# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list


# Suppress display of executed commands.
$(VERBOSE).SILENT:


# A target that is always out of date.
cmake_force:

.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /media/enoch/Software/CLion-2019.1.4/clion-2019.1.4/bin/cmake/linux/bin/cmake

# The command to remove a file.
RM = /media/enoch/Software/CLion-2019.1.4/clion-2019.1.4/bin/cmake/linux/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/enoch/ParallelProgramming/MPI_LU

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/enoch/ParallelProgramming/MPI_LU/cmake-build-debug

# Include any dependencies generated for this target.
include CMakeFiles/mpi_spmv.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/mpi_spmv.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/mpi_spmv.dir/flags.make

CMakeFiles/mpi_spmv.dir/main.cpp.o: CMakeFiles/mpi_spmv.dir/flags.make
CMakeFiles/mpi_spmv.dir/main.cpp.o: ../main.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/enoch/ParallelProgramming/MPI_LU/cmake-build-debug/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/mpi_spmv.dir/main.cpp.o"
	/usr/local/mpich-3.3/bin/mpicxx  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/mpi_spmv.dir/main.cpp.o -c /home/enoch/ParallelProgramming/MPI_LU/main.cpp

CMakeFiles/mpi_spmv.dir/main.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/mpi_spmv.dir/main.cpp.i"
	/usr/local/mpich-3.3/bin/mpicxx $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/enoch/ParallelProgramming/MPI_LU/main.cpp > CMakeFiles/mpi_spmv.dir/main.cpp.i

CMakeFiles/mpi_spmv.dir/main.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/mpi_spmv.dir/main.cpp.s"
	/usr/local/mpich-3.3/bin/mpicxx $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/enoch/ParallelProgramming/MPI_LU/main.cpp -o CMakeFiles/mpi_spmv.dir/main.cpp.s

# Object files for target mpi_spmv
mpi_spmv_OBJECTS = \
"CMakeFiles/mpi_spmv.dir/main.cpp.o"

# External object files for target mpi_spmv
mpi_spmv_EXTERNAL_OBJECTS =

mpi_spmv: CMakeFiles/mpi_spmv.dir/main.cpp.o
mpi_spmv: CMakeFiles/mpi_spmv.dir/build.make
mpi_spmv: CMakeFiles/mpi_spmv.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/enoch/ParallelProgramming/MPI_LU/cmake-build-debug/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable mpi_spmv"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/mpi_spmv.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/mpi_spmv.dir/build: mpi_spmv

.PHONY : CMakeFiles/mpi_spmv.dir/build

CMakeFiles/mpi_spmv.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/mpi_spmv.dir/cmake_clean.cmake
.PHONY : CMakeFiles/mpi_spmv.dir/clean

CMakeFiles/mpi_spmv.dir/depend:
	cd /home/enoch/ParallelProgramming/MPI_LU/cmake-build-debug && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/enoch/ParallelProgramming/MPI_LU /home/enoch/ParallelProgramming/MPI_LU /home/enoch/ParallelProgramming/MPI_LU/cmake-build-debug /home/enoch/ParallelProgramming/MPI_LU/cmake-build-debug /home/enoch/ParallelProgramming/MPI_LU/cmake-build-debug/CMakeFiles/mpi_spmv.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/mpi_spmv.dir/depend

