cmake_minimum_required(VERSION 3.6)
project(heckdarting Fortran)
enable_language(Fortran)


option(INTEL "compile with ifort and mkl" OFF)
option(DEBUG "compile with debug flags" OFF)

if(INTEL)
    set(CMAKE_Fortran_COMPILER mpifort)
else()
    set(CMAKE_Fortran_COMPILER gfortran)
endif()

if(INTEL)
    set(CMAKE_Fortran_FLAGS "${CMAKE_Fortran_FLAGS} -mkl -no-wrap-margin")
    if(DEBUG)
        set(CMAKE_Fortran_FLAGS "${CMAKE_Fortran_FLAGS} -g -traceback -O0 -fpe0") # -Ofast -ipo
    else()
        set(CMAKE_Fortran_FLAGS "${CMAKE_Fortran_FLAGS} -xHost -Ofast") # -ipo
    endif()
else()
    set(CMAKE_Fortran_MODULE_DIRECTORY ${PROJECT_BINARY_DIR}/modules)
    if(DEBUG)
        set(CMAKE_Fortran_FLAGS "${CMAKE_Fortran_FLAGS} -fcheck=all -Wall -Wconversion -g -fbacktrace -ffpe-trap=zero,invalid")
    else()
        set(CMAKE_Fortran_FLAGS "${CMAKE_Fortran_FLAGS} -Ofast -march=native -flto -fno-strict-aliasing")
    endif()
    find_package(BLAS REQUIRED)
    find_package(LAPACK REQUIRED)
    set(libraries ${libraries} ${BLAS_LIBRARIES} ${LAPACK_LIBRARIES})
endif()


add_executable(
        nn.x

        src/nnTest.f90
        src/neuralNetworks.f90
        src/kalmanFilter.f90
        src/util.f90
        src/precision.f90

)

target_link_libraries(nn.x ${libraries})
