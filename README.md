# fortran-NeuralNetwork

This repository contains a simple straight forward implementation of a neural network in fortran. 
An example script on how to use the code is provided in which the neural network is trained using the extended Kalman filter method.


## Compiling the code

To compile the example you need a fortran compiler as well as an installation of Blas/LAPACK. 
Using CMake the code can be compiled with the following commands.
Two CMake flags are provided, that allow to compile with or without OpenMP parallelization and with the intel or gnu compiler.

```bash
mkdir build
cd build
cmake -DOPENMP=ON -DINTEL=OFF .. # compile with OpenMP parallelization and gfortran
make
```
