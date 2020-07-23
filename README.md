# One-Sided Jacobi Rotation algorithm for SVD decomposition through NVIDIA CUDA C libraries on Jetson TK1 (GPU)

# Introduction
This [report](https://github.com/MatteoOrlandini/Embedded-Systems-Exam/blob/master/Report.pdf) describes the work done for the design of the One-sided Jacobi algorithm for the Singular Value Decomposition (SVD) of a matrix, via NVIDIA CUDA C libraries. The code has been tested and designed for the embedded Jetson TK1 platform. There is also an Italian report [here](https://github.com/MatteoOrlandini/Embedded-Systems-Exam/blob/master/Relazione.pdf).

A C code for SVD optimized for CPU work had already been developed in University. Our job was to compare our results with those previously obtained and evaluate the possibility of an alternative implementation that would fully exploit the potential of the GPU parallel calculation.

NOTE: due to the Coronavirus disease we have therefore used as a reference an NVIDIA GTX 610M GPU, whose results are proportional to those obtained with the Jetson. We were unable to use the Jetson TK1 because the University was closed during the lockdown.

## Jetson TK1
The Jetson TK1 is an embedded GPU made by NVIDIA that contains a Tegra K1 SoC in the T124 variant. It also has an Ubuntu Linux operating system and has installed the CUDA 6.5. The GPU’s compute capability, that is the computing capacity that determines the general specifications and available functionality, is 3.2. For more information about this GPU please see the [Device Properties folder](https://github.com/MatteoOrlandini/Embedded-Systems-Exam/tree/master/Device%20Properties).

## SVD
In linear algebra, the singular value decomposition (SVD) is a factorization of a matrix into three different matrices based on the use of eigenvalues and
eigenvectors. The decomposition of a matrix A is based on the fundamental theorem
of linear algebra: 

`Given a matrix A of rank p, the singular value decomposition of A is represented by the product of two unitary matrices U, V and a diagonal matrix Σ, as A = UΣV*.`

To perform the SVD of a matrix, several algorithms have been developed with the aim of optimizing the number of operations carried out by the machine. One of the most widely used is the Jacobi’s algorithm with its One sided Jacobi variant. The approach used is to apply successive rotations to the original matrix, in order to bring to zero the components that are outside the diagonal. Through different iterations, a diagonal matrix containing the required singular values is obtained as the final result.

# Prereqs

(Tested on Ubuntu 16.04)

* Cuda 8.0 - nvcc 8.0
* GCC version 5.5.0 

# How to compile

## Compile C code

In [C files folder](https://github.com/MatteoOrlandini/Embedded-Systems-Exam/tree/master/C%20files), open a terminal window and type

`gcc host_functions.c svd_main.c svd_one_sided_jacobi_C.c -o svd_main_linux -lm`


## Compile Cuda C code

In [Cuda files folder](https://github.com/MatteoOrlandini/Embedded-Systems-Exam/tree/master/Cuda%20files), open a terminal window and type

`nvcc OneSidedJacobiParallelShared.cu svd_one_sided_jacobi_C.cu host_functions.cu -o OneSidedJacobiParallelShared`

You can choose between 4 algorithms:
* OneSidedJacobiSerial.cu
* OneSidedJacobiParallelGlobal.cu
* OneSidedJacobiParallelSemiShared.cu
* OneSidedJacobiParallelShared.cu

# How to run

You need to test the code with the matrices in the [matrix folder](https://github.com/MatteoOrlandini/Embedded-Systems-Exam/tree/master/Matrix). There are already some example matrices, but others can be used as long as they are put in this folder.

## Run C code

In [C files folder](https://github.com/MatteoOrlandini/Embedded-Systems-Exam/tree/master/C%20files), open a terminal window and type

`./svd_main_linux`

or

`./svd_main_linux A32x24`

If you only type `./svd_main_linux`, then insert the name of one of the matrices in the [matrix folder](https://github.com/MatteoOrlandini/Embedded-Systems-Exam/tree/master/Matrix) such as

`A32x24`.

## Run Cuda C code

In [Cuda files folder](https://github.com/MatteoOrlandini/Embedded-Systems-Exam/tree/master/Cuda%20files), open a terminal window and type

`./OneSidedJacobiParallelShared`

# Results

You can see the singular values shown in the terminal window and they are saved into [Singular Values folder](https://github.com/MatteoOrlandini/Embedded-Systems-Exam/tree/master/SingularValues). The time taken by the algorithm is saved in [Time folder](https://github.com/MatteoOrlandini/Embedded-Systems-Exam/tree/master/Time).

Below is shown a comparison between the four GPU algorithms implemented and the CPU algorithm in relation to the time taken to calculate the singular values. For more results, please see the [report](https://github.com/MatteoOrlandini/Embedded-Systems-Exam/blob/master/Report.pdf).
![](https://github.com/MatteoOrlandini/Embedded-Systems-Exam/blob/master/TimeVsColumns.png)
