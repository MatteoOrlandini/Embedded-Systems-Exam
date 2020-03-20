//nvcc cublasSnrm2.cu -o cublasSnrm2 -lcublas

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include "cublas_v2.h"
#define n 6 // length of x
int main ( void ){
	cudaError_t cudaStat ;
	cublasStatus_t stat ;
	cublasHandle_t handle ;
	// cudaMalloc status
	// CUBLAS functions status
	// CUBLAS context
	int j ;	// index of elements
	float * x ;	// n - vector on the host
	x =( float *) malloc ( n * sizeof (* x )); // host memory alloc for x
	for ( j =0; j < n ; j ++)
		x [ j ]=( float ) j ;	// x ={0 ,1 ,2 ,3 ,4 ,5}
	printf ( " x : " );
	for ( j =0; j < n ; j ++)
		printf ( " %2.0f , " ,x [ j ]);	// print x
	printf ( " \n " );						// on the device
	float * d_x ;	// d_x - x on the device
	cudaStat = cudaMalloc (( void **)& d_x , n * sizeof (* x ));	// device memory alloc for x
	stat = cublasCreate (& handle ); // initialize CUBLAS context
	stat = cublasSetVector (n , sizeof (* x ) ,x ,1 , d_x ,1); // cp x - > d_x
	float result ;
	// Euclidean norm of the vector d_x :
	// \ sqrt { d_x [0]^2+...+ d_x [n -1]^2}
	stat=cublasSnrm2(handle,n,d_x,1,&result);
	printf ( " Euclidean norm of x: ");	
	printf ( " %7.3f \n " , result ); 	// print the result
	cudaFree ( d_x );	// free device memory
	cublasDestroy ( handle );	// destroy CUBLAS context
	free ( x );	// free host memory
	return EXIT_SUCCESS ;
}
//x: 0, 1, 2, 3, 4, 5,
									// || x ||=
// Euclidean norm of x : 7.416 		//\ sqrt { 0 ^ 2 + 1 ^ 2 + 2 ^ 2 + 3 ^ 2 + 4 ^ 2 + 5 ^ 2 }
