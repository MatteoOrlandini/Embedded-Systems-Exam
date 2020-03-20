// nvcc sdot.cu -lcublas

# include <stdio.h>
# include <stdlib.h>
# include <cuda_runtime.h>
# include "cublas_v2.h"
# define n 6
int main ( void ){
	cudaError_t cudaStat ;
	cublasStatus_t stat ;
	cublasHandle_t handle ;
	int j ;
	float * x ;
	// length of x , y
	// cudaMalloc status
	// CUBLAS functions status
	// index of elements
	// n - vector on the host
	float * y ;
	// n - vector on the host
	x =( float *) malloc ( n * sizeof (* x )); // host memory alloc for x
	for ( j =0; j < n ; j ++)
		x [ j ]=( float ) j ;
	// x ={0 ,1 ,2 ,3 ,4 ,5}
	y =( float *) malloc ( n * sizeof (* y )); // host memory alloc for y
	for ( j =0; j < n ; j ++)
		y [ j ]=( float ) j ;
	// y ={0 ,1 ,2 ,3 ,4 ,5}
	printf ( "  x , y :\n " );
	for ( j =0; j < n ; j ++)
		printf ( " %2.0f , " ,x [ j ]);
	// print x , y
	printf ( " \n " );
	// on the device
	float * d_x ; // d_x - x on the device
	float * d_y ; // d_y - y on the device
	cudaStat = cudaMalloc (( void **)& d_x , n * sizeof (* x ));	// device memory alloc for x
	cudaStat = cudaMalloc (( void **)& d_y , n * sizeof (* y ));	// device memory alloc for y
	stat = cublasCreate (& handle ); // initialize CUBLAS context
	stat = cublasSetVector (n , sizeof (* x ) ,x ,1 , d_x ,1); // cp x - > d_x
	stat = cublasSetVector (n , sizeof (* y ) ,y ,1 , d_y ,1); // cp y - > d_y
	float result ;
	// dot product of two vectors d_x , d_y :
	// d_x [0]* d_y [0]+...+ d_x [n -1]* d_y [n -1]
	stat=cublasSdot(handle,n,d_x,1,d_y,1,&result);
	printf ( " dot product x . y :\n " );
	printf ( " %7.0f \n " , result ); // print the result
	cudaFree ( d_x ); // free device memory
	cudaFree ( d_y ); // free device memory
	cublasDestroy ( handle ); // destroy CUBLAS context
	free ( x );	// free host memory
	free ( y );	// free host memory
	return EXIT_SUCCESS ;
}
	// x , y :	
	// 0 , 1 , 2 , 3 , 4 , 5 ,
	
	// dot product x.y: 		//x.y =
	// 55						// 1 *1 + 2 *2 + 3 *3 + 4 *4 + 5 *5
