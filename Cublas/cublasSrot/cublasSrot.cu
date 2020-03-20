//nvcc cublasSrot.cu -o cublasSrot -lcublas

#include <stdio.h>
#include  <stdlib.h>
#include  <cuda_runtime.h>
#include "cublas_v2.h"
#define n 6                                         //  length  of x,y
int  main(void){
	cudaError_t  cudaStat;                      //  cudaMalloc  status
	cublasStatus_t  stat;                //  CUBLAS  functions  status
	cublasHandle_t  handle;                        //  CUBLAS  context
	int j;                                        //  index of  elements
	float* x;                                 // n-vector  on the  host
	float* y;                                 // n-vector  on the  host
	x=( float *) malloc (n*sizeof (*x));// host  memory  alloc  for x
	for(j=0;j<n;j++)
		x[j]=( float)j;                              // x={0,1,2,3,4,5}
	y=( float *) malloc (n*sizeof (*y));// host  memory  alloc  for y
	for(j=0;j<n;j++)
		y[j]=( float)j*j;                         // y={0,1,4,9,16,25}
	printf("x: ");
	for(j=0;j<n;j++)
		printf("%7.0f,",x[j]);                              //  print x
	printf("\n");
	printf("y: ");
	for(j=0;j<n;j++)
		printf("%7.0f,",y[j]);                              //  print y
	printf("\n");
	// on the  device
	float* d_x;                             // d_x - x on the  device
	float* d_y;                             // d_y - y on the  device
	cudaStat=cudaMalloc ((void **)&d_x ,n*sizeof (*x));     // device  memory  alloc  for x
	cudaStat=cudaMalloc ((void **)&d_y ,n*sizeof (*y));     // device  memory  alloc  for y
	stat = cublasCreate (& handle );   //  initialize  CUBLAS  context
	stat = cublasSetVector(n,sizeof (*x),x,1,d_x ,1); //cp x->d_x
	stat = cublasSetVector(n,sizeof (*y),y,1,d_y ,1); //cp y->d_y
	float c=0.5;float s=0.8669254;                            // s=sqrt (3.0)/2.0
	//  Givens  rotation
	//                            [ c s ]                       [ row(x) ]
	// multiplies   2x2  matrix [      ]   with 2xn  matrix   [          ]
	//                            [-s c ]                       [ row(y) ]
	//
	//   [1/2           sqrt (3)/2]     [0,1,2,3, 4, 5]
	//   [-sqrt (3)/2        1/2   ]     [0,1,4,9,16,25]
	stat=cublasSrot(handle,n,d_x,1,d_y,1,&c,&s);
	stat=cublasGetVector(n,sizeof(float),d_x ,1,x,1);//cp d_x ->x
	printf("x after  Srot:\n");               //  print x after  Srot
	for(j=0;j<n;j++)
		printf("%7.3f,",x[j]);
	printf("\n");
	stat=cublasGetVector(n,sizeof(float),d_y ,1,y,1);//cp d_y ->y
	printf("y after  Srot:\n");               //  print y after  Srot
	for(j=0;j<n;j++)
		printf("%7.3f,",y[j]);
	printf("\n");
	cudaFree(d_x);                             // free  device  memory
	cudaFree(d_y);                             // free  device  memory
	cublasDestroy(handle );               //  destroy  CUBLAS  context
	free(x);                                       // free  host  memory
	free(y);                                       // free  host  memory
	return  EXIT_SUCCESS;
}

// x:      0,     1,     2,     3,     4,     5,
// y:      0,     1,     4,     9,    16,    25,
// x after  Srot:
//   0.000,   1.367,   4.468,   9.302,  15.871 ,  24.173 ,
// y after  Srot:
//   0.000,  -0.367,   0.266,   1.899,   4.532,   8.165,
//                        // [x]   [ 0.5    0.867]  [0 1 2 3   4   5]
//                        // [ ]=  [               ]*[                ]
//                        // [y]   [ -0.867   0.5 ] [0 1 4 9 16 25]
