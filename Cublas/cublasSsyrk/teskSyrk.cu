// nvcc testSyrk.cu -lcublas
#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include "host_functions.h" //for host functions
#include "cuda_error_check.h" //for error checking
#include  <cuda_runtime.h>
#include "cublas_v2.h"

//nvcc teskSyrk.cu host_functions.cu -o teskSyrk -lcublas

static const float eps = 1e-4;
float * host_B;
float * host_AUX1;

int main (int argc, char * argv[]) {
    
    int rows = 4;
    int cols = 3;
    float matrix[] = {0,1,2,3,4,5,6,7,8,9,10,11};
    float *A_ij,*G;
    float host_G [cols*cols];
    
    createColumnMajorOrderMatrix(&host_B, matrix, rows, cols);
    
    printf ("ROW MAJOR ORDER\tCOLUMN MAJOR ORDER\n");
    for (int i=0;i<rows*cols;i++) {
		printf("%f\t%f\n", matrix[i], host_B[i]);
		printf("\n");
    }
    
	cudaMalloc( (void**)&A_ij, rows * cols * sizeof(float) );
	cudaMalloc( (void**)&G, cols * cols * sizeof(float) );
	cudaMemcpy( A_ij, host_B, rows * cols * sizeof(float), cudaMemcpyHostToDevice);
	

	cublasStatus_t stat;
	cublasHandle_t  handle; 
	stat = cublasCreate(&handle); 

	float beta = 0.0f;
	float alpha = 1.0f;
	
	stat = cublasSsyrk(handle,CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_T, cols, rows, &alpha, A_ij, rows, &beta, G, cols); //A'*A
	
	cudaMemcpy( host_G, G, cols * cols * sizeof(float), cudaMemcpyDeviceToHost);
	
	for (int i=0;i<cols;i++){
		for (int j = 0; j < cols; j++){
		 	printf("%f\t",host_G[j*cols+i]);
		 }
    printf("\n");
    }
}	
