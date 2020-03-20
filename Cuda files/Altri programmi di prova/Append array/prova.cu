#include <stdio.h>
#include <stdlib.h>

__global__ void append (float *B,float *A_ij,int i,int j,int k,int rows ) { 
	int h = threadIdx.x;
	if (h>2*k*rows) return;
	//printf("%d ",h);
	if (h<k*rows) A_ij[h]=*(B+i*rows+h);
	else A_ij[h]=*(B+j*rows+h-k*rows);
}

/*
for (int i=0; i<cols-2*k; i+=k)
	for (int j=i+k; j<cols-k; j+=k)
		append..
		syrk..
		autov..
		gemm..
*/

int main (){
	int rows = 3;
	int cols = 6;
	int k = 2;
	//int A_i[rows*k], A_j[rows*k], A_ij[2*rows*k];
	float h_B [] = {0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17};
	float * A_ij, *B;
	float h_A_ij[2*rows*k];
	
	cudaMalloc( (void**)&A_ij, 2*k*rows * sizeof(float) );
	cudaMalloc( (void**)&B, rows*cols* sizeof(float) );
	cudaMemcpy( B, h_B, rows*cols* sizeof(float), cudaMemcpyHostToDevice);
	
	append<<<1,2*k*rows>>>(B,A_ij,1,4,k,rows);

	cudaMemcpy( h_A_ij, A_ij, 2*k*rows * sizeof(float), cudaMemcpyDeviceToHost);

	for (int i = 0; i < 2*rows*k; i++){
		 printf ("%f\t",h_A_ij[i]);
	}
}
