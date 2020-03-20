#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

__global__ void coefficientCalc (float * B, float * alpha, float * beta, float * gamm, int *Mdev, int *Ndev){
	int i = threadIdx.x; 
	int j = threadIdx.y;
	int k = blockIdx.z;
	int M = *Mdev;
	int N = *Ndev;
	for (int i = 0; i < rows; i++){
		for (int j = 0; j < columns; j++){
			printf ("%f\t", B[i*N+j]);
		}
	printf("\n");
	}
	if ((i < j) && (i < M) && (j < N) && (k < M)) {
		float *pi = (B + M * i) + k, *pj = (B + M * j) + k;
		alpha[i*M+j] += *pi * *pi; // atomic add?
		beta[i*M+j] += *pj * *pj;
		gamm[i*M+j] += *pi * *pj;
	}
}

int main (int argc, char * argv[]){
	float host_B [] = {1, 2, 3, 4, 5, 6, 7, 8, 9};

	float host_alpha []= {0, 0, 0, 0, 0, 0, 0, 0, 0};
	float host_beta [] = {0, 0, 0, 0, 0, 0, 0, 0, 0};
	float host_gamm [] = {0, 0, 0, 0, 0, 0, 0, 0, 0};
	int rows = 3;
	int columns = 3;
	float * B, * alpha, * beta, * gamm;
	int * M, * N;
    cudaMalloc( (void**)&B, rows * columns * sizeof(float) );		
	cudaMalloc( (void**)&alpha, rows * columns * sizeof(float) );	
	cudaMalloc( (void**)&beta, rows * columns * sizeof(float) );	
	cudaMalloc( (void**)&gamm, rows * columns * sizeof(float) );	
	cudaMalloc( (void**)&M, sizeof(int) );
	cudaMalloc( (void**)&N, sizeof(int) );

    cudaMemcpy( B, host_B, rows * columns * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy( alpha, host_alpha, rows * columns * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy( beta, host_beta, rows * columns * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy( gamm, host_gamm, rows * columns * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy( M, &rows, sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy( N, &columns, sizeof(int), cudaMemcpyHostToDevice);
	
	dim3 block(rows, columns, 1);
	dim3 grid(1, 1, columns);

	coefficientCalc<<<grid,block>>> (B, alpha, beta, gamm, M, N);

	cudaMemcpy( host_B,B, rows * columns * sizeof(float), cudaMemcpyDeviceToHost);
	cudaMemcpy( host_alpha,alpha,  rows * columns * sizeof(float), cudaMemcpyDeviceToHost);
	cudaMemcpy( host_beta,beta,  rows * columns * sizeof(float), cudaMemcpyDeviceToHost);
	cudaMemcpy( host_gamm,gamm, rows * columns * sizeof(float), cudaMemcpyDeviceToHost);
	cudaMemcpy( & rows,M, sizeof(int), cudaMemcpyDeviceToHost);
	cudaMemcpy( &columns, N, sizeof(int), cudaMemcpyDeviceToHost);

	printf("device alpha\n");
	for (int i = 0; i < rows; i++){
		for (int j = 0; j < columns; j++){
			printf ("%f \t", host_alpha[i*columns+j]);
		}
	printf("\n");
	}
	printf("\n");
	printf("device beta\n");
	for (int i = 0; i < rows; i++){
		for (int j = 0; j < columns; j++){
			printf ("%f \t", host_beta[i*columns+j]);
		}
	printf("\n");
	}
	printf("\n");
	printf("device gamma\n");
	for (int i = 0; i < rows; i++){
		for (int j = 0; j < columns; j++){
			printf ("%f \t", host_gamm[i*columns+j]);
		}
	printf("\n");
	}
	printf("\n");
	
	float alpha2 [3][3]= {{0, 0, 0},{ 0, 0, 0},{ 0, 0, 0}};
	float beta2 [3][3]= {{0, 0, 0},{ 0, 0, 0},{ 0, 0, 0}};
	float gamm2 [3][3]= {{0, 0, 0},{ 0, 0, 0},{ 0, 0, 0}};
	float B2 [] = {1, 2, 3, 4, 5, 6, 7, 8, 9};
	
	for (int j = 3 - 1; j >= 1; --j){
			for (int i = j - 1; i >= 0; --i) {
				float *pi = B2 + 3 * i, *pj = B2 + 3 * j;
				for (int k = 0; k < 3; ++k) {
					alpha2[i][j] += *pi * *pi;
					beta2[i][j] += *pj * *pj;
					gamm2[i][j] += *pi++ * *pj++;
				}
			//printf("%f\t",alpha2);
			}
	//printf("\n");
	}

	printf("host alpha\n");
	for (int i = 0; i < rows; i++){
		for (int j = 0; j < columns; j++){
			printf ("%f\t", alpha2[i][j]);
		}
	printf("\n");
	}
	printf("\n");
	printf("host beta\n");
	for (int i = 0; i < rows; i++){
		for (int j = 0; j < columns; j++){
			printf ("%f\t", beta2[i][j]);
		}
	printf("\n");
	}
	printf("\n");	
	printf("host gamma\n");
	for (int i = 0; i < rows; i++){
		for (int j = 0; j < columns; j++){
			printf ("%f\t", gamm2[i][j]);
		}
	printf("\n");
	}
	printf("\n");
}

