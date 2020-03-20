#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>

#define MAXCHAR 3000

float * host_B;
float * host_AUX1;

//__device__ float eps = 1e-4; 
 
int sign (const float num){
    if (num > 0)
		return 1;
    if (num < 0)
		return -1;
    else 
		return 0;
}

__global__ void clearMatrix (float * alpha, float * beta, float * gamm, int * Mdev, int * Ndev) {
	int i = threadIdx.x; 
	int j = threadIdx.y;
	int M = *Mdev;
	int N = *Ndev;
	if ((i < j) && (i < M) && (j < N)) {
		alpha[i*N+j] = 0;
		beta[i*N+j] = 0;
		gamm[i*N+j] = 0;
	}
}

__global__ void coefficientCalc (float * B, float * alpha, float * beta, float * gamm, int *Mdev, int *Ndev){
	int i = threadIdx.x; 
	int j = threadIdx.y;
	int k = blockIdx.z;
	int M = *Mdev;
	int N = *Ndev;
	if ((i < j) && (i < M) && (j < N) && (k < M)) {
		float *pi = (B + M * i) + k, *pj = (B + M * j) + k;
		atomicAdd(&alpha[i*N+j], *pi * *pi);
		atomicAdd(&beta[i*N+j], *pj * *pj);
		atomicAdd(&gamm[i*N+j], *pi * *pj);
	}
}

__global__ void flagCheck (float * alpha, float * beta, float * gamma, int *Mdev, int *Ndev, int * exit_flag) {
	int i = threadIdx.x; 
	int j = threadIdx.y;
	int M = *Mdev;
	int N = *Ndev;
	float eps = 1e-4;
	if ((i < j) && (i < M) && (j < N)) {
		if (exit_flag) {
			const float limit = fabsf(gamma[i*N+j]) / sqrtf(alpha[i*N+j] * beta[i*N+j]);
			if (limit > eps) *exit_flag = 1;
		}
	}
}

void rotate (float * B, float * alpha, float * beta, float * gamma, int M, int N) {
	for (int j = N - 1; j >= 1; --j)
			for (int i = j - 1; i >= 0; --i) {
				float t = (beta[i*N+j] - alpha[i*N+j]) / (2 * gamma[i*N+j]);
				t = sign(t) / (fabsf(t) + sqrtf(1 + t * t));
				// t[i][i] = sign(t[i][j]) / (fabsf(t[i][j]) + sqrtf(1 + t[i][j] * t[i][j]))
				const float c = expf(-0.5f * log1pf(t * t));  // new trick by Giorgio! Better than passing to 64 bits.
				const float s = c * t;
				float *pi = B + M * i, *pj = B + M * j;
				for (int k = 0; k < M; ++k) {
					const float t = *pi;
					*pi++ = c * t - s * *pj;
					*pj = s * t + c * *pj;
					++pj;
				}
	}
}

__global__ void singVal (float * B, float * t, int *Mdev, int *Ndev) {
	int j = threadIdx.x; 
	int k = threadIdx.y;
	int M = *Mdev;
	int N = *Ndev;
	if ((k < M) && (j < N)) {
		float *pj = B + M * j;
		atomicAdd(&t[j], *(pj + k) * *(pj + k));
	}
}

int svd_one_sided_jacobi_C(int rows, int columns);

int main (int argc, char * argv[])
{
    char fileName[100];
    FILE *fp;
    do
    {
        printf ("Inserisci il nome del file: \n");
        scanf ("%s", fileName);
    }
    while((fp = fopen(fileName, "r")) == NULL);
    char buf[MAXCHAR]; //buffer for reading the file
    char * numChar; //elemento della matrice (salvato come ascii)
    float * matrix;
    matrix = (float *)malloc(sizeof(float));
    int rows = 0, columns = 0, numMatrixEle = 0;
    while (fgets(buf, MAXCHAR, fp) != NULL){
        numChar = strtok(buf, " ");
        columns = 0;
        while (numChar != NULL){
            *(matrix + numMatrixEle) = atof(numChar);
            //printf("A[%d][%d]: %2.9f \t", rows, columns, *(matrix+numMatrixElem));
            numChar = strtok(NULL, " ");
            columns++;
            numMatrixEle++;
            matrix=(float*)realloc(matrix, (numMatrixEle+1)*sizeof(float));
        }
        //printf("\n");
        rows++;
    }
	//printf("rows: %d , columns: %d \n", rows, columns);
    fclose(fp);
    /* Column order matrix */
    host_B = (float*)malloc(rows*columns*sizeof(float));
    if(host_B == NULL)
    {
        printf("Memoria esaurita\n");
        exit(1);
    }
    int cont = 0;
    for (int i = 0; i < columns; i++){
        for (int j = 0; j < rows; j++){
            host_B[cont] = matrix[j*columns+i];
            //printf("B[%d]: %2.9f \n", cont, host_B[cont]);
            cont++;
        }
    }

    host_AUX1 = (float*)malloc(columns*sizeof(float));
    /* inizializzo il vettore AUX1 */
    for (int i = 0; i < columns; i++){
        host_AUX1[i] = 0;
    }
    /* Open new file to store the singular values */
	sprintf(fileName, "Singular values %dX%d.txt", rows, columns);
    fp = fopen(fileName, "w");
    //int iterations = svd_one_sided_jacobi_C(rows, columns); //eliminato parte Alessandrini
	int iterations = 0;
    printf("iterations: %d \n", iterations);
    fprintf(fp, "iterations: %d \n", iterations);
    /* Loop for descending ordering */
    for (int i = 0; i < columns; i++)
	{
		for (int j = 0; j < columns; j++)             //Loop for comparing other values
		{
			if (host_AUX1[j] < host_AUX1[i])                //Comparing other array elements
			{
				float tmp = host_AUX1[i];         //Using temporary variable for storing last value
				host_AUX1[i] = host_AUX1[j];            //replacing value
				host_AUX1[j] = tmp;             //storing last value
			}
		}
	}
    for (int i = 0; i < columns; i++){
        printf ("AUX[%d]: %f \n", i, host_AUX1[i]);
        fprintf(fp, "%f\n", host_AUX1[i]);
    }
	/* free the memory allocated on the CPU */
    fclose(fp);  
	/**************************************************************************************************************/
	/**************************************************************************************************************/
	/* Inizio scrittura file comune CPU + GPU */
	float * host_t, *t, *B, *host_alpha, *host_beta, *host_gamm, *alpha, *beta, *gamm;
	int *M, *N;

	host_alpha = (float*)malloc(rows*columns*sizeof(float));
	host_beta = (float*)malloc(rows*columns*sizeof(float));
	host_gamm = (float*)malloc(rows*columns*sizeof(float));
	host_t = (float*)malloc(columns * sizeof(float));

	// fill the matrix alpha, beta, gamm on the CPU
	for (int i = 0; i < (rows * columns); i++){
		host_alpha[i]= 0;
		host_beta[i] = 0;
		host_gamm[i] = 0;
	}

	for (int i = 0; i < columns; i++){
		host_t[i] = 0;
	}
	
    cudaMalloc( (void**)&B, rows * columns * sizeof(float) );		
	cudaMalloc( (void**)&alpha, rows * columns * sizeof(float) );	
	cudaMalloc( (void**)&beta, rows * columns * sizeof(float) );	
	cudaMalloc( (void**)&gamm, rows * columns * sizeof(float) );	
	cudaMalloc( (void**)&M, sizeof(int) );
	cudaMalloc( (void**)&N, sizeof(int) );
	cudaMalloc( (void**)&t, columns * sizeof(float) );

    cudaMemcpy( B, host_B, rows * columns * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy( alpha, host_alpha, rows * columns * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy( beta, host_beta, rows * columns * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy( gamm, host_gamm, rows * columns * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy( M, &rows, sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy( N, &columns, sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy( t, host_t, columns * sizeof(float), cudaMemcpyHostToDevice);
		
	dim3 block(rows, columns, 1);
	dim3 grid(1, 1, columns);
	int iter = 0;

	int host_exit_flag = 0;
	int * exit_flag;

	cudaMalloc( (void**)&exit_flag, sizeof(int) );
	cudaMemcpy( exit_flag, &host_exit_flag, sizeof(int), cudaMemcpyHostToDevice);
	/*
	for (int i = 0; i < rows; i++){
		for (int j = 0; j < columns; j++){
			printf ("%f \t", host_B[i*columns+j]);
		}
	}
	*/
	while (!host_exit_flag) {
		iter++;
		clearMatrix<<<1,block>>> (alpha, beta, gamm, M, N);
		coefficientCalc<<<grid,block>>> (B, alpha, beta, gamm, M, N);
		flagCheck<<<1,block>>> (alpha, beta, gamm, M, N, exit_flag);
		cudaMemcpy( host_B, B, rows * columns * sizeof(float), cudaMemcpyDeviceToHost);
		cudaMemcpy( host_alpha, alpha, rows * columns * sizeof(float), cudaMemcpyDeviceToHost);
		cudaMemcpy( host_beta, beta, rows * columns * sizeof(float), cudaMemcpyDeviceToHost);
		cudaMemcpy( host_gamm, gamm, rows * columns * sizeof(float), cudaMemcpyDeviceToHost);
		printf ("2\n");
		rotate(host_B, host_alpha, host_beta, host_gamm, rows, columns);
		printf ("3\n");
		cudaMemcpy( &host_exit_flag, exit_flag, sizeof(int), cudaMemcpyDeviceToHost);
		printf ("Iterations: %d, Flag: %d, Alpha[0][1]: %f\n", iter, host_exit_flag, *(host_alpha+1));
		/* AZZERARE LE MATRICI ALPHA, BETA, GAMM ? */
	}
	//printf ("Iterations: %d\n", iter);
	printf ("4\n");
	singVal<<<1,block>>> (B, t, M, N);
	
	//copy the array t on the CPU
	cudaMemcpy( host_t, t, columns * sizeof(float), cudaMemcpyDeviceToHost);
    	printf ("5\n");
	for (int i = 0; i < columns; i++){ //for o kernel
		host_AUX1[i] = sqrtf(host_t[i]);
	}
	
	/* Loop for descending ordering */
    for (int i = 0; i < columns; i++)
	{
		for (int j = 0; j < columns; j++)             //Loop for comparing other values
		{
			if (host_AUX1[j] < host_AUX1[i])                //Comparing other array elements
			{
				float tmp = host_AUX1[i];         //Using temporary variable for storing last value
				host_AUX1[i] = host_AUX1[j];            //replacing value
				host_AUX1[j] = tmp;             //storing last value
			}
		}
	}
	
	/* Print AUX in descending order */
    for (int i = 0; i < columns; i++){
        printf ("host_AUX1[%d]: %f \n", i, host_AUX1[i]);
    }
	
	/* free the memory allocated on the CPU */
	
	free(host_B);
	free(host_AUX1);
	
	/* free the memory allocated on the GPU */
	cudaFree(B);
	cudaFree(alpha);
	cudaFree(beta);
	cudaFree(gamm);
	
	printf("\nMemoria Liberata\n");
	
	return 0;
}


