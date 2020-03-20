//nvcc OneSidedJacobiParallelSemiShared.cu svd_one_sided_jacobi_C.cu host_functions.cu -o OneSidedJacobiParallelSemiShared

#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include <unistd.h> //for chdir
#include <stdbool.h> //for bool type
#include "host_functions.h" //for host functions
#include "cuda_error_check.h" //for error checking

// Define this to turn on error checking
#define CUDA_ERROR_CHECK

#define CudaSafeCall( err ) __cudaSafeCall( err, __FILE__, __LINE__ )
#define CudaCheckError()    __cudaCheckError( __FILE__, __LINE__ )

static const float eps = 1e-4;
float * host_B;
float * host_AUX1;

__device__ int sign (float num){
	if (num > 0) return 1;
	if (num < 0) return -1;
	return 0;
}


__global__ void scheduling (int *v1, int *v2, int cols){
	int tmp = v2[0];
	for (int i = 0; i < (cols/2) - 1; i++)
		v2[i] = v2[i+1];	
	v2[cols/2 - 1] = v1[cols/2 - 1];
	for (int i = (cols/2) -1; i > 1; i--)
		v1[i] = v1[i-1];	
	v1[1] = tmp;
}


__global__ void round (float *B, int *v1, int *v2, int cols, int rows, bool * exit_flag) {
	int blockId = blockIdx.x; //max(blockId) = (cols/2) - 1
	int threadId = threadIdx.x; //max(blockId) = rows - 1
	float * pi, *pj;
	__shared__ float alpha, beta, gamm, limit, tao, t, c, s;
	__shared__ int i, j;
	if ((blockId < cols/2) && (threadId < rows)){
		i = *(v1 + blockId);
		j = *(v2 + blockId);
		pi = B + rows * i + threadId;
		pj = B + rows * j + threadId;
		alpha = beta = gamm = 0;
		__syncthreads();
		atomicAdd(&alpha, *pi * *pi);
		atomicAdd(&beta, *pj * *pj);	
		atomicAdd(&gamm, *pi * *pj);
		__syncthreads();
		if ( *exit_flag) {
			limit = fabsf(gamm) / sqrtf(alpha * beta);
			if (limit > eps){
				*exit_flag = false;
			}
		} 
		tao = (beta - alpha) / (2 * gamm);
		t = sign (tao) / (fabsf(tao) + sqrtf(1 + tao * tao)); 
		c = expf(-0.5f * log1pf(t * t));
		s = c * t;
		const float tmp = *pi;
		*pi = c * tmp - s * *pj;
		*pj = s * tmp + c * *pj;
	}
}

__global__ void computeSingVals (float * B, float * AUX1, int rows, int columns){
	int k = threadIdx.x; //max(k)=rows-1
	int j = blockIdx.x; //max(j)=columns-1
	__shared__ float t;
	if ((j < columns) && (k < rows)){
		float *pj = B + rows * j + k;
		t = 0;
		__syncthreads();
		atomicAdd(&t, *pj * *pj);
		__syncthreads();
		AUX1[j] = sqrtf(t);
	}
}

int main (int argc, char * argv[]) {
	char input [100], fileName[] = {"Matrix/"};
	chdir("../");
	FILE *fp;
	if (argc == 1){
	    do
	    {
	        printf ("Insert matrix name: \n");
	        scanf ("%s", input);
	        strcat(fileName, input);
	    }
	    while(openFile(&fp, fileName, "r") == false);
	}
	else if (argc == 2){
	    sprintf(fileName, "Matrix/%s", argv[1]);
	    if (openFile(&fp, fileName, "r") == false)
	    	exit(1);
	} 
	float * matrix;
	int rows, cols;
	fillRowMajorOrderMatrix(&fp, &matrix, &rows, &cols);
	fclose(fp);
	// Column order matrix
	createColumnMajorOrderMatrix(&host_B, matrix, rows, cols);
	// initialize host_AUX1 array to zero
	initializeArray (&host_AUX1, cols);
	// Open new file to store the singular values
	sprintf(fileName, "SingularValues/CudaHost/Singular Values Cuda Host %dx%d.txt", rows, cols);
	openFile(&fp, fileName, "w");
	// cuda events
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start);
	//compute one sided jacobi
	int iterations = svd_one_sided_jacobi_C(rows, cols);
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	float milliseconds_host = 0;
	cudaEventElapsedTime(&milliseconds_host, start, stop);
	cudaEventDestroy (start); 
	cudaEventDestroy (stop); 
	printf("Iterations on host: %d \n", iterations);
	descentOrdering(host_AUX1, cols);
	//print array and save on file
	printAndSaveArray (&fp, host_AUX1, cols);
	fclose(fp);

	/****************************************************************************************************/
	/****************************************************************************************************/
	/****************************************************************************************************/
	/****************************************************************************************************/
	free(host_B);
	createColumnMajorOrderMatrix(&host_B, matrix, rows, cols);
	/*
	for (int i = 0; i < rows; i++){
		for (int j = 0; j < cols; j++){
			printf ("B[%d]:%f\t", i*cols+j, host_B[i*cols+j]);
		}
		printf ("\n");
	}
	*/

	float * dev_AUX1, * dev_B;
	int * dev_v1, * dev_v2;	
	bool * dev_exit_flag;

	CudaSafeCall(cudaMalloc( (void**)&dev_B, rows*cols*sizeof(float) ));
	CudaSafeCall(cudaMalloc( (void**)&dev_exit_flag, sizeof(bool) ));
	CudaSafeCall(cudaMalloc( (void**)&dev_AUX1, cols * sizeof(float) ));
	CudaSafeCall(cudaMalloc( (void**)&dev_v1, (cols/2) * sizeof(float) ));
	CudaSafeCall(cudaMalloc( (void**)&dev_v2, (cols/2) * sizeof(float) ));
	/***********************************/
	int host_v1[cols/2], host_v2[cols/2];
	for (int i = 0; i < cols/2; i++) {
		host_v1[i] = i*2;
		host_v2[i] = i*2 + 1;
	}
	/***********************************/
	CudaSafeCall(cudaMemcpy( dev_B, host_B, rows * cols * sizeof(float), cudaMemcpyHostToDevice));
	CudaSafeCall(cudaMemcpy( dev_v1, host_v1, (cols/2) * sizeof(float), cudaMemcpyHostToDevice));
	CudaSafeCall(cudaMemcpy( dev_v2, host_v2, (cols/2) * sizeof(float), cudaMemcpyHostToDevice));

	//printf ("MATRIX ALLOCATED ON DEVICE \n");
	int iter = 0;
	bool host_exit_flag = false;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start);
	while(!host_exit_flag) {
		++iter;
		host_exit_flag = true; //rimuovendo si perde accuratezza
		CudaSafeCall(cudaMemcpy( dev_exit_flag, &host_exit_flag, sizeof(bool), cudaMemcpyHostToDevice));
		for(int set = 0; set < cols; set++) { //rimuovendo il for si perde un po' di accuratezza ma dimuiscono le iterazioni
			scheduling<<<1,1>>> (dev_v1, dev_v2, cols);
			CudaCheckError();
			round <<<cols/2, rows>>> (dev_B, dev_v1, dev_v2, cols, rows, dev_exit_flag);
			CudaCheckError();		
		}
		CudaSafeCall(cudaMemcpy( &host_exit_flag, dev_exit_flag, sizeof(bool), cudaMemcpyDeviceToHost));
	}
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	float milliseconds_device = 0;
	cudaEventElapsedTime(&milliseconds_device, start, stop);
	// open new file to store the singular values on device
	sprintf(fileName, "SingularValues/CudaDevice/OneSidedParallelSemiShared/Singular Values Cuda Device %dx%d.txt", rows, cols);
	openFile(&fp, fileName, "w");

	printf ("Iterations on device: %d\n", iter);

	// calculate singular values
	computeSingVals<<<cols, rows>>> (dev_B, dev_AUX1, rows, cols);
	CudaCheckError();
	CudaSafeCall(cudaMemcpy( host_AUX1, dev_AUX1, cols * sizeof(float),  cudaMemcpyDeviceToHost));

	descentOrdering(host_AUX1, cols);
	printAndSaveArray (&fp, host_AUX1, cols);
	fclose(fp);
	//printf("SINGULAR VALUES STORED TO FILE \n");

	// free the memory allocated on host
	free(host_B);
	free(host_AUX1);

	//printf ("MEMORY ON HOST DEALLOCATED \n");

	// free the memory allocated on device
	cudaFree(dev_B);
	cudaFree(dev_exit_flag);
	cudaFree(dev_AUX1);
	cudaFree(dev_v1);
	cudaFree(dev_v2);

	//printf ("MEMORY ON DEVICE DEALLOCATED \n");

	printf ("Time on host: %f ms\n", milliseconds_host);
	printf ("Time on device: %f ms\n", milliseconds_device);

	sprintf(fileName, "Time/CudaHost/Time %dx%d.txt", rows, cols);
	openFile(&fp, fileName, "w");
	fprintf(fp, "%f", milliseconds_host);	
	fclose(fp);

	sprintf(fileName, "Time/CudaDevice/OneSidedParallelSemiShared/Time %dx%d.txt", rows, cols);
	openFile(&fp, fileName, "w");
	fprintf(fp, "%f", milliseconds_device);	
	fclose(fp);

	cudaDeviceReset ( ); 

	return 0;
}
		

