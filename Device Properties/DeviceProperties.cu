#include <stdio.h>

int main( void ) {

	cudaDeviceProp prop;

	int count;
	cudaGetDeviceCount( &count ) ;


	for (int i=0; i< count; i++) {
		cudaGetDeviceProperties( &prop, i );
		printf("--- General Information for device %d ---\n", i );
		printf( "Name: %s\n", prop.name );
		printf( "Compute capability: %d.%d\n", prop.major, prop.minor );
		printf( "Clock rate: %d [MHz]\n", (prop.clockRate)/1000 );
		printf( "Device copy overlap: " );
		if (prop.deviceOverlap)
			printf( "Enabled\n" );
		else
			printf( "Disabled\n" );
			printf( "Kernel execition timeout: " );
		if (prop.kernelExecTimeoutEnabled)
			printf( "Enabled\n" );
		else
			printf( "Disabled\n" );
		printf( "--- Memory Information for device %d ---\n", i );
		printf( "Total global mem: %ld [bytes]\n", prop.totalGlobalMem );
		printf( "Total constant Mem: %ld [bytes]\n", prop.totalConstMem );
		printf( "Maximum pitch allowed by memory copies: %ld [bytes]\n", prop.memPitch );
		printf( "Texture Alignment: %ld\n", prop.textureAlignment );
		printf( "--- MP Information for device %d ---\n", i );
		printf( "Multiprocessor count: %d\n", prop.multiProcessorCount );
		printf( "Shared memory available per block: %ld [bytes]\n", prop.sharedMemPerBlock );
		printf( "32-bit registers available per block: %d\n", prop.regsPerBlock );
		printf( "Warp size: %d [threads]\n", prop.warpSize );
		printf( "Maximum number of threads per block: %d\n", prop.maxThreadsPerBlock );
		printf( "Maximum size of each dimension of a block: (%d, %d, %d)\n", prop.maxThreadsDim[0], prop.maxThreadsDim[1], prop.maxThreadsDim[2] );
		printf( "Maximum size of each dimension of a grid: (%d, %d, %d)\n",prop.maxGridSize[0], prop.maxGridSize[1], prop.maxGridSize[2] );
		printf( "\n" );
	}

}

