#include <cuda_runtime.h>
#include <curand.h>
#include <iostream>
#include <common.cc>

// Error handling calls modified from DNNMark
#define CURAND_CALL(x) \
do {\
	if(x != CURAND_STATUS_SUCCESS) {\
		std::cerr << "CURAND failed in " << __FILE__ << " at line " << __LINE__ << ".\n";\
		exit(EXIT_FAILURE);\
	}\
} while(0)\

#define CUDA_CALL(x) \
do {\
	cudaError_t err = x;\
	if(err != cudaSuccess) {\
		std::cerr << "CUDA failed in " <<  __FILE__ << " at line " << __LINE__ << " with code " << cudaGetErrorString(err) << ".\n";\
		exit(EXIT_FAILURE);\
	}\
} while(0)\

int getSum(const float *R, int numPoints);
