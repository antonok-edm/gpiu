#include <cuda_runtime.h>
#include <curand.h>
#include <common.h>
#include <ctime>

void genCoordsKernel(float *X, float *Y, int numPoints) {
	curandGenerator_t gen;
	CURAND_CALL(curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT));
	CURAND_CALL(curandSetPseudoRandomGeneratorSeed(gen, time(0)));

	CURAND_CALL(curandGenerateUniform(gen, X, numPoints));
	CURAND_CALL(curandGenerateUniform(gen, Y, numPoints));

	CURAND_CALL(curandDestroyGenerator(gen));
}

__global__ void findCircleProportionKernel(const float *X, const float *Y, float *R, int numPoints) {
	int i = blockDim.x * blockIdx.x + threadIdx.x;

	if(i < numPoints) {
		R[i] = ((X[i]-.5)*(X[i]-.5) + (Y[i]-.5)*(Y[i]-.5) <= .25);
	}
}

double kernel_calculation(int numPoints, size_t size, double *runtime) {
	clock_t start = clock();

	float *d_X = NULL;
	float *d_Y = NULL;
	float *d_R = NULL;
	CUDA_CALL(cudaMalloc((void **)&d_X, size));
	CUDA_CALL(cudaMalloc((void **)&d_Y, size));
	CUDA_CALL(cudaMalloc((void **)&d_R, size));
	float *k_R = (float *)malloc(size);
	int threadsPerBlock = 1024;
	int blocksPerGrid =(numPoints + threadsPerBlock - 1) / threadsPerBlock;

	genCoordsKernel(d_X, d_Y, numPoints);
	findCircleProportionKernel<<<blocksPerGrid, threadsPerBlock>>>(d_X, d_Y, d_R, numPoints);
	CUDA_CALL(cudaGetLastError());
	CUDA_CALL(cudaMemcpy(k_R, d_R, size, cudaMemcpyDeviceToHost));
	int kernel_proportion = getSum(k_R, numPoints);

	cudaFree(d_X);
	cudaFree(d_Y);
	cudaFree(d_R);
	free(k_R);

	clock_t end = clock();
	*runtime = difftime(end, start)/CLOCKS_PER_SEC;
	return (double) kernel_proportion / (double) numPoints * 4;
}
