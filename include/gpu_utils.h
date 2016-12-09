#include <cuda_runtime.h>
#include <curand.h>
#include <gpu_utils.cu>

void genCoordsKernel(float *X, float*Y, int numPoints);

__global__ void findCircleProportionKernel(const float *X, const float *Y, float *R, int numPoints);

double kernel_calculation(int numPoints, size_t size, double *runtime);
