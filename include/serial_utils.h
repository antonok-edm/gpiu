#include <cstdlib>
#include <serial_utils.cc>

void genCoordsSerial(float *X, float *Y, int numPoints);

void findCircleProportionSerial(const float *X, const float *Y, float *R, int numPoints);

double serial_calculation(int numPoints, size_t size, double *runtime);
