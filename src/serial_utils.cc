#include <cstdlib>
#include <ctime>

void genCoordsSerial(float *X, float *Y, int numPoints) {
	srand(time(0));
	for(int i = 0; i < numPoints; i++) {
		X[i] = (float) rand() / RAND_MAX;
		Y[i] = (float) rand() / RAND_MAX;
	}
}

void findCircleProportionSerial(const float *X, const float *Y, float *R, int numPoints) {
	for(int i = 0; i < numPoints; i++) {
		R[i] = ((X[i]-.5)*(X[i]-.5) + (Y[i]-.5)*(Y[i]-.5) <= .25);
	}
}

double serial_calculation(int numPoints, size_t size, double *runtime) {
	clock_t start = clock();

	float *s_X = (float *)malloc(size);
	float *s_Y = (float *)malloc(size);
	float *s_R = (float *)malloc(size);
	
	genCoordsSerial(s_X, s_Y, numPoints);
	findCircleProportionSerial(s_X, s_Y, s_R, numPoints);
	int serial_proportion = getSum(s_R, numPoints);

	free(s_X);
	free(s_Y);
	free(s_R);

	clock_t end = clock();
	*runtime = difftime(end, start)/CLOCKS_PER_SEC;
	return (double) serial_proportion / (double) numPoints * 4;
}
