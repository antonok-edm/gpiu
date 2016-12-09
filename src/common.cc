int getSum(const float *R, int numPoints) {
	int sum = 0;
	for(int i = 0; i < numPoints; i++) {
		sum += R[i];
	}
	return sum;
}
