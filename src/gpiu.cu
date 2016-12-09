#include <iostream>
#include <iomanip>
#include <cmath>
#include <cstdlib>
#include <gpu_utils.h>
#include <serial_utils.h>

/*
CuRAND generates 2 arrays of random x and y coordinates from 0 to 1
A kernel function checks if the coordinates fall within a circle of radius 1/2
and returns result array of 1s and 0s.

Two serial functions performs the same tasks, but element-wise.

A sum function sums up the 1s in each result array.
The sum is divided by the total number of points and multiplied by 4 to approximate PI.
*/

int main() {
	size_t size;
	double runtime;
	double kernel_pi;
	double kernel_time;
	double serial_pi;
	double serial_time;

	std::cout << "gPIu - measure the speed of pi calculation with serial vs. parallel processing.\n";
	std::cout << "              \033[7m____________________________SERIAL\033[0m  \033[7m____________________________KERNEL\033[0m\n";
	std::cout << "  \033[7m____Points\033[0m  \033[7m________Pi\033[0m  \033[7m______%Err\033[0m  \033[7m______Time\033[0m  \033[7m________Pi\033[0m  \033[7m______%Err\033[0m  \033[7m______Time\033[0m  \033[7m__%Runtime\033[0m\n";

	//Increasing numPoints much past 21350000 seems to break the program somehow...
	for(int numPoints = 500000; numPoints <= 21000000; numPoints += 500000) {
		size = numPoints * sizeof(float);

		serial_pi = serial_calculation(numPoints, size, &runtime);
		serial_time = runtime;
		kernel_pi = kernel_calculation(numPoints, size, &runtime);
		kernel_time = runtime;	

		std::cout << std::setw(12) << numPoints <<\
			std::setw(12) << std::setprecision(5) << serial_pi <<\
			std::setw(12) << std::setprecision(5) << (serial_pi-M_PI)/M_PI*100 <<\
			std::setw(12) << std::setprecision(5) << serial_time <<\
			std::setw(12) << std::setprecision(5) << kernel_pi <<\
			std::setw(12) << std::setprecision(5) << (kernel_pi-M_PI)/M_PI*100 <<\
			std::setw(12) << std::setprecision(5) << kernel_time <<\
			std::setw(12) << std::setprecision(5) << kernel_time/serial_time*100 << "\n";
	}

	return 0;
}
