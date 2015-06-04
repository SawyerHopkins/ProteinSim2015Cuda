#ifndef UTILITIES_H
#define UTILITIES_H

#include <iostream>
#include <iomanip>
#include <math.h>

namespace mathTools
{

	//Contains some useful math wrappers.
	class utilities
	{
		public:

			//fmod can't do negative numbers so use this.
			static double safeMod(double val, double base);

			//gets the distance between to points while
			//considering periodic boundary conditions.
			static double pbcDist(double v1, double v2, double size);

			//Shows the completed progress in the console.
			static void loadBar(double x0, int n, long counter ,int w = 50);

			//Normalizes the distances to create a unit vector in &acc[3].
			static void unitVector(double dX, double dY, double dZ, double r, double (&acc)[3]);

	};

}

#endif // UTILITIES_H
