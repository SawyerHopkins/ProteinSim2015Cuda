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
			static float safeMod(float val, float base);

			//gets the distance between to points while
			//considering periodic boundary conditions.
			static float pbcDist(float v1, float v2, float size);

			//Shows the completed progress in the console.
			static void loadBar(int x, int n, int w = 50);

			static void unitVector(float dX, float dY, float dZ, float r, float (&acc)[3]);

	};

}

#endif // UTILITIES_H
