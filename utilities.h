#ifndef UTILITIES_H
#define UTILITIES_H

#include <iostream>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <math.h>
#include <string>

namespace utilities
{

	//Contains some useful math wrappers.
	class util
	{
		public:
			/**
			 * @brief fmod can't do negative numbers so use this.
			 * @param val The value of the perform the division on.
			 * @param base The base of the modular divison.
			 * @return 
			 */
			static double safeMod(double val, double base);

			/**
			 * @brief Gets the distance between to points.
			 * @param v1 The position of the reference particle.
			 * @param v2 The position of the second particle.
			 * @param size The size of the system.
			 * @return 
			 */
			static double pbcDist(double v1, double v2, double size);

			//
			/**
			 * @brief Shows the completed progress in the console.
			 * @param x0 The amount done
			 * @param n The total amount to be done. 
			 * @param counter A logic counter.
			 * @param w The width of the progress bar. Default 50.
			 */
			static void loadBar(double x0, int n, long counter ,int w = 50);

			/**
			 * @brief Normalizes the distances to create a unit vector in &acc[3].
			 * @param dX The distance in the X direction.
			 * @param dY The distance in the Y direction.
			 * @param dZ The distance in the Z direction.
			 * @param r The magnitude of the distance.
			 */
			static void unitVector(double dX, double dY, double dZ, double r, double (&acc)[3]);

	};

}

#endif // UTILITIES_H
