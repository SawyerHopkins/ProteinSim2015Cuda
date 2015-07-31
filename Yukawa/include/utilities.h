#ifndef UTILITIES_H
#define UTILITIES_H

#include <fstream>
#include <sstream>
#include <iomanip>
#include <string>
#include <random>
#include <vector>
#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>
#include <map>
#include "error.h"
#include "timer.h"

namespace utilities
{

	#define __BLACK "\033[0;30m"
	#define __RED "\033[0;31m"
	#define __GREEN "\033[0;32m"
	#define __BROWN "\033[0;33m"
	#define __BLUE "\033[0;34m"
	#define __MAGENTA "\033[0;35m"
	#define __CYAN "\033[0;36m"
	#define __GREY "\033[0;37m"
	#define __NORMAL "\033[0m"

	enum Colour { Black, Red, Green, Brown, Blue, Magenta, Cyan, Grey, Normal };

	/**
	 * @class util
	 * @author Sawyer Hopkins
	 * @date 06/27/15
	 * @file utilities.h
	 * @brief Contains some useful math wrappers.
	 */
	class util
	{
		public:

			//Header Version.
			static const int version = 1;

			/**
			 * @brief fmod can't do negative numbers so use this.
			 * @param val The value of the perform the division on.
			 * @param base The base of the modular divison.
			 * @return 
			 */
			static double safeMod(double val, double base);

			/**
			 * @brief Looks for PBC check on original position.
			 * @param val0 Original position.
			 * @param val New position.
			 * @param base The size of the system.
			 * @return The new old position.
			 */
			static double safeMod0(double val0, double val, double base);

			/**
			 * @brief Method for getting distance between two points.
			 * @param X,Y,Z The position of the first particle
			 * @param X1,Y1,Z1 The position of the second particle.
			 * @param L The size of the system.
			 * @return The distance between the two particles.
			 */
			static double pbcDist(double X,double Y, double Z,double X1, double Y1,double Z1,double L);

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
			 * @param acc The array hold the unit vectors.
			 */
			static void unitVectorSimple(double dX, double dY, double dZ, double r, double (&acc)[3]);

			/**
			 * @brief Alternative method for getting the normalized distance between two particles.
			 * @param X,Y,Z The position of the first particle
			 * @param X1,Y1,Z1 The position of the second particle.
			 * @param acc The array to hold the unit vectors
			 * @param r The distance between the particles.
			 * @param L The size of the box.
			 */
			static void unitVectorAdv(double X,double Y, double Z,double X1, double Y1,double Z1,double (&acc)[3],double r,int L);

			static void setTerminalColour(utilities::Colour c);

			static void writeTerminal(std::string text, utilities::Colour c);

	};

}

#endif // UTILITIES_H
