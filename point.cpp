#include "point.h"
#include <stdlib.h>
#include <time.h>

namespace mathTools
{

	//Function declarations.
	static float randomPos(float x);

	//Creates a new set if 'size' number of particles all located at the origin.
	points::points(int size) : x(new float[size]), y(new float[size]), z(new float[size])
	{
		arr_size = size;
	}

	//Releases the memory blocks
	points::~points()
	{
		delete[] points::x;
		delete[] points::y;
		delete[] points::z;
		delete[] points::vx;
		delete[] points::vy;
		delete[] points::vz;
	}

	//Creates a random distribution of the initial points
	void points::init()
	{
		//Iterates through all points.
		for(int i=0; i<points::arr_size; i++)
		{
			points::setX(i, randomPos);
			points::setY(i, randomPos);
			points::setZ(i, randomPos);
		}
	}

	//A random distribution generator.
	static float randomPos(float x)
	{
		//Random number between 1 and 10.
		return rand() % 10 + 1;
	}

}

