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
		delete[] x;
		delete[] y;
		delete[] z;
		delete[] vx;
		delete[] vy;
		delete[] vz;
	}

	void points::setAllPos(int i, float xVal, float yVal, float zVal)
	{
		*(x+i)=xVal;
		*(y+i)=yVal;
		*(z+i)=zVal;
	}

	void points::setAllVel(int i, float vxVal, float vyVal, float vzVal)
	{
		*(vx+i)=vxVal;
		*(vy+i)=vyVal;
		*(vz+i)=vzVal;
	}

	//Creates a random distribution of the initial points
	void points::init()
	{
		//Iterates through all points.
		for(int i=0; i<arr_size; i++)
		{
			setX(i, randomPos);
			setY(i, randomPos);
			setZ(i, randomPos);
		}
	}

	//A random distribution generator.
	static float randomPos(float x)
	{
		//Random number between 1 and 10.
		return rand() % 10 + 1;
	}

}

