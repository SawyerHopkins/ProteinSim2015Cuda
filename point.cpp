#include "point.h"
#include <stdlib.h>
#include <time.h>

namespace mathTools
{

	//Function declarations.
	static float randomPos(float x);

	//Creates a new set if 'size' number of particles all located at the origin.
	points::points(int size) : 
	x(new float[size]), y(new float[size]), z(new float[size]), 
	vx(new float[size]), vy(new float[size]), vz(new float[size])
	{
		arr_size = size;
	}

	//Copy constructor.
	points::points(const points &obj) : 
	x(new float[obj.arr_size]), y(new float[obj.arr_size]), z(new float[obj.arr_size]), 
	vx(new float[obj.arr_size]), vy(new float[obj.arr_size]), vz(new float[obj.arr_size])
	{
		//Copies the stored values rather than pointers.
		arr_size=obj.arr_size;
		*x = *obj.x;
		*y = *obj.y;
		*z = *obj.z;
		*vx = *obj.vx;
		*vy = *obj.vy;
		*vz = *obj.vz;
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

	//Function to quickly set all three spacial cordinates of a particle.
	void points::setAllPos(int i, float xVal, float yVal, float zVal)
	{
		*(x+i)=xVal;
		*(y+i)=yVal;
		*(z+i)=zVal;
	}

	//Function to quickly set all three velocity cordinates of a particle.
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
			setVX(i, randomPos);
			setY(i, randomPos);
			setVY(i, randomPos);
			setZ(i, randomPos);
			setVZ(i, randomPos);
		}
	}

	//A random distribution generator.
	static float randomPos(float x)
	{
		//Random number between 1 and 10.
		return rand() % 10 + 1;
	}

}

