#include "point.h"
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <random>
namespace mathTools
{

	//Creates a new set if 'size' number of particles all located at the origin.
	points::points(int nParticles, int size) : 
	x(new float[nParticles]), y(new float[nParticles]), z(new float[nParticles]), 
	vx(new float[nParticles]), vy(new float[nParticles]), vz(new float[nParticles]),
	r(new float[nParticles])
	{
		arrSize = nParticles;
		boxSize = size;
	}

	//Copy constructor.
	points::points(const points &obj) : 
	x(new float[obj.arrSize]), y(new float[obj.arrSize]), z(new float[obj.arrSize]), 
	vx(new float[obj.arrSize]), vy(new float[obj.arrSize]), vz(new float[obj.arrSize]),
	r(new float[obj.arrSize])
	{
		//Copies the stored values rather than pointers.
		arrSize=obj.arrSize;
		*x = *obj.x;
		*y = *obj.y;
		*z = *obj.z;
		*vx = *obj.vx;
		*vy = *obj.vy;
		*vz = *obj.vz;
		boxSize = obj.boxSize;
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
		delete[] r;
		delete[] &boxSize;
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
		std::default_random_engine generator;
		std::uniform_real_distribution<double> distribution(0.0,1.0);
		//Iterates through all points.
		for(int i = 0; i <arrSize; i++)
		{
			setX(i, distribution(generator) * boxSize);
			setVX(i, distribution(generator) * boxSize);
			setY(i, distribution(generator) * boxSize);
			setVY(i, distribution(generator) * boxSize);
			setZ(i, distribution(generator) * boxSize);
			setVZ(i, distribution(generator) * boxSize);
			setR(i, 1.0);
		}
	}

	void points::init(float concentration)
	{
		boxSize = (int) cbrt(arrSize / concentration);
		init();
	}

}

