#include "point.h"

namespace mathTools
{

/*-----------------------------------------*/
/*-----------SYSTEM CONSTRUCTION-----------*/
/*-----------------------------------------*/

	//Creates a new set if 'size' number of particles all located at the origin.
	points::points(int nParticles, float radius) : 
	x(new float[nParticles]), y(new float[nParticles]), z(new float[nParticles]), 
	vx(new float[nParticles]), vy(new float[nParticles]), vz(new float[nParticles])
	{
		arrSize = nParticles;
		r = radius;
		seed=0;
	}

	//Copy constructor.
	points::points(const points &obj) : 
	x(new float[obj.arrSize]), y(new float[obj.arrSize]), z(new float[obj.arrSize]), 
	vx(new float[obj.arrSize]), vy(new float[obj.arrSize]), vz(new float[obj.arrSize])
	{
		//Copies the stored values rather than pointers.
		arrSize=obj.arrSize;
		*x = *obj.x;
		*y = *obj.y;
		*z = *obj.z;
		*vx = *obj.vx;
		*vy = *obj.vy;
		*vz = *obj.vz;
		r = obj.r;
		boxSize = obj.boxSize;
		seed=obj.seed;
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
		delete[] &r;
		delete[] &boxSize;
		delete[] &seed;
	}

/*-----------------------------------------*/
/*--------------SYSTEM SETTERS-------------*/
/*-----------------------------------------*/

	//Function to quickly set all three spacial cordinates of a particle.
	void points::setAllPos(int i, float xVal, float yVal, float zVal)
	{
		setX(i,xVal);
		setY(i,yVal);
		setZ(i,zVal);
	}

	//Function to quickly set all three velocity cordinates of a particle.
	void points::setAllVel(int i, float vxVal, float vyVal, float vzVal)
	{
		*(vx+i)=vxVal;
		*(vy+i)=vyVal;
		*(vz+i)=vzVal;
	}

/*-----------------------------------------*/
/*-----------SYSTEM INITIALIZATION---------*/
/*-----------------------------------------*/

	//Creates a random distribution of the initial points
	void points::init()
	{
		if (seed==0)
		{
			std::random_device rd;
			seed=rd();
		}
		std::mt19937 gen(seed);
		std::uniform_real_distribution<double> distribution(0.0,1.0);
		
		//Iterates through all points.
		for(int i = 0; i <arrSize; i++)
		{
			setX(i, distribution(gen) * boxSize);
			setVX(i, 0.0);
			setY(i, distribution(gen) * boxSize);
			setVY(i, 0.0);
			setZ(i, distribution(gen) * boxSize);
			setVZ(i, 0.0);
		}
	}

	//Creates a box corresponding to # of Particles / boxSize^3 = concentration. 
	void points::init(float concentration)
	{
		float vP = arrSize*(4.0/3.0)*atan(1)*4*r*r*r;
		boxSize = (int) cbrt(vP / concentration);
		init();
	}

	//Creates a box corresponding to # of Particles / boxSize^3 = concentration. 
	//Seeds the random number generator with value seedling.
	void points::init(float concentration, int seedling)
	{
		seed=seedling;
		boxSize = (int) cbrt(arrSize / concentration);
		init();
	}

/*-----------------------------------------*/
/*--------------SYSTEM OUTPUT--------------*/
/*-----------------------------------------*/

	//Sends the system to GNUPlot.
	void points::plot()
	{
		Plotting::GnuPlotter::plot(arrSize,x,y,z);
	}

	//Writes the system as CSV.
	void points::writeSystem(std::string name)
	{
		Plotting::GnuPlotter::writeFile(arrSize,x,y,z,name);
	}

}

