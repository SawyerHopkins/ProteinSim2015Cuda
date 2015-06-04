#include "point.h"

namespace mathTools
{

/*-----------------------------------------*/
/*-----------SYSTEM CONSTRUCTION-----------*/
/*-----------------------------------------*/

	//Creates a new set if 'size' number of particles all located at the origin.
	points::points(int nParticles, double radius, double t_initial) : 
	x(new double[nParticles]), y(new double[nParticles]), z(new double[nParticles]), 
	vx(new double[nParticles]), vy(new double[nParticles]), vz(new double[nParticles])
	{
		arrSize = nParticles;
		r = radius;
		t_init = t_initial;
		seed=0;
	}

	//Copy constructor.
	points::points(const points &obj) : 
	x(new double[obj.arrSize]), y(new double[obj.arrSize]), z(new double[obj.arrSize]), 
	vx(new double[obj.arrSize]), vy(new double[obj.arrSize]), vz(new double[obj.arrSize])
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
		t_init = obj.t_init;
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
		delete[] &t_init;
	}

/*-----------------------------------------*/
/*--------------SYSTEM SETTERS-------------*/
/*-----------------------------------------*/

	//Function to quickly set all three spacial cordinates of a particle.
	void points::setAllPos(int i, double xVal, double yVal, double zVal)
	{
		setX(i,xVal);
		setY(i,yVal);
		setZ(i,zVal);
	}

	//Function to quickly set all three velocity cordinates of a particle.
	void points::setAllVel(int i, double vxVal, double vyVal, double vzVal)
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
		//If there is no inital seed create one.
		if (seed==0)
		{
			std::random_device rd;
			seed=rd();
		}
		//Setup random uniform distribution generator.
		std::mt19937 gen(seed);
		std::uniform_real_distribution<double> distribution(0.0,1.0);
		
		//Iterates through all points.
		for(int i = 0; i <arrSize; i++)
		{
			setX(i, distribution(gen) * boxSize);
			setY(i, distribution(gen) * boxSize);
			setZ(i, distribution(gen) * boxSize);
		}
		initCheck(&gen, &distribution);
		maxwellVelocityInit(&gen, &distribution);
	}

	//Creates a box corresponding to # of Particles / boxSize^3 = concentration. 
	void points::init(double concentration)
	{
		double vP = arrSize*(4.0/3.0)*atan(1)*4*r*r*r;
		boxSize = (int) cbrt(vP / concentration);
		init();
	}

	//Creates a box corresponding to # of Particles / boxSize^3 = concentration. 
	//Seeds the random number generator with value seedling.
	void points::init(double concentration, int seedling)
	{
		seed=seedling;
		boxSize = (int) cbrt(arrSize / concentration);
		init();
	}

	//Looks for overlapping particles in the initial random distribution.
	void points::initCheck(std::mt19937* gen, std::uniform_real_distribution<double>* distribution)
	{
		//Keeps track of how many resolutions we have attempted.
		int counter = 0;

		//Search each particle for overlap.
		for(int i = 0; i < arrSize; i++)
		{
			//Is the problem resolved?
			bool resolution = false;

			//If not loop.
			while (resolution == false)
			{
				//Assume resolution.
				resolution = true;
				for(int j = 0; j < arrSize; j++)
				{
					//Exclude self interation.
					if (i != j)
					{
						//Gets the distance between the two particles.
						double distX = utilities::pbcDist(getX(i),getX(j),boxSize);
						double distY = utilities::pbcDist(getY(i),getY(j),boxSize);
						double distZ = utilities::pbcDist(getZ(i),getZ(j),boxSize);

						double radius = std::sqrt((distX*distX)+(distY*distY)+(distZ*distZ));

						//If the particles are slightly closer than twice their radius resolve conflict.
						if (radius < 2.1*r)
						{
							//Update resolution counter.
							counter++;

							//Throw warnings if stuck in resolution loop.
							if (counter > 10*arrSize)
							{
								std::cout << "Could create initial system.\n";
								std::cout << "Try decreasing particle density.";
							}

							//Assume new system in not resolved.
							resolution = false;

							//Set new uniform random position.
							setX(i, (*distribution)(*gen) * boxSize);
							setY(i, (*distribution)(*gen) * boxSize);
							setZ(i, (*distribution)(*gen) * boxSize);
						}
					}
				}
			}
		}
	}

	void points::maxwellVelocityInit(std::mt19937* gen, std::uniform_real_distribution<double>* distribution)
	{
		double r1,r2;
		double vsum,vsum2;
		double sigold,vsig,ratio;
		int i;
		
		for(i=0; i<arrSize; i++)
		{
			r1=(*distribution)(*gen);
			r2=(*distribution)(*gen);
			setVX(i, sqrt(-2.0 * log(r1) ) * cos(8.0*atan(1)*r2));
		}

		for(i=0; i<arrSize; i++)
		{
			r1=(*distribution)(*gen);
			r2=(*distribution)(*gen);
			setVY(i, sqrt(-2.0 * log(r1) ) * cos(8.0*atan(1)*r2));
		}

		for(i=0; i<arrSize; i++)
		{
			r1=(*distribution)(*gen);
			r2=(*distribution)(*gen);
			setVZ(i, sqrt(-2.0 * log(r1) ) * cos(8.0*atan(1)*r2));
		}
		
		//maxwell for vx//
		vsum=0;
		vsum2=0;
		
		for(i=0; i<arrSize; i++)
		{
			vsum=vsum+getVX(i);
			vsum2=vsum2+(getVX(i)*getVX(i));
		}
		vsum=vsum/arrSize;
		vsum2=vsum2/arrSize;
		sigold=sqrt(vsum2-(vsum*vsum));

		vsig= sqrt(double(t_init)) ;
		ratio=vsig/sigold;

		for(i=0; i<arrSize; i++)
		{
			setVX(i,ratio*(getVX(i)-vsum));
		}
	////////////////////

		//maxwell for vy//
		vsum=0;
		vsum2=0;
		
		for(i=0; i<arrSize; i++)
		{
			vsum=vsum+getVY(i);
			vsum2=vsum2+(getVY(i)*getVY(i));
		}
		vsum=vsum/arrSize;
		vsum2=vsum2/arrSize;
		sigold=sqrt(vsum2-(vsum*vsum));

		vsig= sqrt(double(t_init)) ;
		ratio=vsig/sigold;

		for(i=0; i<arrSize; i++)
		{
			setVY(i,ratio*(getVY(i)-vsum));
		}
	////////////////////

		//maxwell for vz//
		vsum=0;
		vsum2=0;
		
		for(i=0; i<arrSize; i++)
		{
			vsum=vsum+getVZ(i);
			vsum2=vsum2+(getVZ(i)*getVZ(i));
		}
		vsum=vsum/arrSize;
		vsum2=vsum2/arrSize;
		sigold=sqrt(vsum2-(vsum*vsum));

		vsig= sqrt(double(t_init)) ;
		ratio=vsig/sigold;

		for(i=0; i<arrSize; i++)
		{
			setVZ(i,ratio*(getVZ(i)-vsum));
		}
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

