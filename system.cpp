#include "system.h"

namespace simulation
{

	/*-----------------------------------------*/
	/*----------CONSTRUCTOR/DESTRUCTOR---------*/
	/*-----------------------------------------*/

	system::system(int nPart, float conc, int scale, float r, float sysTemp, int rnd, integrators::I_integrator* sysInt)
	{

		//Set the random number generator seed.
		seed = rnd;

		//Sets the system temperature.
		temp = sysTemp;

		//Set the number of particles.
		nParticles = nPart;

		//Set the integration method.
		integrator = sysInt;

		//Create a box based on desired concentration.
		double vP = nPart*(4.0/3.0)*atan(1)*4*r*r*r;
		boxSize = (int) cbrt(vP / conc);

		//Sets the actual concentration.
		concentration = vP/pow(boxSize,3.0);

		std::cout << "\nSystem concentration: " << concentration << "\n";

		//Calculates the number of cells needed.
		cellSize = boxSize / scale;
		int numCells = pow(scale,3.0);

		//Create cells.
		initCells(numCells, scale);

		//Create particles.
		initParticles();

	}

	system::~system()
	{
	}

	/*-----------------------------------------*/
	/*---------------SYSTEM INIT---------------*/
	/*-----------------------------------------*/

	//Create cells and initialize their neighbors.
	void system::initCells(int numCells, int scale)
	{

		cells = new cell***[scale];
		for (int i = 0; i < scale; i++)
		{
			cells[i] = new cell**[scale];
			for (int j = 0; j < scale; j++)
			{
				cells[i][j] = new cell*[scale];
			}
		}

		for (int x = 0; x < scale; x++)
		{
			for (int y = 0; y < scale; y++)
			{
				for (int z = 0; z < scale; z++)
				{
					cells[x][y][z] = new cell();
				}
			}
		}
		
	}

	//Create particles and assign initial cells.
	void system::initParticles()
	{

		particles = new particle*[nParticles];

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
		for(int i = 0; i < nParticles; i++)
		{
			particles[i] = new particle();
			particles[i]->setX( distribution(gen) * boxSize , boxSize);
			particles[i]->setY( distribution(gen) * boxSize , boxSize);
			particles[i]->setZ( distribution(gen) * boxSize , boxSize);
		}

		//Checks the system for overlap.
		initCheck(&gen, &distribution);

		//Set initial velocity.
		maxwellVelocityInit(&gen, &distribution);

		//Assign cells.
		for(int i = 0; i < nParticles; i++)
		{
			int cX = particles[i]->getX() / cellSize;
			int cY = particles[i]->getY() / cellSize;
			int cZ = particles[i]->getZ() / cellSize;

			particles[i]->setCell(cX,cY,cZ);
			cells[cX][cY][cZ]->addMember(particles[i]);

		}

	}

	//Looks for overlapping particles in the initial random distribution.
	void system::initCheck(std::mt19937* gen, std::uniform_real_distribution<double>* distribution)
	{
		//Keeps track of how many resolutions we have attempted.
		int counter = 0;

		//Search each particle for overlap.
		for(int i = 0; i < nParticles; i++)
		{
			//Is the problem resolved?
			bool resolution = false;

			//If not loop.
			while (resolution == false)
			{
				//Assume resolution.
				resolution = true;
				for(int j = 0; j < nParticles; j++)
				{
					//Exclude self interation.
					if (i != j)
					{
						//Gets the distance between the two particles.
						double distX = utilities::pbcDist(particles[i]->getX(),particles[j]->getX(),boxSize);
						double distY = utilities::pbcDist(particles[i]->getY(),particles[j]->getY(),boxSize);
						double distZ = utilities::pbcDist(particles[i]->getZ(),particles[j]->getZ(),boxSize);

						double radius = std::sqrt((distX*distX)+(distY*distY)+(distZ*distZ));

						//Gets the sum of the particle radius.
						double r = particles[i]->getRadius() + particles[j]->getRadius();

						//If the particles are slightly closer than twice their radius resolve conflict.
						if (radius < 2.1*r)
						{
							//Update resolution counter.
							counter++;

							//Throw warnings if stuck in resolution loop.
							if (counter > 10*nParticles)
							{
								std::cout << "Could create initial system.\n";
								std::cout << "Try decreasing particle density.";
							}

							//Assume new system in not resolved.
							resolution = false;

							//Set new uniform random position.
							particles[i]->setX( (*distribution)(*gen) * boxSize , boxSize );
							particles[i]->setY( (*distribution)(*gen) * boxSize , boxSize );
							particles[i]->setZ( (*distribution)(*gen) * boxSize , boxSize );
						}
					}
				}
			}
		}
	}

	//Create an initial maxwell distribution of velocities at the specified system temperature.
	void system::maxwellVelocityInit(std::mt19937* gen, std::uniform_real_distribution<double>* distribution)
	{
		double r1,r2;
		double vsum,vsum2;
		double sigold,vsig,ratio;
		int i;
		
		for(i=0; i<nParticles; i++)
		{
			r1=(*distribution)(*gen);
			r2=(*distribution)(*gen);

			particles[i]->setVX(sqrt(-2.0 * log(r1) ) * cos(8.0*atan(1)*r2));
		}

		for(i=0; i<nParticles; i++)
		{
			r1=(*distribution)(*gen);
			r2=(*distribution)(*gen);
			particles[i]->setVY(sqrt(-2.0 * log(r1) ) * cos(8.0*atan(1)*r2));
		}

		for(i=0; i<nParticles; i++)
		{
			r1=(*distribution)(*gen);
			r2=(*distribution)(*gen);
			particles[i]->setVZ(sqrt(-2.0 * log(r1) ) * cos(8.0*atan(1)*r2));
		}
		
		//maxwell for vx//
		vsum=0;
		vsum2=0;
		
		for(i=0; i<nParticles; i++)
		{
			double vx = particles[i]->getVX();
			vsum=vsum+vx;
			vsum2=vsum2+(vx*vx);
		}
		vsum=vsum/nParticles;
		vsum2=vsum2/nParticles;
		sigold=sqrt(vsum2-(vsum*vsum));

		vsig= sqrt(temp) ;
		ratio=vsig/sigold;

		for(i=0; i<nParticles; i++)
		{
			particles[i]->setVX(ratio*(particles[i]->getVX()-vsum));
		}
	////////////////////

		//maxwell for vy//
		vsum=0;
		vsum2=0;
		
		for(i=0; i<nParticles; i++)
		{
			double vy = particles[i]->getVY();
			vsum=vsum+vy;
			vsum2=vsum2+(vy*vy);
		}
		vsum=vsum/nParticles;
		vsum2=vsum2/nParticles;
		sigold=sqrt(vsum2-(vsum*vsum));

		vsig= sqrt(temp) ;
		ratio=vsig/sigold;

		for(i=0; i<nParticles; i++)
		{
			particles[i]->setVY(ratio*(particles[i]->getVY()-vsum));
		}
	////////////////////

		//maxwell for vz//
		vsum=0;
		vsum2=0;
		
		for(i=0; i<nParticles; i++)
		{
			double vz = particles[i]->getVZ();
			vsum=vsum+vz;
			vsum2=vsum2+(vz*vz);
		}
		vsum=vsum/nParticles;
		vsum2=vsum2/nParticles;
		sigold=sqrt(vsum2-(vsum*vsum));

		vsig= sqrt(temp) ;
		ratio=vsig/sigold;

		for(i=0; i<nParticles; i++)
		{
			particles[i]->setVZ(ratio*(particles[i]->getVZ()-vsum));
		}
	}

	/*-----------------------------------------*/
	/*------------PARTICLE HANDLING------------*/
	/*-----------------------------------------*/

	void system::moveParticle(int index, float x, float y, float z)
	{
		//New cell
		int cX = particles[index]->getX() / cellSize;
		int cY = particles[index]->getY() / cellSize;
		int cZ = particles[index]->getZ() / cellSize;

		//Old cell
		int cX0 = particles[index]->getCX();
		int cY0 = particles[index]->getCY();
		int cZ0 = particles[index]->getCZ();

		//If cell has changed
		if ((cX != cX0) || (cY != cY0) || (cZ != cZ0))
		{
			//Remove from old.
			cells[cX0][cY0][cZ0]->removeMember(particles[index]->getIndex());
			//Add to new.
			cells[cX][cY][cZ]->addMember(particles[index]);
		}

		//Update the position of the particle.
		particles[index]->setX(x,boxSize);
		particles[index]->setY(x,boxSize);
		particles[index]->setZ(x,boxSize);

	}

}

