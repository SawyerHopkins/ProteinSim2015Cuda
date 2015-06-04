#ifndef POINT_H
#define POINT_H
#include <vector>
#include <iostream>
#include <random>
#include <string.h>
#include "GnuPlotter.h"
#include "utilities.h"

//Contains point and matrix manipulation.
namespace mathTools
{
	//A set of points using parallel arrays.
	class points
	{
	private:

/*-----------------------------------------*/
/*------------SYSTEM VARIABLES-------------*/
/*-----------------------------------------*/

			//Contains the spacial information for x,y,z cordinates.
			double * x;
			double * y;
			double * z;
			//Contains the velocity information for x,y,z cordinates.
			double * vx;
			double * vy;
			double * vz;
			//Contains the radius of each particle.
			double r;

			//Contains the system information.
			int boxSize;

			//Seed for random system initialization.
			int seed;

			//Initial system temperature
			double t_init;

			//Base initialization
			void init();

		public:

/*-----------------------------------------*/
/*-----------SYSTEM CONSTRUCTION-----------*/
/*-----------------------------------------*/

			//The number of particles/points
			int arrSize;

			//Constructor/Destructor
			points(int nParticles, double radius, double t_initial);
			points( const points &obj );
			~points();

/*-----------------------------------------*/
/*--------------SYSTEM GETTERS-------------*/
/*-----------------------------------------*/

			//Getters for each spacial cordinate.
			double getX(int index) { return *(x+index); }
			double getY(int index) { return *(y+index); }
			double getZ(int index) { return *(z+index); }
			double getR() { return r; }
			double getBoxSize() { return boxSize; }

			//Getters for each velocity cordinate.
			double getVX(int index) { return *(vx+index); }
			double getVY(int index) { return *(vy+index); }
			double getVZ(int index) { return *(vz+index); }

/*-----------------------------------------*/
/*--------------SYSTEM SETTERS-------------*/
/*-----------------------------------------*/

			//Standard setters for position
			void setX( int index, double val ) { *(x+index) = utilities::safeMod(val,boxSize); }
			void setY( int index, double val ) { *(y+index) = utilities::safeMod(val,boxSize); }
			void setZ( int index, double val ) { *(z+index) = utilities::safeMod(val,boxSize); }
			void setR( double val ) { r = val; }
			void setAllPos( int index, double xVal, double yVal, double zVal);

			//Setters for velocity
			void setVX( int index, double val ) { *(vx+index) = val; }
			void setVY( int index, double val ) { *(vy+index) = val; }
			void setVZ( int index, double val ) { *(vz+index) = val; }
			void setAllVel( int index, double vxVal, double vyVal, double vzVal);

/*-----------------------------------------*/
/*---------------SYSTEM DEBUG--------------*/
/*-----------------------------------------*/

			//Debugging tools to write position and velocity of a particle.
			void writePosition(int index) { std::cout << *(x+index) << "," << *(y+index) << "," << *(z+index) << "\n"; }
			void writeVelocity(int index) { std::cout << *(vx+index) << "," << *(vy+index) << "," << *(vz+index) << "\n"; }

/*-----------------------------------------*/
/*-----------SYSTEM INITIALIZATION---------*/
/*-----------------------------------------*/

			//Creates an initial distribution of the particles.
			void init(double concentration);
			void init(double concentration, int seedling);
			void initCheck(std::mt19937* gen, std::uniform_real_distribution<double>* distribution);
			void maxwellVelocityInit(std::mt19937* gen, std::uniform_real_distribution<double>* distribution);

/*-----------------------------------------*/
/*--------------SYSTEM OUTPUT--------------*/
/*-----------------------------------------*/

			//Sends the system to GNUPlot.
			void plot();

			//Writes the system as CSV.
			void writeSystem(std::string name);

	};

}

#endif // POINT_H
