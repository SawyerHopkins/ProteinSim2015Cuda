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
			float * x;
			float * y;
			float * z;
			//Contains the velocity information for x,y,z cordinates.
			float * vx;
			float * vy;
			float * vz;
			//Contains the radius of each particle.
			float r;

			//Contains the system information.
			int boxSize;

			//Seed for random system initialization.
			int seed;

			//Base initialization
			void init();

		public:

/*-----------------------------------------*/
/*-----------SYSTEM CONSTRUCTION-----------*/
/*-----------------------------------------*/

			//The number of particles/points
			int arrSize;

			//Constructor/Destructor
			points(int nParticles, float radius);
			points( const points &obj );
			~points();

/*-----------------------------------------*/
/*--------------SYSTEM GETTERS-------------*/
/*-----------------------------------------*/

			//Getters for each spacial cordinate.
			float getX(int index) { return *(x+index); }
			float getY(int index) { return *(y+index); }
			float getZ(int index) { return *(z+index); }
			float getR() { return r; }
			float getBoxSize() { return boxSize; }

			//Getters for each velocity cordinate.
			float getVX(int index) { return *(vx+index); }
			float getVY(int index) { return *(vy+index); }
			float getVZ(int index) { return *(vz+index); }

/*-----------------------------------------*/
/*--------------SYSTEM SETTERS-------------*/
/*-----------------------------------------*/

			//Standard setters for position
			void setX( int index, float val ) { *(x+index) = utilities::safeMod(val,boxSize); }
			void setY( int index, float val ) { *(y+index) = utilities::safeMod(val,boxSize); }
			void setZ( int index, float val ) { *(z+index) = utilities::safeMod(val,boxSize); }
			void setR( float val ) { r = val; }
			void setAllPos( int index, float xVal, float yVal, float zVal);

			//Setters for velocity
			void setVX( int index, float val ) { *(vx+index) = val; }
			void setVY( int index, float val ) { *(vy+index) = val; }
			void setVZ( int index, float val ) { *(vz+index) = val; }
			void setAllVel( int index, float vxVal, float vyVal, float vzVal);

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
			void init(float concentration);
			void init(float concentration, int seedling);
			void initCheck(std::mt19937* gen, std::uniform_real_distribution<double>* distribution);

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
