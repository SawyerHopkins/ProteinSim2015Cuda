#ifndef POINT_H
#define POINT_H
#include <vector>
#include <iostream>
#include <math.h>

//Contains point and matrix manipulation.
namespace mathTools
{
	//A set of points using parallel arrays.
	class points
	{
		private:

			//Contains the spacial information for x,y,z cordinates.
			float * x;
			float * y;
			float * z;
			//Contains the velocity information for x,y,z cordinates.
			float * vx;
			float * vy;
			float * vz;
			//Contains the radius of each particle
			float * r;

			//Contains the system information
			int boxSize;

		public:

			//The number of particles/points
			int arrSize;

			//Constructor/Destructor
			points(int nParticles, int size);
			points( const points &obj );
			~points();

			//Getters for each spacial cordinate.
			float getX(int index) { return *(x+index); }
			float getY(int index) { return *(y+index); }
			float getZ(int index) { return *(z+index); }
			float getR(int index) { return *(r+index); }

			//Getters for each velocity cordinate.
			float getVX(int index) { return *(vx+index); }
			float getVY(int index) { return *(vy+index); }
			float getVZ(int index) { return *(vz+index); }

			//Standard setters for position
			void setX( int index, float val ) { *(x+index) = fmod(val,boxSize); }
			void setY( int index, float val ) { *(y+index) = fmod(val,boxSize); }
			void setZ( int index, float val ) { *(z+index) = fmod(val,boxSize); }
			void setR( int index, float val ) { *(r+index) = fmod(val,boxSize); }
			void setAllPos( int index, float xVal, float yVal, float zVal);

			//Setters for velocity
			void setVX( int index, float val ) { *(vx+index) = val; }
			void setVY( int index, float val ) { *(vy+index) = val; }
			void setVZ( int index, float val ) { *(vz+index) = val; }
			void setAllVel( int index, float vxVal, float vyVal, float vzVal);

			//Debugging tools to write position and velocity of a particle.
			void writePosition(int index) { std::cout << *(x+index) << "," << *(y+index) << "," << *(z+index) << "\n"; }
			void writeVelocity(int index) { std::cout << *(vx+index) << "," << *(vy+index) << "," << *(vz+index) << "\n"; }

			//Creates an initial distribution of the particles.
			void init();
			void init(float concentration);
	};

}

#endif // POINT_H
