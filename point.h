#ifndef POINT_H
#define POINT_H
#include <vector>
#include <verlet.h>
#include "force.h"

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

	public:

		//The number of particles/points
		int arr_size;

		//Constructor/Destructor
		points(int size);
		~points();

		//Getters for each spacial cordinate.
		float getX(int index) { return *(x+index); }
		float getY(int index) { return *(y+index); }
		float getZ(int index) { return *(z+index); }

		//Setters take an arbitrary function of type float->float.
		void setX( int index, float (*f)(float) ) { *(x+index) = (*f)( *(x+index) ); }
		void setY( int index, float (*f)(float) ) { *(y+index) = (*f)( *(y+index) ); }
		void setZ( int index, float (*f)(float) ) { *(z+index) = (*f)( *(z+index) ); }

		//Setters for using I_integrator
		void setX( int index, integrators::I_integrator* f, physics::IForce* g) { *(x+index) = f->nextPosition( *(x+index) , *(vx+index) , g ); }
		void setY( int index, integrators::I_integrator* f, physics::IForce* g) { *(y+index) = f->nextPosition( *(y+index) , *(vx+index) , g ); }
		void setZ( int index, integrators::I_integrator* f, physics::IForce* g) { *(z+index) = f->nextPosition( *(z+index) , *(vx+index) , g ); }


		//Setters take an arbitrary function of type float->float.
		void setVX( int index, float (*f)(float) ) { *(vx+index) = (*f)( *(vx+index) ); }
		void setVY( int index, float (*f)(float) ) { *(vy+index) = (*f)( *(vy+index) ); }
		void setVZ( int index, float (*f)(float) ) { *(vz+index) = (*f)( *(vz+index) ); }

		//Setters for using I_integrator
		void setVX( int index, integrators::I_integrator* f, physics::IForce* g) { *(vx+index) = f->nextPosition( *(x+index) , *(vx+index) , g ); }
		void setVY( int index, integrators::I_integrator* f, physics::IForce* g) { *(vy+index) = f->nextPosition( *(y+index) , *(vx+index) , g ); }
		void setVZ( int index, integrators::I_integrator* f, physics::IForce* g) { *(vz+index) = f->nextPosition( *(z+index) , *(vx+index) , g ); }

		//Creates an initial distribution of the particles.
		void init();

};

}

#endif // POINT_H
