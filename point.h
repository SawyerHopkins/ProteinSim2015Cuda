#ifndef POINT_H
#define POINT_H
#include <vector>

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

		//Standard setters for position
		void setX( int index, float val ) { *(x+index) = val; }
		void setY( int index, float val ) { *(y+index) = val; }
		void setZ( int index, float val ) { *(z+index) = val; }
		void setAllPos( int index, float xVal, float yVal, float zVal);

		//Setters take an arbitrary function of type float->float.
		void setX( int index, float (*f)(float) ) { *(x+index) = (*f)( *(x+index) ); }
		void setY( int index, float (*f)(float) ) { *(y+index) = (*f)( *(y+index) ); }
		void setZ( int index, float (*f)(float) ) { *(z+index) = (*f)( *(z+index) ); }

		//Setters for using I_integrator
		void setVX( int index, float val ) { *(vx+index) = val; }
		void setVY( int index, float val ) { *(vy+index) = val; }
		void setVZ( int index, float val ) { *(vz+index) = val; }
		void setAllVel( int index, float vxVal, float vyVal, float vzVal);

		//Standard setters for velocity
		void setVX( int index, float (*f)(float) ) { *(vx+index) = (*f)( *(vx+index) ); }
		void setVY( int index, float (*f)(float) ) { *(vy+index) = (*f)( *(vy+index) ); }
		void setVZ( int index, float (*f)(float) ) { *(vz+index) = (*f)( *(vz+index) ); }

		//Creates an initial distribution of the particles.
		void init();

};

}

#endif // POINT_H
