#ifndef PARTICLE_H
#define PARTICLE_H
#include "utilities.h"

namespace simulation
{

	class particle
	{

		/*-----------------------------------------*/
		/*-----------PARTICLE VARIABLES------------*/
		/*-----------------------------------------*/

		private:
			//Contains the current spacial information for x,y,z cordinates.
			double x;
			double y;
			double z;

			//Contains the previous spacial information for x,y,z cordinates.
			double x0;
			double y0;
			double z0;

			//Contains the velocity information for x,y,z cordinates.
			double vx;
			double vy;
			double vz;

			//Contains the current force information for the x,y,z cordinates.
			double fx;
			double fy;
			double fz;

			//Contains the previous force information for the x,y,z cordinates.
			double fx0;
			double fy0;
			double fz0;

			//Contains the radius of each particle.
			double r;

			//Contains the mass of each particle.
			double m;

			//Contains the current cell identification.
			int cx;
			int cy;
			int cz;
			int index;

		public:

			/*-----------------------------------------*/
			/*-----------SYSTEM CONSTRUCTION-----------*/
			/*-----------------------------------------*/

			//Constructor/Destructor
			particle();
			~particle();

			/*-----------------------------------------*/
			/*--------------SYSTEM GETTERS-------------*/
			/*-----------------------------------------*/

			//Getters for current position.
			double getX() { return x; }
			double getY() { return y; }
			double getZ() { return z; }

			//Getters for previous position.
			double getX0() { return x0; }
			double getY0() { return y0; }
			double getZ0() { return z0; }

			//Getters for velocity.
			double getVX() { return vx; }
			double getVY() { return vy; }
			double getVZ() { return vz; }
 
			//Getters for current force.
			double getFX() { return fx; }
			double getFY() { return fy; }
			double getFZ() { return fz; }

			//Getters for previous force.
			double getFX0() { return fx0; }
			double getFY0() { return fy0; }
			double getFZ0() { return fz0; }

			//Getters for containing cell.
			double getCX() { return cx; }
			double getCY() { return cy; }
			double getCZ() { return cz; }
			double getIndex() { return index; }

			//Getters for particle properties
			double getRadius() { return r; }
			double getMass() { return m; }
 
			/*-----------------------------------------*/
			/*--------------SYSTEM SETTERS-------------*/
			/*-----------------------------------------*/
 
			//Setters for current position.
			void setX(float val, float boxSize) { x0 = x; x = mathTools::utilities::safeMod(val,boxSize); }
			void setY(float val, float boxSize) { y0 = y; y = mathTools::utilities::safeMod(val,boxSize); }
			void setZ(float val, float boxSize) { z0 = z; z = mathTools::utilities::safeMod(val,boxSize); }

			//Setters for velocity
			void setVX(float val) { vx = val; }
			void setVY(float val) { vy = val; }
			void setVZ(float val) { vz = val; }

			//Setters for current force.
			void setFX(float val) { fx0 = fx; fx = val; }
			void setFY(float val) { fy0 = fy; fy = val; }
			void setFZ(float val) { fz0 = fz; fz = val; }

			//Setter for containing cell.
			void setCell(int x, int y, int z) { cx = x; cy = y; cz = z; }
			void setIndex(int i) { index = i; }

			//Setters for particle properties
			void setRadius(float val) { r = val; }
			void setMass(float val) { m = val; }

	};

}

#endif // PARTICLE_H
