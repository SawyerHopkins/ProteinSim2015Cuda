#ifndef PARTICLE_H
#define PARTICLE_H
#include "utilities.h"

namespace simulation
{

	/**
	 * @class particle
	 * @author Sawyer Hopkins
	 * @date 06/27/15
	 * @file particle.h
	 * @brief 
	 */
	class particle
	{

		/********************************************//**
		*----------------SYSTEM VARIABLES----------------
		************************************************/

		private:

			//The particle identifier.
			int name;

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

			//Testing information
			int coorNumber;
			//Effective average potenital.
			double eap;

		public:

			/********************************************//**
			*--------------SYSTEM CONSTRUCTION---------------
			************************************************/

			//Constructor/Destructor
			/**
			 * @brief Creates a new particle with specified name.
			 * @param pid The name of the particle.
			 * @return 
			 */
			particle(int pid);
			/**
			 * @brief Removes particle resources from memory.
			 * @return Nothing.
			 */
			~particle();

			/********************************************//**
			*-----------------SYSTEM GETTERS-----------------
			************************************************/

			//Getters for current position.

			/**
			 * @brief Get the X position
			 * @return  exposes private variable x.
			 */
			const double getX() const { return x; }
			/**
			 * @brief Get the Y position
			 * @return  exposes private variable y.
			 */
			const double getY() const { return y; }
			/**
			 * @brief Get the Z position
			 * @return  exposes private variable z.
			 */
			const double getZ() const { return z; }

			//Getters for previous position.

			/**
			 * @brief Get the previous X position
			 * @return  exposes private variable x0.
			 */
			const double getX0() const { return x0; }
			/**
			 * @brief Get the previous Y position
			 * @return  exposes private variable y0.
			 */
			const double getY0() const { return y0; }
			/**
			 * @brief Get the previous X position
			 * @return  exposes private variable z0.
			 */
			const double getZ0() const { return z0; }

			//Getters for velocity.

			/**
			 * @brief Get the x velocity.
			 * @return  exposes private variable x.
			 */
			const double getVX() const { return vx; }
			/**
			 * @brief Get the y velocity.
			 * @return  exposes private variable y.
			 */
			const double getVY() const { return vy; }
			/**
			 * @brief Get the z velocity.
			 * @return  exposes private variable z.
			 */
			const double getVZ() const { return vz; }
 
			//Getters for current force.

			/**
			 * @brief Get the current x force.
			 * @return  exposes private variable fx.
			 */
			const double getFX() const { return fx; }
			/**
			 * @brief Get the current y force.
			 * @return  exposes private variable fy.
			 */
			const double getFY() const { return fy; }
			/**
			 * @brief Get the current z force.
			 * @return  exposes private variable fz.
			 */
			const double getFZ() const { return fz; }

			//Getters for previous force.

			/**
			 * @brief Get the previous x force.
			 * @return  exposes private variable fx0.
			 */
			const double getFX0() const { return fx0; }
			/**
			 * @brief Get the previous y force.
			 * @return  exposes private variable fy0.
			 */
			const double getFY0() const { return fy0; }
			/**
			 * @brief Get the previous z force.
			 * @return  exposes private variable fz0.
			 */
			const double getFZ0() const { return fz0; }

			//Getters for containing cell.

			/**
			 * @brief Get the x value of the containing cell.
			 * @return  exposes private variable cx.
			 */
			const int getCX() const { return cx; }
			/**
			 * @brief Get the y value of the containing cell.
			 * @return  exposes private variable cy.
			 */
			const int getCY() const { return cy; }
			/**
			 * @brief Get the z value of the containing cell.
			 * @return  exposes private variable cz.
			 */
			const int getCZ() const { return cz; }

			//Getters for particle properties

			/**
			 * @brief Get the radius of the particle.
			 * @return  exposes private variable r.
			 */
			const double getRadius() const { return r; }
			/**
			 * @brief Get the mass of the particle.
			 * @return  exposes private variable m.
			 */
			const double getMass() const { return m; }
			/**
			 * @brief Get the name of the particle.
			 * @return  exposes private variable name.
			 */
			const double getName() const { return name; }
			/**
			 * @brief Returns the coordination number.
			 * @return
			 */
			const int getCoorNumber() const { return coorNumber; }
			/**
			 * @brief Gets average potential.
			 * @return 
			 */
			const int getEAP() const { return eap; }
 			/********************************************//**
			*-----------------SYSTEM SETTERS-----------------
			************************************************/

			//Setters for current position.

			/**
			 * @brief Set the x position of the particle. Handles PBC and updates x0.
			 * @param val The new x position. 
			 * @param boxSize The size of the system box.
			 */
			void setX(double val, double boxSize);
			/**
			 * @brief Set the y position of the particle. Handles PBC and updates y0.
			 * @param val The new y position. 
			 * @param boxSize The size of the system box.
			 */
			void setY(double val, double boxSize);
			/**
			 * @brief Set the z position of the particle. Handles PBC and updates z0.
			 * @param val The new z position. 
			 * @param boxSize The size of the system box.
			 */
			void setZ(double val, double boxSize);
			/**
			 * @brief Set the position of the particle. Handles PBC and updates the previous value.
			 * @param xVal,yVal,zVal The new position. 
			 * @param boxSize The size of the system box.
			 */
			void setPos(double xVal, double yVal, double zVal, double boxSize);

			//Setters for velocity

			/**
			 * @brief Set the x velocity.
			 * @param val The velocity to set.
			 */
			void setVX(double val) { vx = val; }
			/**
			 * @brief Set the y velocity.
			 * @param val The velocity to set.
			 */
			void setVY(double val) { vy = val; }
			/**
			 * @brief Set the z velocity.
			 * @param val The velocity to set.
			 */
			void setVZ(double val) { vz = val; }

			//Setters for current force.

			/**
			 * @brief Adds the the current value of force.
			 * @param xVal,yVal,zVal The values of force to add.
			 */
			void updateForce(double xVal, double yVal, double zVal);

			/**
			 * @brief Clears the current force and updates previous force.
			 */
			void clearForce();


			//Setter for containing cell.

			/**
			 * @brief Set the location of the containing cell.
			 * @param x,y,z The position variables of the containing cell.
			 */
			void setCell(int x, int y, int z) { cx = x; cy = y; cz = z; }

			//Setters for particle properties

			/**
			 * @brief Sets the radius of the particle.
			 * @param val Radius value.
			 */
			void setRadius(double val) { r = val; }

			/**
			 * @brief Sets the mass of the particle.
			 * @param val Mass value.
			 */
			void setMass(double val) { m = val; }

			/**
			 * @brief Increase coordination number.
			 */
			void incCoorNumber() { coorNumber++; }
			/**
			 * @brief Research coordination number.
			 */
			void resetCoorNumber() { coorNumber = 0; }
			/**
			 * @brief Add to the average potential.
			 * @param val
			 */
			void setEAP(double val) { eap = val; }


			/********************************************//**
			*------------------SYSTEM OUTPUT-----------------
			************************************************/

			/**
			 * @brief Writes the position of the particle to the console.
			 */
			void writePosition() { std::cout << x << ", " << y << ", " << z << "\n"; }

	};

}

#endif // PARTICLE_H