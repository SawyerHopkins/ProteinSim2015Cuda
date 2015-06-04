#ifndef INTEGRATOR_H
#define INTEGRATOR_H
#include <iostream>
#include "force.h"
#include "point.h"

using namespace mathTools;

namespace integrators
{

/*-----------------------------------------*/
/*----------INTEGRATOR INTERFACE-----------*/
/*-----------------------------------------*/

	//Creates an abstract parent class (interface) for a generic integrator.
	class I_integrator
	{
		protected:
			//Delta t value used.
			double timeStep;
			//Keeps track of total time.
			double systemTime;

		public:
			//Generic functions for advancing position and velocity
			virtual int nextSystem(double index, mathTools::points* pnt, physics::forces* f)=0;

			//getter/setter for 'timeStep'
			void setTimeStep(double newStep) { timeStep = newStep; }
			double getTimeStep() { return timeStep; }

			//getter for 'systemTime'
			double getSystemTime() { return systemTime; }
			//output current system time.
			void writeSystemTime() { std::cout << "Current time: " << systemTime << ".\n"; }
			//increases system time by one timestep.
			void advanceTime() { systemTime = systemTime + timeStep; }
	};

/*-----------------------------------------*/
/*------------VERLET INTEGRATOR------------*/
/*-----------------------------------------*/

	//Verlet Velocity integrator.
	class verlet: public I_integrator
	{
		public:

			//Constructor/Destructor
			verlet(double dt);
			~verlet();

			//Calculates the next system state.
			int nextSystem(double index, mathTools::points* pnt, physics::forces* f);

			//Gets the next position and velocity terms.
			int nextPosition(double index, mathTools::points* pnt, physics::forces* f);
			int nextVelocity(double index, mathTools::points* pnt, physics::forces* f);

			//The velocity verlet Algorithms to obtain the next position and velocity terms.
			double posAlgorithm(double pos, double vel, double acc, double t);
			double velAlgorithm(double pos, double vel, double acc, double accnext, double t);
	};

}

#endif // VERLET_H
