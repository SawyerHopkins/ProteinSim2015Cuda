#ifndef VERLET_H
#define VERLET_H
#include <iostream>
#include "force.h"
#include "point.h"

using namespace mathTools;

namespace integrators
{

//Creates an abstract parent class (interface) for a generic integrator.
class I_integrator
{
	protected:
		//Delta t value used.
		float timeStep;
		float systemTime;

	public:
		virtual float nextPosition(float index, mathTools::points* pnt, physics::forces* f) =0;
		virtual float nextVelocity(float index, mathTools::points* pnt, physics::forces* f) =0;

		//getter/setter for 'timeStep'
		void setTimeStep(float newStep) { timeStep = newStep; }
		float getTimeStep() { return timeStep; }

		//getter for 'systemTime'
		float getSystemTime() { return systemTime; }
		void writeSystemTime() { std::cout << "Current time: " << systemTime << ".\n"; }
};

//Verlet Velocity integrator.
class verlet: public I_integrator
{
	public:

		//Constructor/Destructor
		verlet(float dt);
		~verlet();

		//Gets the next position and velocity terms.
		float nextPosition(float index, mathTools::points* pnt, physics::forces* f);
		float nextVelocity(float index, mathTools::points* pnt, physics::forces* f);
};

}

#endif // VERLET_H
