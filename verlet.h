#ifndef VERLET_H
#define VERLET_H
#include <iostream>
#include "force.h"

namespace integrators
{

//Creates an abstract parent class (interface) for a generic integrator.
class I_integrator
{
	public:

		virtual float nextPosition(float pos, float vel, physics::IForce* f) =0;
		virtual float nextVelocity(float pos, float vel, physics::IForce* f) =0;
};

//Verlet Velocity integrator.
class verlet: public I_integrator
{
	private:
		//Delta t value used.
		float timeStep;
		float systemTime;

	public:

		//Constructor/Destructor
		verlet(float dt);
		~verlet();

		//Gets the next position and velocity terms.
		float nextPosition(float pos, float vel, physics::IForce* f);
		float nextVelocity(float pos, float vel, physics::IForce* f);

		//getter/setter for 'timeStep'
		void setTimeStep(float newStep) { timeStep = newStep; }
		float getTimeStep() { return timeStep; }
		
		//getter for 'systemTime'
		float getSystemTime() { return systemTime; }
		void writeSystemTime() { std::cout << "Current time: " << systemTime << ".\n"; }
};

}

#endif // VERLET_H
