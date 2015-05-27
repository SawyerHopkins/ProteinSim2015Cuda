#ifndef INTEGRATOR_H
#define INTEGRATOR_H
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
			//Keeps track of total time.
			float systemTime;

		public:
			//Generic functions for advancing position and velocity
			virtual int nextSystem(float index, mathTools::points* pnt, physics::forces* f)=0;

			//getter/setter for 'timeStep'
			void setTimeStep(float newStep) { timeStep = newStep; }
			float getTimeStep() { return timeStep; }

			//getter for 'systemTime'
			float getSystemTime() { return systemTime; }
			void writeSystemTime() { std::cout << "Current time: " << systemTime << ".\n"; }
			void advanceTime() { systemTime = systemTime + timeStep; }
	};

	//Verlet Velocity integrator.
	class verlet: public I_integrator
	{
		public:

			//Constructor/Destructor
			verlet(float dt);
			~verlet();

			//Gets the next position and velocity terms.
			int nextPosition(float index, mathTools::points* pnt, physics::forces* f);
			int nextVelocity(float index, mathTools::points* pnt, physics::forces* f);
			int nextSystem(float index, mathTools::points* pnt, physics::forces* f);

			//The velocity verlet Algorithms to obtain the next position and velocity terms.
			float posAlgorithm(float pos, float vel, float acc, float t);
			float velAlgorithm(float pos, float vel, float acc, float accnext, float t);
	};

}

#endif // VERLET_H
