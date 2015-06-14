#ifndef INTEGRATOR_H
#define INTEGRATOR_H
#include <iostream>
#include "force.h"

namespace integrators
{

/*-----------------------------------------*/
/*----------INTEGRATOR INTERFACE-----------*/
/*-----------------------------------------*/

	//Creates an abstract parent class (interface) for a generic integrator.
	class I_integrator
	{
		public:
			/**
			 * @brief Integrates to the next system state.
			 * @param time The current system time.
			 * @param dt The amount of time to advance.
			 * @param items The particles in the the system.
			 * @param f The force acting on the system.
			 * @return Return 0 for no error.
			 */
			virtual int nextSystem(float time, float dt, simulation::particle* items, physics::forces* f)=0;
	};

	class brownianIntegrator : I_integrator
	{

		private:



		public:

		brownianIntegrator() {};
		~brownianIntegrator() {};

	};

}

#endif // VERLET_H
