#ifndef INTEGRATOR_H
#define INTEGRATOR_H
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
			virtual int nextSystem(double time, double dt, int nParticles, int boxSize, simulation::particle** items, simulation::cell **** cells, physics::forces* f)=0;

	};

	class brownianIntegrator : public I_integrator
	{

	private:

			//System variables
			double mass;
			double temp;
			int memSize;

			//Variables vital to the force.
			double gamma;
			double dt;
			double y;

			//Variables for integrator.
			//See GUNSTEREN AND BERENDSEN 1981 EQ 2.6
			double coEff0;
			double coEff1;
			double coEff2;
			double coEff3;

			//The previous kick.
			double * memX;
			double * memY;
			double * memZ;

			//The correlation to the previous kick.
			double * memCorrX;
			double * memCorrY;
			double * memCorrZ;

			//Gaussian width.
			double sig1;
			double sig2;
			double corr;
			double dev;

			//Random gaussian generator for the random kicks.
			std::mt19937* gen;
			std::normal_distribution<double>* Dist;

			/**
			 * @brief Gets the width of the random gaussians according to G+B 2.12
			 * @param gdt gamma * dT
			 * @return 
			 */
			double getWidth(double gdt);

		public:

			/**
			 * @brief Constructs the brownian motion integrator.
			 * @param dragCoeff The drag coefficent of the system.
			 * @return Nothing
			 */
			brownianIntegrator(int nPart, double tempInit, double m, double dragCoeff, double dTime);
			/**
			 * @brief Deconstructs the integrator.
			 * @return Nothing.
			 */
			~brownianIntegrator();

			/**
			 * @brief Integrates to the next system state.
			 * @param time The current system time.
			 * @param dt The amount of time to advance.
			 * @param items The particles in the the system.
			 * @param f The force acting on the system.
			 * @return Return 0 for no error.
			 */
			int nextSystem(double time, double dt, int nParticles, int boxSize, simulation::particle** items, simulation::cell **** cells, physics::forces* f);

			/**
			 * @brief Special integration for the first time step.
			 * @param time The current system time.
			 * @param dt The amount of time to advance.
			 * @param items The particles in the the system.
			 * @param f The force acting on the system.
			 * @return Return 0 for no error.
			 */
			int firstStep(double time, double dt, int nParticles, int boxSize, simulation::particle** items, physics::forces* f);

			/**
			 * @brief Standard integration for the next time steps.
			 * @param time The current system time.
			 * @param dt The amount of time to advance.
			 * @param items The particles in the the system.
			 * @param f The force acting on the system.
			 * @return Return 0 for no error.
			 */
			int normalStep(double time, double dt, int nParticles, int boxSize, simulation::particle** items, physics::forces* f);

	};

}

#endif // VERLET_H
