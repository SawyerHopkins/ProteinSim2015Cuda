#ifndef FORCE_H
#define FORCE_H
#include "cell.h"

namespace physics
{

/*-----------------------------------------*/
/*-------------FORCE INTERFACE-------------*/
/*-----------------------------------------*/

	//A generic force container.
	class IForce
	{
		public:
			//virtual methods for forces of various parameters.
			virtual void getAcceleration(int index, double time, simulation::particle** items)=0;
			//Mark if the force is time dependent.
			virtual bool isTimeDependent()=0;
	};

/*-----------------------------------------*/
/*----------INTEGRATOR MANAGEMENT----------*/
/*-----------------------------------------*/

	//Management system for a collection of forces.
	class forces
	{
		private:

			//A vector of all forces in the system.
			std::vector<IForce*> flist;
			//Flagged if flist contains a time dependant force.
			bool timeDependent;

		public:

			//Constructor/Destructor
			forces();
			~forces();

			//Adds a force to the  stack
			void addForce(IForce* f);

			//Calculates the total acceleration
			void getAcceleration(int index, double time, simulation::particle** items);
			bool isTimeDependent() { return timeDependent; }
	};

/*-----------------------------------------*/
/*---------------AO POTENTIAL--------------*/
/*-----------------------------------------*/

	//Drag force.
	class AOPotential : public IForce
	{

	private:

			//Variables vital to the force.
			double gamma;
			double cutOff;

			//Secondary variables.
			double coEff1;
			double coEff2;

		public:

			//Constructor/Destructor
			AOPotential(double coeff, double cut);
			~AOPotential();

			//Evaluates the force.
			void getAcceleration(int index, double time, simulation::particle** items);
			bool isTimeDependent() { return false; }
	};

/*-----------------------------------------*/
/*-------------BROWNIAN FORCE--------------*/
/*-----------------------------------------*/

	//Drag force.
	class brownianForce : public IForce
	{

	private:

			//Variables vital to the force.
			double gamma;
			double sigma;

			//Secondary variables.
			double sig1;
			double sig2;
			double corr;
			double rc12;
			double c0;

			//The previous kick.
			double * memX;
			double * memY;
			double * memZ;

			//The correlation to the previous kick.
			double * memCorrX;
			double * memCorrY;
			double * memCorrZ;

			//Number of particles to remember.
			int memSize;

			//Random gaussian generator for the random kicks.
			std::mt19937* gen;
			std::normal_distribution<double>* distribution;

		public:

			//Constructor/Destructor
			brownianForce(double coEff, double stDev, double t_initial, double dt, int size);
			~brownianForce();

			//Setup the secondary variables.
			void init(double dt, double t_initial);

			//Evaluates the force.
			void getAcceleration(int index, double time, simulation::particle** items);
			bool isTimeDependent() { return false; }
	};

}

#endif // FORCE_H
