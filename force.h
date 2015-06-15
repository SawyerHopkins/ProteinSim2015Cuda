#ifndef FORCE_H
#define FORCE_H
#include "particle.h"

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
			virtual void getAcceleration(int index, int nPart, int boxSize, int cellScale, double time, simulation::particle** items)=0;
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
			void getAcceleration(int nPart, int boxSize, int cellScale, double time, simulation::particle** items);
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
			void getAcceleration(int index, int nPart, int boxSize, int cellScale, double time, simulation::particle** items);
			bool isTimeDependent() { return false; }

			bool isInRange(int index, int j, int cellScale, simulation::particle** items);

	};

}

#endif // FORCE_H
