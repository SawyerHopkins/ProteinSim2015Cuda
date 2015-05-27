#ifndef FORCE_H
#define FORCE_H
#include <vector>
#include <cmath>
#include "point.h"

namespace physics
{

	//A generic force container.
	class IForce
	{
		public:
			//virtual methods for forces of various parameters.
			virtual void getAcceleration(float index, float time, mathTools::points* pts, float (&acc)[3])=0;
			//Mark if the force is time dependent.
			virtual bool isTimeDependent()=0;
	};

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
			void getAcceleration(float index, float time, mathTools::points* pts, float (&acc)[3]);
			bool isTimeDependent() { return timeDependent; }
	};

	//Coloumb potential.
	class electroStaticForce : public IForce
	{
		public:

			//Constructor/Destructor
			electroStaticForce() {};
			~electroStaticForce() {};

			//Evaluates the force.
			void getAcceleration(float index, float time, mathTools::points* pts, float (&acc)[3]);
			bool isTimeDependent() { return false; }
	};

	//Coloumb potential.
	class dragForce : public IForce
	{
		public:

			//Constructor/Destructor
			dragForce() {};
			~dragForce() {};

			//Evaluates the force.
			void getAcceleration(float index, float time, mathTools::points* pts, float (&acc)[3]);
			bool isTimeDependent() { return false; }
	};

}

#endif // FORCE_H
