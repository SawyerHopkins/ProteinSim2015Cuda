#ifndef FORCE_H
#define FORCE_H
#include <vector>

namespace physics
{

	//A generic force container.
	class IForce
	{
		public:
			//virtual methods for forces of various parameters.
			virtual float getAcceleration(float pos, float vel, float time)=0;
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
			void addForce(IForce* f) { flist.push_back(f); }

			//Calculates the total acceleration
			void getAcceleration(float pos[], float vel[], float t, float (&acc)[3]);
	};

	//Coloumb potential.
	class electroStaticForce : public IForce
	{
		public:

			//Constructor/Destructor
			electroStaticForce() {};
			~electroStaticForce() {};

			//Evaluates the force.
			float getAcceleration(float pos, float vel, float time);
			bool isTimeDependent() { return false; }
	};

}

#endif // FORCE_H
