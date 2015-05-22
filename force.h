#ifndef FORCE_H
#define FORCE_H
#include <vector>

namespace physics
{

class IForce
{
	public:

		//virtual methods for forces of various parameters.
		virtual float getAcceleration(float pos, float vel, float time)=0;
};

class forces
{
	private:

		std::vector<IForce*> flist;

	public:
		//Adds a force to the  stack
		void addForce(IForce* f) { flist.push_back(f); }

		//Calculates the total acceleration
		void getAcceleration(float pos[], float vel[], float t, float (&acc)[3]);
};

class electroStaticForce : public IForce
{
	public:

		//Constructor/Destructor
		electroStaticForce() {};
		~electroStaticForce() {};

		//Evaluates the force.
		float getAcceleration(float pos, float vel, float time);
};

}

#endif // FORCE_H
