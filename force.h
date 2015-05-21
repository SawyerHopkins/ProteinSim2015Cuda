#ifndef FORCE_H
#define FORCE_H

namespace physics
{

class IForce
{
	public:

		//virtual methods for forces of various parameters.
		virtual float getAcceleration(float pos, float vel, float time)=0;
};


class electroStaticForce : public IForce
{
	public:

		//Constructor/Destructor
		electroStaticForce();
		~electroStaticForce();

		//Evaluates the force.
		float getAcceleration();
};

}

#endif // FORCE_H
