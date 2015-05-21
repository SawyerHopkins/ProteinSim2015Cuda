#include "verlet.h"
#include "force.h"

namespace integrators
{
	verlet::verlet(float dt)
	{
		verlet::timeStep = dt;
		verlet::systemTime = 0.0;
	}

	verlet::~verlet()
	{
		delete[] &(verlet::timeStep);
		delete[] &(verlet::systemTime);
	}

	float verlet::nextPosition(float pos, float vel, physics::IForce* f)
	{
		return pos+(vel*verlet::timeStep);
	}

	float verlet::nextVelocity(float pos, float vel, physics::IForce* f)
	{
		return 0.0;
	}
}

