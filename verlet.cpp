#include "verlet.h"
#include "force.h"
#include "point.h"

namespace integrators
{
	//Sets time step and zeros 'systemTime'.
	verlet::verlet(float dt)
	{
		timeStep = dt;
		systemTime = 0.0;
	}

	//Releases time information from memory.
	verlet::~verlet()
	{
		delete[] &timeStep;
		delete[] &systemTime;
	}

	//Uses velocity verlet to get next position
	float verlet::nextPosition(float index, mathTools::points* pts, physics::forces* f)
	{
		return 0.0;
		//return pos+(vel*timeStep)+0.5*timeStep*timeStep*(f->getAcceleration(pos,vel,systemTime));
	}

	//Uses velocity verlet to get next velocity
	float verlet::nextVelocity(float index, mathTools::points* pts, physics::forces* f)
	{
		return 0.0;
		//return vel+0.5*timeStep*(f->getAcceleration(pos,vel,systemTime)+f->getAcceleration(pos,vel,systemTime+timeStep));
	}
}

