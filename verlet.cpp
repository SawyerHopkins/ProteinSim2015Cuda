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
	//Iterates across all spacial cordinates.
	//Directly updates pts through the pointer.
	//Use float return for debugging if needed.
	int verlet::nextPosition(float index, float pos[], float vel[], mathTools::points* pts, physics::forces* f)
	{
		float acc[3] = {0.0,0.0,0.0};
		f->getAcceleration(pos,vel,systemTime,acc);
		//Error code 0.0 -> No Error.
		return 0;
	}

	//Uses velocity verlet to get next velocity
	//Iterates across all velocity cordinates.
	//Directly updates pts through the pointer.
	//Use float return for debugging if needed.
	int verlet::nextVelocity(float index, float pos[], float vel[], mathTools::points* pts, physics::forces* f)
	{
		float acc[3] = {0.0,0.0,0.0};
		f->getAcceleration(pos,vel,systemTime,acc);
		//Error code 0.0 -> No Error.
		return 0;
	}

	//The velocity verlet Algorithms to obtain the next position term.
	//Performs operation on one spacial cordinate.
	float verlet::posAlgorithm(float pos, float vel, float acc, float t)
	{
		return pos+(vel*timeStep)+0.5*timeStep*timeStep*acc;
	}

	//The velocity verlet Algorithms to obtain the next velocity term term.
	//Performs operation on one velocity cordinate.
	float verlet::velAlgorithm(float pos, float vel, float acc, float accnext, float t)
	{
		return vel+0.5*timeStep*(acc+accnext);
	}
}

