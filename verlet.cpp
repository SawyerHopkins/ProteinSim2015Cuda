#include "integrator.h"

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

	//Calculates the next position and velocity
	int verlet::nextSystem(float index, mathTools::points* pts, physics::forces* f)
	{
		//A(t)
		float acc[3] = {0.0,0.0,0.0};
		//A(t+dt)
		float accNext[3] = {0.0,0.0,0.0};

		//Gets A(t)
		f->getAcceleration(index,systemTime,pts,acc);

		//Gets the new position.
		pts->setX(index, posAlgorithm(pts->getX(index),pts->getVX(index),acc[0],systemTime));
		pts->setY(index, posAlgorithm(pts->getY(index),pts->getVY(index),acc[1],systemTime));
		pts->setZ(index, posAlgorithm(pts->getZ(index),pts->getVZ(index),acc[2],systemTime));

		//Gets A(t+dt)
		f->getAcceleration(index,systemTime,pts,accNext);

		//Sets the new velocity.
		pts->setVX(index, velAlgorithm(pts->getX(index),pts->getVX(index),acc[0],accNext[0],systemTime));
		pts->setVY(index, velAlgorithm(pts->getY(index),pts->getVY(index),acc[1],accNext[1],systemTime));
		pts->setVZ(index, velAlgorithm(pts->getZ(index),pts->getVZ(index),acc[2],accNext[2],systemTime));

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

