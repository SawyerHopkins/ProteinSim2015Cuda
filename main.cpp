#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <string.h>
#include <float.h> 
#include <point.h>
#include "verlet.h"
#include "force.h"
#include <iostream>

using namespace std;

int main(int argc, char **argv)
{
	/*-------------Variables-------------*/

	//Initialize random number generator.
	srand (time(NULL));
	//Set the maximum time.
	float endTime = 1000;
	//Set the time step for the integrator.
	float timeStep = 1;
	//Set the number of particles.
	float nParticles = 10;

	/*-------------Setup-------------*/

	//Create the integrator.
	integrators::verlet * difeq = new integrators::verlet(timeStep);

	//Creates the particle system.
	mathTools::points * pt = new mathTools::points(nParticles, 5);
	//Initialize the particle system with random position and velocity.
	pt->init(0.01);

	//Creates a force manager.
	physics::forces * force = new physics::forces();
	force->addForce(new physics::electroStaticForce()); //Adds the Coloumb potential.

	/*-------------Debugging-------------*/
	/*-Out the position of each particle-*/
	for (int i = 0; i < pt->arrSize; i++)
	{
		pt->writePosition(i);
	}

	/*-------------Iterator-------------*/
	while(difeq->getSystemTime() < endTime)
	{
		for (int i =0; i < pt->arrSize; i++)
		{
			float pos[3] = {pt->getX(i), pt->getY(i), pt->getZ(i)};
			float vel[3] = {pt->getVX(i), pt->getVY(i), pt->getVZ(i)};
			difeq->nextPosition(i,pos,vel,pt,force);
			difeq->advanceTime();
		}
	}

	//Debug code 0 -> No Error:
	return 0;
}