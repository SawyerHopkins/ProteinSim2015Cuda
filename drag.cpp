#include "force.h"

namespace physics
{
	//Get the acceleration from the Coloumb potential.
	void dragForce::getAcceleration(float index, float time, mathTools::points* pts, float (&acc)[3])
	{
		//Creates drag in each direction with coefficent 1/1.5
		float dragX = -(pts->getVX(index))/1.5;
		float dragY = -(pts->getVY(index))/1.5;
		float dragZ = -(pts->getVZ(index))/1.5;

		//Updates the particle acceleration.

		*(acc+0)=dragX;
		*(acc+1)=dragY;
		*(acc+2)=dragZ;
	}
}