#include "force.h"

namespace physics
{
	//Get the acceleration from the Coloumb potential.
	void dragForce::getAcceleration(float index, float time, mathTools::points* pts, float (&acc)[3])
	{
		float dragX = -(pts->getVX(index))/1.5;
		float dragY = -(pts->getVY(index))/1.5;
		float dragZ = -(pts->getVZ(index))/1.5;

		*(acc+0)=dragX;
		*(acc+1)=dragY;
		*(acc+2)=dragZ;
	}
}