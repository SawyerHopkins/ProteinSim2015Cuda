#include "force.h"

namespace physics
{
	//Get the acceleration from the Coloumb potential.
	void dragForce::getAcceleration(int index, float time, mathTools::points* pts, float (&acc)[3])
	{
		//Creates drag in each direction with coefficent gamma.
		float dragX = -gamma*(pts->getVX(index));
		float dragY = -gamma*(pts->getVY(index));
		float dragZ = -gamma*(pts->getVZ(index));

		//Updates the particle acceleration.
		*(acc+0)=dragX;
		*(acc+1)=dragY;
		*(acc+2)=dragZ;
	}
}