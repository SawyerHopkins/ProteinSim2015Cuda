#include "force.h"

namespace physics
{
	//Get the acceleration from the Coloumb potential.
	void dragForce::getAcceleration(int index, double time, mathTools::points* pts, double (&acc)[3])
	{
		//Creates drag in each direction with coefficent gamma.
		double dragX = -gamma*(pts->getVX(index));
		double dragY = -gamma*(pts->getVY(index));
		double dragZ = -gamma*(pts->getVZ(index));

		//Updates the particle acceleration.
		*(acc+0)=dragX;
		*(acc+1)=dragY;
		*(acc+2)=dragZ;
	}
}