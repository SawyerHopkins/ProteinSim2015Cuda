#include "force.h"

namespace physics
{
	//Get the acceleration from the Coloumb potential.
	void electroStaticForce::getAcceleration(float index, float time, mathTools::points* pts, float (&acc)[3])
	{
		//Iterate across all particles.
		for (int i = 0; i < pts->arrSize; i++)
		{
			//Excluse self interation.
			if (i != index)
			{
				//Get the distance between the two particles.
				float difX = (pts->getX(index)) - (pts->getX(i));
				float difY = (pts->getY(index)) - (pts->getY(i));
				float difZ = (pts->getZ(index)) - (pts->getZ(i));

				//Check that the particles are not on top of eachother.
				if (fabs(difX) > 0.1/std::sqrt(3))
				{
					//Create equal and opposite forces.
					if (difX < 0)
					{
						*(acc+0) += .1/(difX*difX);
					}
					else
					{
						*(acc+0) += -.1/(difX*difX);
					}
				}
				if (fabs(difY) > 0.1/std::sqrt(3))
				{
					//Create equal and opposite forces.
					if (difY < 0)
					{
						*(acc+1) += .1/(difY*difY);
					}
					else
					{
						*(acc+1) += -.1/(difY*difY);
					}
				}
				if (fabs(difZ) > 0.1/std::sqrt(3))
				{
					//Create equal and opposite forces.
					if (difZ < 0)
					{
						*(acc+2) += .1/(difZ*difZ);
					}
					else
					{
						*(acc+2) += -.1/(difZ*difZ);
					}
				}
			}
		}
	}
}