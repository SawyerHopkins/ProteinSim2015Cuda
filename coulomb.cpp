#include "force.h"

namespace physics
{
	//Get the acceleration from the Coloumb potential.
	void electroStaticForce::getAcceleration(float index, float time, mathTools::points* pts, float (&acc)[3])
	{
		for (int i = 0; i < pts->arrSize; i++)
		{
			if (i != index)
			{
				float difX = (pts->getX(index)) - (pts->getX(i));
				float difY = (pts->getY(index)) - (pts->getY(i));
				float difZ = (pts->getZ(index)) - (pts->getZ(i));
				if (fabs(difX) > 0.1/std::sqrt(3))
				{
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