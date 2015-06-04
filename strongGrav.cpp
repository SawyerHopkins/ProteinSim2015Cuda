#include "force.h"

namespace physics
{
	//Get the acceleration from the gravitational potential.
	void strongGravForce::getAcceleration(int index, float time, mathTools::points* pts, float (&acc)[3])
	{
		//Iterate across all particles.
		for (int i = 0; i < pts->arrSize; i++)
		{
			//Excluse self interation.
			if (i != index)
			{
				//Get the distance between the two particles.
				float difX = mathTools::utilities::pbcDist(pts->getX(index),pts->getX(i),pts->getBoxSize());
				float difY = mathTools::utilities::pbcDist(pts->getY(index),pts->getY(i),pts->getBoxSize());
				float difZ = mathTools::utilities::pbcDist(pts->getZ(index),pts->getZ(i),pts->getBoxSize());

				float mag = strength/(difx*difX+difY*difY+difZ*difZ);

				//Check that the particles are not on top of eachother.
				if (fabs(difX) > 0.1/std::sqrt(3))
				{
					//Create equal and opposite forces.
					if (difX < 0)
					{
						*(acc+0) += strength/(difX*difX);
					}
					else
					{
						*(acc+0) += -strength/(difX*difX);
					}
				}
				if (fabs(difY) > 0.1/std::sqrt(3))
				{
					//Create equal and opposite forces.
					if (difY < 0)
					{
						*(acc+1) += strength/(difY*difY);
					}
					else
					{
						*(acc+1) += -strength/(difY*difY);
					}
				}
				if (fabs(difZ) > 0.1/std::sqrt(3))
				{
					//Create equal and opposite forces.
					if (difZ < 0)
					{
						*(acc+2) += strength/(difZ*difZ);
					}
					else
					{
						*(acc+2) += -strength/(difZ*difZ);
					}
				}
			}
		}
	}
}