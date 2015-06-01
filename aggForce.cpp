#include "force.h"

namespace physics
{

	aggForce::aggForce(float coeff, float cut)
	{
		gamma = coeff; 
		cutOff = cut;

		float val1 = -gamma*pow((cutOff/(cutOff-1.0)),3.0);
		float val2 = -3.0/(2.0*cutOff);
		float val3 = 1.0/(2.0*pow(cutOff,3.0));
		coEff1 = -val1*val2;
		coEff2 = -3.0*val1*val3;

	}

	//Get the acceleration from the Coloumb potential.
	void aggForce::getAcceleration(int index, float time, mathTools::points* pts, float (&acc)[3])
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

				float distSquared = (difX*difX)+(difY*difY)+(difZ*difZ);

				if (distSquared <= cutOff)
				{
					float dist = std::sqrt(distSquared);

					//Throw warning if particles are acting badly.
					if (dist < 1.5*pts->getR())
					{
						std::cout << "\nSignificant particle overlap. Consider time-step reduction.\n";
					}

					float distInverse = 1.0/dist;
					float dist_36 = pow(distInverse,36);
					float dist_norm = dist_36/distSquared;

					float forceMag = ((36.0*dist_norm) + (coEff1*distInverse) + (coEff2*dist));

					float unitVec[3] = {0.0,0.0,0.0};
					mathTools::utilities::unitVector(difX, difY, difZ, dist, unitVec);

					acc[0]+=forceMag*unitVec[0];
					acc[1]+=forceMag*unitVec[1];
					acc[2]+=forceMag*unitVec[2];

				}

			}
		}

	}

}