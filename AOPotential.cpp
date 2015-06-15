#include "force.h"

namespace physics
{

	//Releases the memory blocks
	AOPotential::~AOPotential()
	{
		delete &gamma;
		delete &cutOff;
		delete &coEff1;
		delete &coEff2;
	}

	//Create the aggregation force.
	AOPotential::AOPotential(double coeff, double cut)
	{
		//Set vital variables.
		gamma = coeff; 
		cutOff = cut;

		//Create secondary variables.
		double val1 = -gamma*pow((cutOff/(cutOff-1.0)),3.0);
		double val2 = -3.0/(2.0*cutOff);
		double val3 = 1.0/(2.0*pow(cutOff,3.0));
		coEff1 = -val1*val2;
		coEff2 = -3.0*val1*val3;

		std::cout << "---AO Potential successfully added.\n\n";

	}

	//Get the acceleration from the Coloumb potential.
	void AOPotential::getAcceleration(int index, int nPart, int boxSize, int cellScale, double time, simulation::particle** items)
	{

		//Iterate across all particles.
		for (int i = 0; i < nPart; i++)
		{
			//Excluse self interation.
			if ( (i != index) && (isInRange(index,i,cellScale,items)) )
			{

				//Get the distance between the two particles.
				double difX = utilities::util::pbcDist(items[index]->getX(), items[i]->getX(), boxSize);
				double difY = utilities::util::pbcDist(items[index]->getY(), items[i]->getY(), boxSize);
				double difZ = utilities::util::pbcDist(items[index]->getZ(), items[i]->getZ(), boxSize);

				//Gets the square distance between the two particles.
				double distSquared = (difX*difX)+(difY*difY)+(difZ*difZ);

				if (distSquared <= cutOff)
				{
					//Gets the distance between the particles.
					double dist = std::sqrt(distSquared);

					//Throw warning if particles are acting badly.
					if (dist < 0.75*cutOff)
					{
						std::cout << "\nSignificant particle overlap. Consider time-step reduction.\n";
						exit(100);
					}

					//Builds the force.
					double distInverse = 1.0/dist;
					double dist_36 = pow(distInverse,36);
					double dist_norm = dist_36/distSquared;
					double forceMag = -((36.0*dist_norm) + (coEff1*distInverse) + (coEff2*dist));

					//Projects the force onto the unit vectors.
					double unitVec[3] = {0.0,0.0,0.0};
					utilities::util::unitVector(difX, difY, difZ, dist, unitVec);

					//Updates the acceleration. (Mass === 1);
					double fx = forceMag*unitVec[0];
					double fy = forceMag*unitVec[1];
					double fz = forceMag*unitVec[2];

					items[index]->updateForce(fx,fy,fz);

				}

			}
		}

	}

	bool AOPotential::isInRange(int index, int j, int cellScale, simulation::particle** items)
	{
		bool inRange = true;

		int dX = items[index]->getCX() - items[j]->getCX();
		int dY = items[index]->getCY() - items[j]->getCY();
		int dZ = items[index]->getCZ() - items[j]->getCZ();

		if ( (dX != 1) && (dX != 0) && (dX != cellScale) )
		{
			inRange = false;
		}
		if ( (dY != 1) && (dY != 0) && (dY != cellScale) )
		{
			inRange = false;
		}
			if ( (dZ != 1) && (dZ != 0) && (dZ != cellScale) )
		{
			inRange = false;
		}

		return inRange;

	}

}