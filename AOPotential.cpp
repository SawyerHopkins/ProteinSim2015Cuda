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

	void AOPotential::iterCells(int boxSize, double time, simulation::particle* index, simulation::cell* itemCell)
	{
		for(std::map<int,simulation::particle*>::iterator it=itemCell->getBegin(); it != itemCell->getEnd(); it++)
		{
			if (it->second->getName() != index->getName())
			{
				double difX = utilities::util::pbcDist(index->getX(), it->second->getX(), boxSize);
				double difY = utilities::util::pbcDist(index->getY(), it->second->getY(), boxSize);
				double difZ = utilities::util::pbcDist(index->getZ(), it->second->getZ(), boxSize);

				double distSquared = (difX*difX)+(difY*difY)+(difZ*difZ);

				if (distSquared <= cutOff)
				{
					//Gets the distance between the particles.
					double dist = std::sqrt(distSquared);

					//Throw warning if particles are acting badly.
					if (dist < 0.5*cutOff)
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

					index->updateForce(fx,fy,fz);
				}
			}
		}
	}

	//Get the acceleration from the Coloumb potential.
	void AOPotential::getAcceleration(int index, int nPart, int boxSize, double time, simulation::cell* itemCell ,simulation::particle** items)
	{

		iterCells(boxSize,time,items[index],itemCell);
		iterCells(boxSize,time,items[index],itemCell->left);
		iterCells(boxSize,time,items[index],itemCell->right);
		iterCells(boxSize,time,items[index],itemCell->top);
		iterCells(boxSize,time,items[index],itemCell->bot);
		iterCells(boxSize,time,items[index],itemCell->front);
		iterCells(boxSize,time,items[index],itemCell->back);

	}

}