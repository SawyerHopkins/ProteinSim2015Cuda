#include "force.h"

namespace physics
{

	AOPotential::~AOPotential()
	{
		delete &gamma;
		delete &cutOff;
		delete &coEff1;
		delete &coEff2;
	}

	AOPotential::AOPotential(double coeff, double cut, double dTime)
	{
		//Set vital variables.
		gamma = coeff; 
		cutOff = cut;

		//Create secondary variables.
		double a1=-gamma*(cutOff/(cutOff-1.0))*(cutOff/(cutOff-1.0))*(cutOff/(cutOff-1.0));
		double a2=-3.0/(2.0*cutOff);
		double	a3=1.0/(2.0*cutOff*cutOff*cutOff);
		coEff1 = -a1*a2;
		coEff2 = -3.0*a1*a3;

		std::cout << "---AO Potential successfully added.\n\n";

	}

	void AOPotential::iterCells(int boxSize, double time, simulation::particle* index, simulation::cell* itemCell)
	{
		for(std::map<int,simulation::particle*>::iterator it=itemCell->getBegin(); it != itemCell->getEnd(); it++)
		{
			if (it->second->getName() != index->getName())
			{
				double rSquared = utilities::util::pbcDistAlt(index->getX(), index->getY(), index->getZ(), 
																	it->second->getX(), it->second->getY(), it->second->getZ(),
																	boxSize);

				if (rSquared <= cutOff)
				{
					double r = sqrt(rSquared);
					if(r<(0.8))
					{
						std::cout << "\n" << rSquared;
						std::cout << "\n" << it->second->getName() << " : " << it->second->getX() << "," << it->second->getY() << "," << it->second->getZ();
						std::cout << "\n" << index->getName() << " : " << index->getX() << "," << index->getY() << "," << index->getZ();
						std::cout << "\nSignificant particle overlap. Consider time-step reduction.\n";
						exit(100);
					}
					double rInv=1.0/r; 
					double r_36=pow(rInv,36);
					double r_38=r_36/rSquared;

					double fNet=36.0*r_38+coEff1*rInv+coEff2*r; 
					fNet=-fNet;

					double unitVec[3] {0.0,0.0,0.0};
					utilities::util::unitVectorAlt(index->getX(), index->getY(), index->getZ(), 
														it->second->getX(), it->second->getY(), it->second->getZ(),
														unitVec, r, boxSize);

					//Updates the acceleration.;
					double fx = fNet*unitVec[0];
					double fy = fNet*unitVec[1];
					double fz = fNet*unitVec[2];

					if (isnan(fNet))
					{
						std::cout << "\nBad news bears.";
					}

					index->updateForce(fx,fy,fz);
				}
			}
		}
	}

	void AOPotential::getAcceleration(int index, int nPart, int boxSize, double time, simulation::cell* itemCell ,simulation::particle** items)
	{

		//The cell itself.
		iterCells(boxSize,time,items[index],itemCell);

		//Cross section at cell.
		iterCells(boxSize,time,items[index],itemCell->left);
		iterCells(boxSize,time,items[index],itemCell->right);
		iterCells(boxSize,time,items[index],itemCell->top);
		iterCells(boxSize,time,items[index],itemCell->bot);
		iterCells(boxSize,time,items[index],itemCell->top->left);
		iterCells(boxSize,time,items[index],itemCell->top->right);
		iterCells(boxSize,time,items[index],itemCell->bot->left);
		iterCells(boxSize,time,items[index],itemCell->bot->right);

		//Cross section in front of the cell.
		iterCells(boxSize,time,items[index],itemCell->front);
		iterCells(boxSize,time,items[index],itemCell->front->left);
		iterCells(boxSize,time,items[index],itemCell->front->right);
		iterCells(boxSize,time,items[index],itemCell->front->top);
		iterCells(boxSize,time,items[index],itemCell->front->bot);
		iterCells(boxSize,time,items[index],itemCell->front->top->left);
		iterCells(boxSize,time,items[index],itemCell->front->top->right);
		iterCells(boxSize,time,items[index],itemCell->front->bot->left);
		iterCells(boxSize,time,items[index],itemCell->front->bot->right);

		//Cross section behind the cell.
		iterCells(boxSize,time,items[index],itemCell->back);
		iterCells(boxSize,time,items[index],itemCell->back->left);
		iterCells(boxSize,time,items[index],itemCell->back->right);
		iterCells(boxSize,time,items[index],itemCell->back->top);
		iterCells(boxSize,time,items[index],itemCell->back->bot);
		iterCells(boxSize,time,items[index],itemCell->back->top->left);
		iterCells(boxSize,time,items[index],itemCell->back->top->right);
		iterCells(boxSize,time,items[index],itemCell->back->bot->left);
		iterCells(boxSize,time,items[index],itemCell->back->bot->right);

	}

}