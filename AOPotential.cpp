/*The MIT License (MIT)

Copyright (c) <2015> <Sawyer Hopkins>

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.*/

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
		//Sets the name
		name = "AOPotential";

		//Set vital variables.
		gamma = coeff; 
		cutOff = cut;
		dt = dTime;

		//Create secondary variables.
		a1=-gamma*(cutOff/(cutOff-1.0))*(cutOff/(cutOff-1.0))*(cutOff/(cutOff-1.0));
		a2=-3.0/(2.0*cutOff);
		a3=1.0/(2.0*cutOff*cutOff*cutOff);

		coEff1 = -a1*a2;
		coEff2 = -3.0*a1*a3;

		std::cout << "---AO Potential successfully added.\n\n";

	}

	double AOPotential::iterCells(int boxSize, double time, simulation::particle* index, simulation::cell* itemCell)
	{
		double eao = 0;

		for(std::map<int,simulation::particle*>::iterator it=itemCell->getBegin(); it != itemCell->getEnd(); ++it)
		{
			if (it->second->getName() != index->getName())
			{
				//Distance between the two particles.
				double rSquared = utilities::util::pbcDist(index->getX(), index->getY(), index->getZ(), 
																	it->second->getX(), it->second->getY(), it->second->getZ(),
																	boxSize);

				//If the particles are in the potential well.
				if (rSquared <= cutOff)
				{
					index->incCoorNumber();
					double r = sqrt(rSquared);

					//If the particles overlap there are problems.
					double size = (index->getRadius() + it->second->getRadius());
					if(r< (0.8*size) )
					{
						debugging::error::throwParticleOverlapError(index->getName(), it->second->getName(), r);
					}

					//Math
					double rInv=1.0/r; 
					double r_36=pow(rInv,36);
					double r_38=r_36/rSquared;
					double fNet=36.0*r_38+coEff1*rInv+coEff2*r; 
					fNet=-fNet;

					//Update net potential.
					eao += r_36+a1*(1.0+a2*r+a3*r*rSquared);

					//Normalize the force.
					double unitVec[3] {0.0,0.0,0.0};
					utilities::util::unitVectorAdv(index->getX(), index->getY(), index->getZ(), 
														it->second->getX(), it->second->getY(), it->second->getZ(),
														unitVec, r, boxSize);

					//Updates the acceleration.;
					double fx = fNet*unitVec[0];
					double fy = fNet*unitVec[1];
					double fz = fNet*unitVec[2];

					//If the force is infinite then there are worse problems.
					if (isnan(fNet))
					{
						debugging::error::throwInfiniteForce();
					}

					//Add to the net force on the particle.
					index->updateForce(fx,fy,fz);
				}
			}
		}
		return eao;
	}

	double AOPotential::getAcceleration(int index, int nPart, int boxSize, double time, simulation::cell* itemCell ,simulation::particle** items)
	{

		double eao = 0;
		itemCell->getMember(index)->resetCoorNumber();

		//The cell itself.
		eao += iterCells(boxSize,time,items[index],itemCell);

		//Cross section at cell.
		eao += iterCells(boxSize,time,items[index],itemCell->left);
		eao += iterCells(boxSize,time,items[index],itemCell->right);
		eao += iterCells(boxSize,time,items[index],itemCell->top);
		eao += iterCells(boxSize,time,items[index],itemCell->bot);
		eao += iterCells(boxSize,time,items[index],itemCell->top->left);
		eao += iterCells(boxSize,time,items[index],itemCell->top->right);
		eao += iterCells(boxSize,time,items[index],itemCell->bot->left);
		eao += iterCells(boxSize,time,items[index],itemCell->bot->right);

		//Cross section in front of the cell.
		eao += iterCells(boxSize,time,items[index],itemCell->front);
		eao += iterCells(boxSize,time,items[index],itemCell->front->left);
		eao += iterCells(boxSize,time,items[index],itemCell->front->right);
		eao += iterCells(boxSize,time,items[index],itemCell->front->top);
		eao += iterCells(boxSize,time,items[index],itemCell->front->bot);
		eao += iterCells(boxSize,time,items[index],itemCell->front->top->left);
		eao += iterCells(boxSize,time,items[index],itemCell->front->top->right);
		eao += iterCells(boxSize,time,items[index],itemCell->front->bot->left);
		eao += iterCells(boxSize,time,items[index],itemCell->front->bot->right);

		//Cross section behind the cell.
		eao += iterCells(boxSize,time,items[index],itemCell->back);
		eao += iterCells(boxSize,time,items[index],itemCell->back->left);
		eao += iterCells(boxSize,time,items[index],itemCell->back->right);
		eao += iterCells(boxSize,time,items[index],itemCell->back->top);
		eao += iterCells(boxSize,time,items[index],itemCell->back->bot);
		eao += iterCells(boxSize,time,items[index],itemCell->back->top->left);
		eao += iterCells(boxSize,time,items[index],itemCell->back->top->right);
		eao += iterCells(boxSize,time,items[index],itemCell->back->bot->left);
		eao += iterCells(boxSize,time,items[index],itemCell->back->bot->right);

		itemCell->getMember(index)->setEAP(eao);

		return eao;

	}

}