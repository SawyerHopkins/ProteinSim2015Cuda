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

	AOPotential::AOPotential(configReader::config* cfg)
	{
		//Sets the name
		name = "AOPotential";

		//Set vital variables.

		//Sets the system drag.
		std::string keyName = "gamma";
		if (cfg->containsKey(keyName))
		{
			gamma = cfg->getParam<double>(keyName);
		}
		else
		{
			std::cout << "-Option: '" << keyName << "' missing\n";
			std::cout << "-Using default.\n\n";
			gamma = 0.5;
		}
		std::cout << "---" << "temp: " << gamma << "\n";

		keyName = "timeStep";
		if (cfg->containsKey(keyName))
		{
			dt = cfg->getParam<double>(keyName);
		}
		else
		{
			std::cout << "-Option: '" << keyName << "' missing\n";
			std::cout << "-Using default.\n\n";
			dt = 0.001;
		}
		std::cout << "---" << keyName << ": " << dt << "\n";

		keyName = "cutOff";
		if (cfg->containsKey(keyName))
		{
			cutOff = cfg->getParam<double>(keyName);
		}
		else
		{
			std::cout << "-Option: '" << keyName << "' missing\n";
			std::cout << "-Using default.\n\n";
			cutOff = 1.1;
		}
		std::cout << "---" << keyName << ": " << cutOff << "\n";

		//Create secondary variables.
		a1=-gamma*(cutOff/(cutOff-1.0))*(cutOff/(cutOff-1.0))*(cutOff/(cutOff-1.0));
		a2=-3.0/(2.0*cutOff);
		a3=1.0/(2.0*cutOff*cutOff*cutOff);

		coEff1 = -a1*a2;
		coEff2 = -3.0*a1*a3;

		std::cout << "---AO Potential successfully added.\n\n";

	}

	void AOPotential::iterCells(int boxSize, double time, simulation::particle* index, simulation::cell* itemCell)
	{
		double pot = 0;

		for(std::map<int,simulation::particle*>::iterator it=itemCell->getBegin(); it != itemCell->getEnd(); ++it)
		{
			if (it->second->getName() != index->getName())
			{
				//Distance between the two particles.
				double rSquared = utilities::util::pbcDist(index->getX(), index->getY(), index->getZ(), 
																	it->second->getX(), it->second->getY(), it->second->getZ(),
																	boxSize);

				double rCutSquared = cutOff*cutOff;

				//If the particles are in the potential well.
				if (rSquared <= rCutSquared)
				{
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
					pot += r_36+a1*(1.0+a2*r+a3*r*rSquared);

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
					index->updateForce(fx,fy,fz,pot,it->second);
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