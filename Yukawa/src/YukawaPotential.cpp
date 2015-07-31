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

#include "YukawaPotential.h"
using namespace simulation;

Yukawa::~Yukawa()
{
	delete[] &cutOff;
	delete[] &dampening;
	delete[] &mass;
	delete[] &wellDepth;
}

Yukawa::Yukawa(configReader::config* cfg)
{
	//Sets the name
	name = "Yukawa";

	//Get the radius
	radius = cfg->getParam<double>("radius",0.5);

	//Get the mass
	mass = cfg->getParam<double>("mass",1.0);

	//Get the well depth
	wellDepth = cfg->getParam<double>("yukawaDepth",100.0);

	//Get the cutoff range
	cutOff = cfg->getParam<double>("cutOff",1.1);

	//Find dampening so that the potential is 2 orders of magnitude smaller at cutoff than at two radii.
	double rm = 2.0*radius;
	double s = 100;
	double diff = cutOff - rm;
	dampening = (std::log(s*rm/cutOff))/(diff*mass);

	//Find gamma such that the force is the correct well depth at rm.
	double rmSquared = rm*rm;
	double exp_scale = std::exp(dampening*mass*rm);
	double line_scale = (-dampening*mass*rm)-1;
	gamma = wellDepth*exp_scale*rmSquared/line_scale;

	std::cout << "---gamma: " << gamma << "\n";
	std::cout << "---dampening: " << dampening << "\n";

	utilities::util::writeTerminal("---Yukawa Potential successfully added.\n\n", utilities::Colour::Cyan);

}

void Yukawa::iterCells(int boxSize, double time, particle* index, cell* itemCell)
{
	double pot = 0;

	for(std::map<int,particle*>::iterator it=itemCell->getBegin(); it != itemCell->getEnd(); ++it)
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
				double fNet = 0;
				double rInv=-dampening/r; 
				double rSquaredInv=-1.0/(r*r);
				double rExp=std::exp(-dampening*mass*r);

				if (r >= size)
				{
					//Negative for attractive.
					fNet = -gamma*rExp*(rInv + rSquaredInv);
				}
				else
				{
					/*
					---Slight offset
					---Discontinuous.
					---Seems to make aggregates.
					*/
					double overR = 1.0/r;
					double hardShell = std::pow(overR,36);
					//fNet=(hardShell-wellDepth);

					/*
					---Jump to relative infinity.
					---Discontinuous.
					---Needs to be retested.
					*/
					//fNet = 10*wellDepth;

					/*
					---Smooth to hard shell.
					---Continuous.
					---To be tested.
					*/
					double rOffset = 1.0/size;
					fNet=(hardShell-rOffset-wellDepth);
				}
				//Need to switch the sign of the force.
				//Positive is attractive; Negative repulsive.
				fNet = -fNet;

				//Update net potential.
				pot += gamma*rExp/r;

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
					//This error should only get thrown in the case of numerical instability.
					debugging::error::throwInfiniteForce();
				}

				//Add to the net force on the particle.
				index->updateForce(fx,fy,fz,pot,it->second);
			}
		}
	}
}

void Yukawa::getAcceleration(int index, int nPart, int boxSize, double time, simulation::cell* itemCell, simulation::particle** items)
{
	for(auto it = itemCell->getFirstNeighbor(); it != itemCell->getLastNeighbor(); ++it)
	{
		iterCells(boxSize,time,items[index],*it);
	}
}

