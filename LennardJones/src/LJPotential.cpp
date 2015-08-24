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

#include "LJPotential.h"
using namespace simulation;

LennardJones::~LennardJones()
{
	delete[] &cutOff;
	delete[] &dampening;
	delete[] &mass;
	delete[] &wellDepth;
}

LennardJones::LennardJones(configReader::config* cfg)
{
	//Sets the name
	name = "Lennard Jones";

	//Get the radius
	radius = cfg->getParam<double>("radius",0.5);

	//Get the mass
	mass = cfg->getParam<double>("mass",1.0);

	//Get the well depth
	wellDepth = cfg->getParam<double>("LJDepth",100.0);

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

	utilities::util::writeTerminal("---Lennard Jones Potential successfully added.\n\n", utilities::Colour::Cyan);

}

void LennardJones::iterCells(int boxSize, double time, particle* index, cell* itemCell)
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

				//Attractive LJ.

				double fNet = 0;
				double rMin = 1.0;
				double rInv = rMin/r;
				double n = 18.0;
				double n2 = 2.0*n;
				double v = 21.0;

				double fn = n+1.0;
				double f2n = n2+1.0;

				double att = (n2*std::pow(rInv,f2n));
				att -= (n*std::pow(rInv,fn));
				att = (att*v*4.0);

				//Repulsive Yukawa

				double k = 0.5;
				double kInv = 1.0/k;
				double oneOver = 1.0/r;
				double oneOver2 = std::pow(oneOver,2.0);
				double rexp = std::exp(-kInv*r);

				double repel = (oneOver2*rexp);
				repel += (kInv*oneOver*rexp);
				repel = (repel * 8 * v * k);

				fNet = (att + repel);

				//Positive is attractive; Negative repulsive.
				fNet = -fNet;

				//Update net potential.
				double ljPot = std::pow(rInv,n2);
				ljPot -= std::pow(rInv,n);
				ljPot = (ljPot * 4 * v);

				double yukPot = (rexp*oneOver);
				yukPot = (yukPot * 8 * v * k);

				pot += yukPot;
				pot += ljPot;

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

void LennardJones::getAcceleration(int index, int nPart, int boxSize, double time, simulation::cell* itemCell, simulation::particle** items)
{
	for(auto it = itemCell->getFirstNeighbor(); it != itemCell->getLastNeighbor(); ++it)
	{
		iterCells(boxSize,time,items[index],*it);
	}
}

