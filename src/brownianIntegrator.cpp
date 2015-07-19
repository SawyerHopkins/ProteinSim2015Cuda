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

#include "integrator.h"

namespace integrators
{

	brownianIntegrator::brownianIntegrator(configReader::config* cfg)
	{

		//Sets the name
		name = "brownianIntegrator";

		//Set the number of particles.
		std::string keyName = "nParticles";
		if (cfg->containsKey(keyName))
		{
			memSize = cfg->getParam<int>(keyName);
		}
		else
		{
			std::cout << "-Option: '" << keyName << "' missing\n";
			std::cout << "-Using default.\n\n";
			memSize = 1000;
		}
		std::cout << "---" << keyName << ": " << memSize << "\n";

		//Create he memory blocks for mem and memCoor
		memX = new double[memSize];
		memY = new double[memSize];
		memZ = new double[memSize];
		memCorrX = new double[memSize];
		memCorrY = new double[memSize];
		memCorrZ = new double[memSize];

		//Sets the system temperature.
		keyName = "temp";
		if (cfg->containsKey(keyName))
		{
			temp = cfg->getParam<double>(keyName);
		}
		else
		{
			std::cout << "-Option: '" << keyName << "' missing\n";
			std::cout << "-Using default.\n\n";
			temp = 1.0;
		}
		std::cout << "---" << keyName << ": " << temp << "\n";

		//Set the mass.
		keyName = "mass";
		if(cfg->containsKey(keyName))
		{
			mass = cfg->getParam<double>(keyName);
		}
		else
		{
			std::cout << "-Option: '" << keyName << "' missing\n";
			std::cout << "-Using default.\n\n";
			mass = 1.0;
		}
		std::cout << "---" << keyName << ": " << mass << "\n";

		//Sets the system drag.
		keyName = "gamma";
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
		std::cout << "---" << keyName << ": " << gamma << "\n";

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

		//Create vital variables
		y = gamma*dt;

		//Create G+B Variables
		coEff0 = exp(-y);
		coEff1 = (1.0-coEff0)/y;
		coEff2 = ((0.5*y*(1.0+coEff0))-(1.0-coEff0))/(y*y);
		coEff3 = (y-(1.0-coEff0))/(y*y);

		//Create G+B EQ 2.12 for gaussian width.
		double sig0 = temp/(mass*gamma*gamma);
		sig1 = std::sqrt( sig0 * getWidth(y) );
		sig2 = std::sqrt( -sig0 * getWidth(-y) );

		double gn = exp(y) - exp(-y) - (2.0*y);
		corr = (temp/(gamma*gamma)) * (gn/(sig1*sig2));
		dev = sqrt(1.0 - (corr*corr));

		//Set the random number generator seed.
		keyName = "seed";
		int rSeed = 90210;
		if (cfg->containsKey(keyName))
		{
			rSeed = cfg->getParam<int>(keyName);
		}
		else
		{
			std::cout << "-Option: '" << keyName << "' missing\n";
			std::cout << "-Using default.\n\n";
		}
		std::cout << "---" << keyName << ": " << rSeed << "\n";

		//Creates the random device.
		gen = new std::mt19937(rSeed);
		Dist = new std::normal_distribution<double>(0.0,1.0);

		goy2 = gn / (y*y);
		goy3 = gn / (y*y*y);
		hn  = y/(exp(y)-exp(-y));

		std::cout.precision(7);

		std::cout << "\n---y: " << y;
		std::cout << "\n---sig1: " << sig1;
		std::cout << "\n---sig2: " << sig2;
		std::cout << "\n---coor: " << corr;
		std::cout << "\n---dev: " << dev;
		std::cout << "\n---c0: " << coEff0;
		std::cout << "\n---c1: " << coEff1;
		std::cout << "\n---c2: " << coEff2;
		std::cout << "\n---c3: " << coEff3;
		std::cout << "\n---goy2: " << goy2;
		std::cout << "\n---goy3: " << goy3;
		std::cout << "\n---Brownian integrator successfully added.\n\n";

	}

	brownianIntegrator::~brownianIntegrator()
	{
		delete &mass;
		delete &temp;
		delete &memSize;

		delete &gamma;
		delete &dt;
		delete &y;

		delete &coEff0;
		delete &coEff1;
		delete &coEff2;
		delete &coEff3;

		delete[] memX;
		delete[] memY;
		delete[] memZ;

		delete[] memCorrX;
		delete[] memCorrY;
		delete[] memCorrZ;

		delete &sig1;
		delete &sig2;
		delete &corr;
		delete &dev;

		delete gen;
		delete Dist;
	}

	double brownianIntegrator::getWidth(double y)
	{
		return (2*y) - 3.0 + (4.0*exp(-y)) - exp(-2.0*y);
	}

	int brownianIntegrator::nextSystem(double time, double dt, int nParticles, int boxSize, simulation::cell**** cells, simulation::particle** items, physics::forces* f)
	{

		//Gets the force on each particle.
		f->getAcceleration(nParticles, boxSize, time, cells ,items);

		//Checks what method is needed.
		if (time == 0)
		{
			firstStep(time, dt, nParticles, boxSize, items, f);
		}
		else
		{
			normalStep(time, dt, nParticles, boxSize, items, f);
		}
		return 0;
	}

	int brownianIntegrator::firstStep(double time, double dt, int nParticles, int boxSize, simulation::particle** items, physics::forces* f)
	{
		//Add 4 threads to the team.
		for (int i=0; i < nParticles; i++)
		{

			//SEE GUNSTEREN AND BERENDSEN 1981 EQ 2.26

			memCorrX[i] = 0.0;
			memCorrY[i] = 0.0;
			memCorrZ[i] = 0.0;

			memX[i] = (*Dist)(*gen);
			memY[i] = (*Dist)(*gen);
			memZ[i] = (*Dist)(*gen);

			double m = 1.0/items[i]->getMass();
			double xNew = items[i]->getX() + (items[i]->getVX() * coEff1 * dt) + (items[i]->getFX() * coEff3 * dt * dt * m) + (sig1 * memX[i]);
			double yNew = items[i]->getY() + (items[i]->getVY() * coEff1 * dt) + (items[i]->getFY() * coEff3 * dt * dt * m) + (sig1 * memY[i]);
			double zNew = items[i]->getZ() + (items[i]->getVZ() * coEff1 * dt) + (items[i]->getFZ() * coEff3 * dt * dt * m) + (sig1 * memZ[i]);
			items[i]->setPos(xNew,yNew,zNew,boxSize);

		}
		return 0;
	}

	int brownianIntegrator::normalStep(double time, double dt, int nParticles, int boxSize, simulation::particle** items, physics::forces* f)
	{

		for (int i=0; i < nParticles; i++)
		{

			//SEE GUNSTEREN AND BERENDSEN 1981 EQ 2.26

			//New random walk.
			
			memCorrX[i] = (*Dist)(*gen);
			memCorrY[i] = (*Dist)(*gen);
			memCorrZ[i] = (*Dist)(*gen);

			//Correlation to last random walk.
			memCorrX[i] = sig2 * ((corr * memX[i])+(dev * memCorrX[i]));
			memCorrY[i] = sig2 * ((corr * memY[i])+(dev * memCorrY[i]));
			memCorrZ[i] = sig2 * ((corr * memZ[i])+(dev * memCorrZ[i]));

			memX[i] = (*Dist)(*gen);
			memY[i] = (*Dist)(*gen);
			memZ[i] = (*Dist)(*gen);

			double m = 1.0/items[i]->getMass();

			double x0 = items[i]->getX0();
			double y0 = items[i]->getY0();
			double z0 = items[i]->getZ0();

			double xNew = ((1.0+coEff0) * items[i]->getX());
			xNew -= (coEff0 * x0);
			xNew += (m * dt * dt * coEff1 * items[i]->getFX());
			xNew += (m * dt * dt * coEff2 * (items[i]->getFX() - items[i]->getFX0()));
			xNew += (sig1 * memX[i]) + (coEff0 * memCorrX[i]);

			double yNew = ((1.0+coEff0) * items[i]->getY()) ;
			yNew -= (coEff0 * y0);
			yNew += (m * dt * dt * coEff1 * items[i]->getFY());
			yNew += (m * dt * dt * coEff2 * (items[i]->getFY() - items[i]->getFY0()));
			yNew += (sig1 * memY[i]) + (coEff0 * memCorrY[i]);

			double zNew = ((1.0+coEff0) * items[i]->getZ());
			zNew -= (coEff0 * z0);
			zNew += (m * dt * dt * coEff1 * items[i]->getFZ());
			zNew += (m * dt * dt * coEff2 * (items[i]->getFZ() - items[i]->getFZ0()));
			zNew += (sig1 * memZ[i]) + (coEff0 * memCorrZ[i]);

			items[i]->setPos(xNew, yNew, zNew, boxSize);

		}

		return 0;

	}
}