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

#include "system.h"

namespace simulation
{

	/********************************************//**
	*-------------CONSTRUCTOR/DESTRUCTOR-------------
	************************************************/

	system::system(std::string tName, int nPart, double conc, int scale, double m, double r, double sysTemp, double sysDT, int rnd, integrators::I_integrator* sysInt, physics::forces* sysFcs)
	{

		//Sets the trial name
		trialName = tName;

		//Set time information
		currentTime = 0;
		dTime = sysDT;

		//Set the random number generator seed.
		seed = rnd;

		//Sets the system temperature.
		temp = sysTemp;

		//Set the number of particles.
		nParticles = nPart;

		//Set the integration method.
		integrator = sysInt;

		//Set the internal forces.
		sysForces = sysFcs;

		//Create a box based on desired concentration.
		double vP = nPart*(4.0/3.0)*atan(1)*4*r*r*r;
		boxSize = (int) cbrt(vP / conc);

		//Calculates the number of cells needed.
		cellSize = boxSize / scale;
		boxSize = cellSize * scale;
		cellScale = scale;
		int numCells = pow(scale,3.0);

		//Sets the actual concentration.
		concentration = vP/pow(boxSize,3.0);

		std::cout << "---System concentration: " << concentration << "\n";

		//Create particles.
		initParticles(r,m);

		//Create cells.
		initCells(numCells, cellScale);
		std::cout << "Created: " << numCells << " cells from scale: " <<  cellScale << "\n";

		writeSystemInit();

	}

	system::~system()
	{
		//Deletes the particles
		for (int i=0; i < nParticles; i++)
		{
			delete particles[i];
		}
		delete[] particles;

		//Delete the constants.
		delete &nParticles;
		delete &concentration;
		delete &boxSize;
		delete &cellSize;
		delete &temp;
		delete &currentTime;
		delete &dTime;
		delete &seed;
		delete[] integrator;
		delete[] sysForces;
	}

	void system::run(double endTime)
	{
		//Create the snapshot name.
		std::string snap = trialName + "/snapshots";
		mkdir(snap.c_str(),0777);

		//Debugging counter.
		int counter = 0;

		//Diagnostics timer.
		debugging::timer* tmr = new debugging::timer();
		tmr->start();

		//Run system until end time.
		while (currentTime < endTime)
		{
			//Get the next system.
			integrator->nextSystem(currentTime, dTime, nParticles, boxSize, cells, particles, sysForces);
			//Call cell manager.
			updateCells();

			//Output a snapshot every one second.
			int oneSec = 1.0/dTime;
			if ( (counter % oneSec) == 0 )
			{
				std::string outName = std::to_string(int(currentTime));
				std::cout << "\n" << "Writing: " << outName << ".txt";
				writeSystem("/snapshots/" + outName);
				tmr->stop();
				std::cout << "\n" << "Elapsed time: " << tmr->getElapsedSeconds() << " seconds.\n";
				tmr->start();

				int totCoor = 0;
				int totEAP = 0;
				for (int i=0; i<nParticles; i++)
				{
					totCoor+=particles[i]->getCoorNumber();
					totEAP+=particles[i]->getPotential();
				}

				double eap = (totEAP / double(nParticles));
				double nClust = numClusters();

				std::cout <<"<Rg>: " << int(totCoor/nParticles) << " - <Rt>: " << totCoor << "\n";
				std::cout <<"<EAP>: " << eap << "\n";
				std::cout <<"Clusters: " << nClust << "\n\n";

			}

			//Update loading bar.
			utilities::util::loadBar(currentTime,endTime,counter);

			//Increment counters.
			currentTime += dTime;
			counter++;
		}
	}

}

