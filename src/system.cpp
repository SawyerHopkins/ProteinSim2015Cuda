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

	system::system(configReader::config* cfg, integrators::I_integrator* sysInt, physics::forces* sysFcs)
	{

		//Sets the trial name
		std::string keyName = "trialName";
		if (cfg->containsKey(keyName))
		{
			trialName = cfg->getParam<std::string>(keyName);
		}
		else
		{
			std::cout << "-Option: '" << keyName << "' missing\n\n";
			runSetup();
		}
		std::cout << "\n---" << keyName << ": " << trialName << "\n";

		//Set time information
		currentTime = 0;

		keyName = "timeStep";
		if (cfg->containsKey(keyName))
		{
			dTime = cfg->getParam<double>(keyName);
		}
		else
		{
			std::cout << "-Option: '" << keyName << "' missing\n";
			std::cout << "-Using default.\n\n";
			dTime = 0.001;
		}
		std::cout << "---" << keyName << ": " << dTime << "\n";

		//Set the random number generator seed.
		keyName = "seed";
		if (cfg->containsKey(keyName))
		{
			seed = cfg->getParam<int>(keyName);
		}
		else
		{
			std::cout << "-Option: '" << keyName << "' missing\n";
			std::cout << "-Using default.\n\n";
			seed = 90210;
		}
		std::cout << "---" << keyName << ": " << seed << "\n";


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

		//Set the number of particles.
		keyName = "nParticles";
		if (cfg->containsKey(keyName))
		{
			nParticles = cfg->getParam<int>(keyName);
		}
		else
		{
			std::cout << "-Option: '" << keyName << "' missing\n";
			std::cout << "-Using default.\n\n";
			nParticles = 1000;
		}
		std::cout << "---" << keyName << ": " << nParticles << "\n";

		//Set the integration method.
		integrator = sysInt;

		//Set the internal forces.
		sysForces = sysFcs;

		//Set the concentration.
		keyName = "conc";
		double conc = 0.1;
		if(cfg->containsKey(keyName))
		{
			conc = cfg->getParam<double>(keyName);
		}
		else
		{
			std::cout << "-Option: '" << keyName << "' missing\n";
			std::cout << "-Using default.\n\n";
		}
		std::cout << "---" << keyName << ": " << conc << "\n";

		//Set the radius.
		keyName = "radius";
		double r = 0.5;
		if (cfg->containsKey(keyName))
		{
			r = cfg->getParam<double>(keyName);
		}
		else
		{
			std::cout << "-Option: '" << keyName << "' missing\n";
			std::cout << "-Using default.\n\n";
		}
		std::cout << "---" << keyName << ": " << r << "\n";

		//Set the mass.
		keyName = "mass";
		double m = 1.0;
		if(cfg->containsKey(keyName))
		{
			m = cfg->getParam<double>(keyName);
		}
		else
		{
			std::cout << "-Option: '" << keyName << "' missing\n";
			std::cout << "-Using default.\n\n";
		}
		std::cout << "---" << keyName << ": " << m << "\n";

		//Set the scale.
		keyName = "scale";
		int scale = 4;
		if(cfg->containsKey(keyName))
		{
			scale = cfg->getParam<int>(keyName);
		}
		else
		{
			std::cout << "-Option: '" << keyName << "' missing\n";
			std::cout << "-Using default.\n\n";
		}
		std::cout << "---" << keyName << ": " << scale << "\n\n";

		//Create a box based on desired concentration.
		double vP = nParticles*(4.0/3.0)*atan(1)*4*r*r*r;
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

		//Define one second of run time.
		int oneSec = 1.0/dTime;

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

			//Output a snapshot every second.
			if ( (counter % oneSec) == 0 )
			{
				std::string outName = std::to_string(int(std::round(currentTime)));
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
				double avgCoor = double(totCoor) / double(nParticles);

				std::cout <<"<Rg>: " << avgCoor << " - <Rt>: " << totCoor << "\n";
				std::cout <<"<EAP>: " << eap << "\n";
				std::cout <<"Clusters: " << nClust << "\n\n";

				//Output the number of clusters with time.
				std::ofstream myFileClust(trialName + "/clustGraph.txt", std::ios_base::app | std::ios_base::out);
				myFileClust << currentTime << " " << nClust << "\n";
				myFileClust.close();


				//Output the average potential with time.
				std::ofstream myFilePot(trialName + "/potGraph.txt", std::ios_base::app | std::ios_base::out);
				myFilePot << currentTime << " " << eap << "\n";
				myFilePot.close();

				//Output the coordination number with time
				std::ofstream myFileCoor(trialName + "/coorGraph.txt", std::ios_base::app | std::ios_base::out);
				myFileCoor << currentTime << " " << avgCoor << "\n";
				myFileCoor.clear();

			}

			//Update loading bar.
			utilities::util::loadBar(currentTime,endTime,counter);

			//Increment counters.
			currentTime += dTime;
			counter++;
		}
	}

}

