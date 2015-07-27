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
	*-----------------SYSTEM OUTPUT------------------
	************************************************/

	void system::writeSystem(std::string name)
	{
		//Create a stream to the desired file.
		std::ofstream myFile;
		myFile.open(trialName + name + ".txt");
		//Write each point in the system as a line of csv formatted as: X,Y,Z
		for (int i = 0; i < nParticles; i++)
		{
			myFile << particles[i]->getX() << "," << particles[i]->getY() << "," << particles[i]->getZ() << ",";
			myFile << particles[i]->getX0() << "," << particles[i]->getY0() << "," << particles[i]->getZ0() << ",";
			myFile << particles[i]->getFX() << "," << particles[i]->getFY() << "," << particles[i]->getFZ() << ",";
			myFile << particles[i]->getFX0() << "," << particles[i]->getFY0() << "," << particles[i]->getFZ0();
			if (i < (nParticles-1))
			{
				myFile << "\n";
			}
		}
		//Close the stream.
		myFile.close();
	}

	void system::writeInitTemp()
	{
		double v2 = 0.0;
		//Get V^2 for each particle.
		for (int i = 0; i < nParticles; i++)
		{
			v2 += particles[i]->getVX()*particles[i]->getVX();
			v2 += particles[i]->getVY()*particles[i]->getVY();
			v2 += particles[i]->getVZ()*particles[i]->getVZ();
		}
		//Average v2.
		double vAvg = v2 / float(nParticles);
		double temp = (vAvg / 3.0);
		//
		std::cout << "---Temp: " << temp << " m/k" << "\n";
	}

	void system::writeSystemInit()
	{
		std::ofstream myFile;
		myFile.open(trialName + "/sysConfig.txt");

		//Writes the system configuration.
		myFile << "trialName: " << trialName << "\n";
		myFile << "nParticles: " << nParticles << "\n";
		myFile << "Concentration: " << concentration << "\n";
		myFile << "boxSize: " << boxSize << "\n";
		myFile << "cellSize: " << cellSize << "\n";
		myFile << "cellScale: " << cellScale << "\n";
		myFile << "temp: " << temp << "\n";
		myFile << "dTime: " << dTime;

		//Close the stream.
		myFile.close();
	}

	void system::writeSystemState(debugging::timer* tmr)
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

		std::cout <<"<R>: " << avgCoor << " - Rt: " << totCoor << "\n";
		std::cout <<"<EAP>: " << eap << "\n";
		std::cout <<"<N>/Nc: " << nClust << "\n\n";

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

}