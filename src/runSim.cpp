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

using namespace std;

/**
 * @brief Run a new simulation.
 */
void runScript()
{
	/*-------------Variables-------------*/

	//Set the maximum time.
	double endTime = 1000;
	//Set the time step for the integrator.
	double timeStep = .001;
	//Set the number of particles.
	int nParticles = 40000;
	//Set drag coefficent.
	double gamma = 0.5;
	//Set initial temperature.
	double temp = 1.0;
	//Set rnd seed. 0 for random seed.
	int rnd = 90210;
	//Set particle mass.
	double m = 1.0;
	//Set the cutoff.
	double cutOff = 1.1;
	//Set the kT well depth.
	double kT = 0.460;

	/*------------GET TRIAL NAME----------*/

	//string trialName = runSetup();

	/*--------WRITE SYSTEM SETTINGS-------*/

	//Make a directory for the run.
	//mkdir(trialName.c_str(),0777);
	//Create a stream to the desired file.
	//std::ofstream myFile;
	//myFile.open(trialName + "/settings.txt");

	//Output System Constants.

	/*-------------INTEGRATOR-------------*/

	//Create the integrator.
	cout << "Creating integrator.\n";
	integrators::brownianIntegrator * difeq = new integrators::brownianIntegrator(nParticles, temp, m, gamma, timeStep,rnd);

	/*---------------FORCES---------------*/

	//Creates a force manager.
	cout << "Adding required forces.\n";
	physics::forces * force = new physics::forces();
	force->addForce(new physics::AOPotential(kT,cutOff,timeStep)); //Adds the aggregation force.

	/*----------------CFG-----------------*/

	cout << "Looking for configuration file.\n\n";
	configReader::config * cfg =new configReader::config("settings.cfg");

	/*---------------SYSTEM---------------*/

	cout << "Creating particle system.\n";
	//Creates the particle system.
	simulation::system * sys = new simulation::system(cfg, difeq, force);

	/*---------------RUNNING--------------*/

	//Output the stats.
	cout << "---Number of Particles: " << sys->getNParticles() << "\n";
	cout << "---Box Size: " << sys->getBoxSize() << "\n";
	cout << "---Cell Size: " << sys->getCellSize() << "\n\n";
 
	//Write the initial system.
	cout << "Writing initial system to file.\n\n";
	sys->writeSystem("/initSys");

	/*-------------Iterator-------------*/

	//Allow user to check system settings before running.
	//Comment this section out if running without terminal access.
	cout << "System initialization complete. Press y/n to continue: ";
	std::string cont;
	cin >> cont;

	if (cont != "Y" && cont != "y")
	{
		exit(100);
	}

	cout << "Starting integration.\n";

	sys->run(endTime);

	//Write the final system.
	cout << "\n" << "Integration complete.\n\n Writing final system to file.";
	sys->writeSystem("/finSys");
}