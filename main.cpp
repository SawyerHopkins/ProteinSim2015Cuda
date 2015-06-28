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

#include "findClusters.h"

using namespace std;

/********************************************//**
*-------------FUNCTION DECLARATIONS--------------
*------------------------------------------------
*------------see function for details------------
************************************************/

static inline void debug(simulation::system* sys);
static inline void greeting();
static void runScript();
static void runAnalysis(string fileName);
static string runSetup();

/********************************************//**
*------------------MAIN PROGRAM------------------
************************************************/

/**
 * @brief The program entry point.
 * @param argc Not implemented.
 * @param argv Not implemented.
 * @return 
 */
int main(int argc, char **argv)
{

	//Program welcome.
	greeting();

	//Program flags.
	bool a = false;
	string aName = "";

	int i = 0;
	//Iterate across input arguments.
	while (i < argc)
	{
		//Flag for analysis mode.
		string str(argv[i]);
		if (str.compare("-a")==0)
		{
			a = true;
			//Make sure the next argument exists.
			if ((i+1) < argc)
			{
				//Name of file to analyze.
				i++;
				string file(argv[i]);
				aName = file;
			}
			else
			{
				debugging::error::throwInputError();
			}
		}
		i++;
	}

	//Choose which mode to run.
	if (a)
	{
		runAnalysis(aName);
	}
	else
	{
		runScript();
	}

	//Debug code 0 -> No Error:
	return 0;
}

/********************************************//**
*-----------------MAIN FUNCTIONS-----------------
************************************************/

/**
 * @brief Run a new simulation.
 */
void runScript()
{
	/*-------------Variables-------------*/

	//Initialize random number generator.
	srand (time(NULL));
	//Set the maximum time.
	double endTime = 1000;
	//Set the time step for the integrator.
	double timeStep = .001;
	//Set the number of particles.
	int nParticles = 10000;
	//Set drag coefficent.
	double gamma = 0.5;
	//Set initial temperature.
	double temp = 1.0;
	//Set concentration.
	double conc = 0.01;
	//Set cell scale.
	int scale = 40;
	//Set rnd seed. 0 for random seed.
	int rnd = 90210;
	//Set particle radius.
	double r = 0.5;
	//Set particle mass.
	double m = 1.0;
	//Set the cutoff.
	double cutOff = 1.1;
	//Set the kT well depth.
	double kT = 0.261;

	/*------------GET TRIAL NAME----------*/

	string trialName = runSetup();

	/*--------WRITE SYSTEM SETTINGS-------*/

	//Make a directory for the run.
	mkdir(trialName.c_str(),0777);
	//Create a stream to the desired file.
	std::ofstream myFile;
	myFile.open(trialName + "/settings.txt");

	//Output System Constants.
	myFile << "trialName: " << trialName << "\n";
	myFile << "nParticles: " << nParticles << "\n";
	myFile << "endTime: " << endTime << "\n";
	myFile << "timeStep: " << timeStep << "\n";
	myFile << "gamma: " << gamma << "\n";
	myFile << "temp: " << temp << "\n";
	myFile << "conc: " << conc << "\n";
	myFile << "scale: " << scale << "\n";
	myFile << "rnd: " << rnd << "\n";
	myFile << "r: " << r << "\n";
	myFile << "m: " << m << "\n";
	myFile << "cutOff: " << cutOff << "\n";
	myFile << "kT: " << kT << "\n";

	/*-------------INTEGRATOR-------------*/

	//Create the integrator.
	cout << "Creating integrator.\n";
	integrators::brownianIntegrator * difeq = new integrators::brownianIntegrator(nParticles, temp, m, gamma, timeStep,rnd);

	myFile << "Integrator: " << difeq->getName() << "\n";

	/*---------------FORCES---------------*/

	//Creates a force manager.
	cout << "Adding required forces.\n";
	physics::forces * force = new physics::forces();
	force->addForce(new physics::AOPotential(kT,cutOff,timeStep)); //Adds the aggregation force.

	//Iterates through all forces.
	for (std::vector<physics::IForce*>::iterator it = force->getBegin(); it != force->getEnd(); ++it)
	{
		myFile << "Force: " << (*it)->getName() << "\n";
	}

	//Close the stream.
	myFile.close();

	/*---------------SYSTEM---------------*/

	cout << "Creating particle system.\n";
	//Creates the particle system.
	simulation::system * sys = new simulation::system(trialName, nParticles, conc, scale, m, r, temp, timeStep, rnd, difeq, force);

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

string runSetup()
{
	//Set the output directory.
	string outDir = "";
	bool validDir = 0;

	//Check that we get a real directory.
	while (!validDir)
	{
		cout << "Working directory: ";
		cin >> outDir;

		//Check that the directory exists.
		struct stat fileCheck;
		if (stat(outDir.c_str(), &fileCheck) != -1)
		{
			if (S_ISDIR(fileCheck.st_mode))
			{
				validDir = 1;
			}
		}

		if (validDir == 0)
		{
			cout << "\n" << "Invalid Directory" << "\n\n";
		}

	}

	//Set the name of the trial.
	string trialName = "";
	bool acceptName = 0;

	//Check that no trials get overwritten by accident.
	while (!acceptName)
	{
		validDir = 0;

		cout << "\n" << "Trial Name: ";
		cin >> trialName;

		//Check input format.
		if (outDir.back() == '/')
		{
			trialName = outDir + trialName;
		}
		else
		{
			trialName = outDir + "/" + trialName;
		}

		//Check that the directory exists.
		struct stat fileCheck;
		if (stat(trialName.c_str(), &fileCheck) != -1)
		{
			if (S_ISDIR(fileCheck.st_mode))
			{
				validDir = 1;
			}
		}

		if (validDir == 1)
		{
			cout << "\n" << "Trial name already exists. Overwrite (y,n): ";

			//Check user input
			std::string cont;
			cin >> cont;

			if (cont == "Y" || cont == "y")
			{
				acceptName = 1;
			}
		}
		else
		{
			acceptName = 1;
		}

	}

	//Output the directory.
	cout << "\n" << "Data will be saved in: " << trialName << "\n\n";

	return trialName;
}

/**
 * @brief Analyze simulation data.
 */
void runAnalysis(string fileName)
{
	analysis::findClusters::runAnalysis(fileName);
}

/********************************************//**
*-----------------AUX FUNCTIONS------------------
************************************************/

/**
 * @brief Writes the state of the system to the console.
 * @param sys The system to output.
 */
void debug(simulation::system* sys)
{
	/*-------------Debugging-------------*/
	/*-Out the position of each particle-*/
	for (int i = 0; i < sys->getNParticles(); i++)
	{
		sys->writePosition(i);
	}
	cout << "\n";
}

/**
 * @brief Output the program name and information.
 */
void greeting()
{
	cout << "---Particle Simulator 2015---\n";
	cout << "----Sawyer Hopkins et al.----\n\n";
}