#include <math.h>
#include <string>
#include <iostream>
#include <chrono>
#include <thread>

#include "point.h"
#include "integrator.h"
#include "force.h"
#include "utilities.h"

using namespace std;

/*-----------------------------------------*/
/*----------FUNCTION DECLARATIONS----------*/
/*-----------------------------------------*/
/*----See function for full description----*/
/*-----------------------------------------*/

static inline void debug(mathTools::points* pt);
static inline void greeting();


/*-----------------------------------------*/
/*--------------PROGRAM MAIN---------------*/
/*-----------------------------------------*/

int main(int argc, char **argv)
{

	greeting();

	/*-------------Variables-------------*/

	//Initialize random number generator.
	srand (time(NULL));
	//Set the maximum time.
	double endTime = 10000;
	//Set the time step for the integrator.
	double timeStep = .001;
	//Set the number of particles.
	double nParticles = 10000;
	//Set drag coefficent.
	double gamma = 750.0;
	//Set initial temperature.
	double t_initial = 1.0;

	/*-------------Setup-------------*/

	//Create the integrator.
	cout << "Creating integrator.\n";
	integrators::verlet * difeq = new integrators::verlet(timeStep);

	cout << "Creating particle system.\n";
	//Creates the particle system.
	mathTools::points * pt = new mathTools::points(nParticles, 0.5, t_initial);
	//Initialize the particle system with random position and velocity.
	pt->init(0.10);

	//Creates a force manager.
	cout << "Adding required forces.\n\n";
	physics::forces * force = new physics::forces();
	force->addForce(new physics::AOPotential(.46,1.1)); //Adds the aggregation force.
	//force->addForce(new physics::dragForce(gamma)); //Adds drag.
	//force->addForce(new physics::brownianForce(gamma,1.0,t_initial,timeStep,nParticles)); //Adds brownian dynamics.

	//Output the stats.
	cout << "Number of Particles: " << pt->arrSize << "\n";
	cout << "Box Size: " << pt->getBoxSize() << "\n\n";

	//Write the initial system.
	cout << "Writing initial system to file.\n\n";
	pt->writeSystem("initSys");

	//std::this_thread::sleep_for(std::chrono::milliseconds(5000));

	/*-------------Iterator-------------*/
	cout << "Starting integration.\n\n";
	long counter = 0;
	while(difeq->getSystemTime() < endTime)
	{
		for (int i =0; i < pt->arrSize; i++)
		{
			difeq->nextSystem(i,pt,force);
		}
		utilities::loadBar(difeq->getSystemTime(),endTime,counter);
		counter++;
		difeq->advanceTime();

		if ((counter % 1000) == 0)
		{
			std::string name = "system" + std::to_string(counter) + ".txt";
			//pt->writeSystem(name);
		}

		//debug(pt);
		//std::this_thread::sleep_for(std::chrono::milliseconds(3000));
	}

	//Write the final system.
	cout << "\n" << "Integration complete. Writing final system to file.";
	pt->writeSystem("finSys");

	//Debug code 0 -> No Error:
	return 0;
}

/*-----------------------------------------*/
/*--------------AUX FUNCTIONS--------------*/
/*-----------------------------------------*/

//Writes the state of the system to the console.
void debug(mathTools::points* pt)
{
	/*-------------Debugging-------------*/
	/*-Out the position of each particle-*/
	for (int i = 0; i < pt->arrSize; i++)
	{
		pt->writePosition(i);
	}
	cout << "\n";
	//cout << "\n" << dist(pt->getX(0),pt->getX(1),pt->getY(0),pt->getY(1),pt->getZ(0),pt->getZ(1)) << "\n\n";
}

//Output the program name and information.
void greeting()
{
	cout << "---Particle Simulator 2015---\n";
	cout << "----Sawyer Hopkins et al.----\n\n";
}