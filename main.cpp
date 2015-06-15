#include "system.h"

using namespace std;

/*-----------------------------------------*/
/*----------FUNCTION DECLARATIONS----------*/
/*-----------------------------------------*/
/*----See function for full description----*/
/*-----------------------------------------*/

static inline void debug(simulation::system* sys);
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
	int nParticles = 100;
	//Set drag coefficent.
	double gamma = 750.0;
	//Set initial temperature.
	double temp = 1.0;
	//Set concentration.
	double conc = 0.10;
	//Set cell scale.
	int scale = 4;
	//Set rnd seed. 0 for random seed.
	int rnd = 90210;
	//Set particle radius.
	double r = 0.5;
	//Set particle mass.
	double m = 1.0;
	//Set the cutoff.
	double cutOff = 1.1;
	//Set the kT well depth.
	double kT = 0.46;

	/*-------------INTEGRATOR-------------*/

	//Create the integrator.
	cout << "Creating integrator.\n";
	integrators::brownianIntegrator * difeq = new integrators::brownianIntegrator(nParticles, temp, m, gamma, timeStep);

	/*---------------FORCES---------------*/

	//Creates a force manager.
	cout << "Adding required forces.\n";
	physics::forces * force = new physics::forces();
	force->addForce(new physics::AOPotential(kT,cutOff)); //Adds the aggregation force.

	/*---------------SYSTEM---------------*/

	cout << "Creating particle system.\n";
	//Creates the particle system.
	simulation::system * sys = new simulation::system(nParticles, conc, scale, m, r, temp, timeStep, rnd, difeq, force);

	/*---------------RUNNING--------------*/

	//Output the stats.
	cout << "Number of Particles: " << sys->getNParticles() << "\n";
	cout << "Box Size: " << sys->getBoxSize() << "\n\n";

	//Write the initial system.
	cout << "Writing initial system to file.\n\n";
	sys->writeSystem("initSys");

	/*-------------Iterator-------------*/
	cout << "Starting integration.\n\n";

	sys->run(endTime);

	//Write the final system.
	cout << "\n" << "Integration complete. Writing final system to file.";
	sys->writeSystem("finSys");

	//Debug code 0 -> No Error:
	return 0;
}

/*-----------------------------------------*/
/*--------------AUX FUNCTIONS--------------*/
/*-----------------------------------------*/

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