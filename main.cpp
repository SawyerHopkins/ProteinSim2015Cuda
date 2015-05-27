#include <math.h>
#include <string.h>
#include <iostream>
#include <iomanip>
#include <chrono>
#include <thread>

#include "point.h"
#include "integrator.h"
#include "force.h"
#include "utilities.h"

using namespace std;

// Process has done i out of n rounds,
// and we want a bar of width w and resolution r.
static inline void loadBar(int x, int n, int r, int w)
{
    // Only update r times.
    if ( x % (n/r +1) != 0 ) return;
 
    // Calculuate the ratio of complete-to-incomplete.
    float ratio = x/(float)n;
    int   c     = ratio * w;
 
    // Show the percentage complete.
    printf("%3d%% [", (int)(ratio*100) );
 
    // Show the load bar.
    for (int x=0; x<c; x++)
       printf("=");
 
    for (int x=c; x<w; x++)
       printf(" ");
 
    // ANSI Control codes to go back to the
    // previous line and clear it.
    printf("]\n\033[F\033[J");
}

static inline float dist(float x1, float x2, float y1, float y2, float z1, float z2)
{
	return std::sqrt((x2-x1)*(x2-x1)+(y2-y1)*(y2-y1)*(z2-z1)*(z2-z1));
}

static inline void debug(mathTools::points* pt)
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

int main(int argc, char **argv)
{
	/*-------------Variables-------------*/

	//Initialize random number generator.
	srand (time(NULL));
	//Set the maximum time.
	float endTime = 10000;
	//Set the time step for the integrator.
	float timeStep = .01;
	//Set the number of particles.
	float nParticles = 100;

	/*-------------Setup-------------*/

	//Create the integrator.
	integrators::verlet * difeq = new integrators::verlet(timeStep);

	//Creates the particle system.
	mathTools::points * pt = new mathTools::points(nParticles, 1);
	//Initialize the particle system with random position and velocity.
	pt->init();

	debug(pt);
	pt->writeSystem("initSys");

	//Creates a force manager.
	physics::forces * force = new physics::forces();
	force->addForce(new physics::electroStaticForce()); //Adds the Coloumb potential.
	force->addForce(new physics::dragForce()); //Adds drag.

	/*-------------Iterator-------------*/
	while(difeq->getSystemTime() < endTime)
	{
		for (int i =0; i < pt->arrSize; i++)
		{
			difeq->nextSystem(i,pt,force);
			difeq->advanceTime();
		}
		//debug(pt);
		//std::this_thread::sleep_for(std::chrono::milliseconds(3000));
		loadBar(difeq->getSystemTime(),endTime,1000,50);
	}

	debug(pt);
	pt->writeSystem("finSys");

	//Debug code 0 -> No Error:
	return 0;
}