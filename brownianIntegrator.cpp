#include "integrator.h"

namespace integrators
{

	brownianIntegrator::brownianIntegrator(int nPart, double tempInit, double m, double dragCoeff, double dTime) :
	memX(new double[nPart]), memY(new double[nPart]), memZ(new double[nPart]),
	memCorrX(new double[nPart]), memCorrY(new double[nPart]), memCorrZ(new double[nPart])
	{

		//Stores the system information.
		memSize = nPart;
		temp = tempInit;
		mass = m;

		//Create vital variables
		gamma = dragCoeff;
		dt = dTime;
		y = gamma*dt;

		//Create G+B Variables
		coEff0 = exp(-y);
		coEff1 = (1.0-coEff0)/y;
		coEff2 = ((0.5*y*(1.0+coEff0))-(1.0-coEff0))/(y*y);
		coEff3 = (y-(1.0-coEff0))/(y*y);

		//Create G+B EQ 2.12 for gaussian width.
		double sig0 = temp/(mass*gamma*gamma);
		double sig1 = sig0 * getWidth(y);
		double sig2 = -sig0 * getWidth(-y);

		//Creates the random device.
		std::random_device rd;
		gen = new std::mt19937(rd());
		posDist = new std::normal_distribution<double>(0.0,sig1);
		negDist = new std::normal_distribution<double>(0.0,sig2);

	}

	double brownianIntegrator::getWidth(double gdt)
	{
		return (2*gdt) - 3.0 + (4.0*exp(-gdt)) - exp(-2.0*gdt);
	}

	int brownianIntegrator::nextSystem(double time, double dt, int nParticles, int boxSize, simulation::particle** items, physics::forces* f)
	{
		//Updates the force.
		for (int i=0; i < nParticles; i++)
		{
			f->getAcceleration(i, time, items);
		}

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
		for (int i=0; i < nParticles; i++)
		{
			//SEE GUNSTEREN AND BERENDSEN 1981 EQ 2.26

			memCorrX[i] = 0.0;
			memCorrY[i] = 0.0;
			memCorrZ[i] = 0.0;

			memX[i] = (*posDist)(*gen);
			memY[i] = (*posDist)(*gen);
			memZ[i] = (*posDist)(*gen);

			double m = 1.0/items[i]->getMass();
			double xNew = items[i]->getX() + (items[i]->getVX() * coEff1 * dt) + (items[i]->getFX() * coEff3 * dt * dt * m) + memX[i];
			double yNew = items[i]->getY() + (items[i]->getVY() * coEff1 * dt) + (items[i]->getFY() * coEff3 * dt * dt * m) + memY[i];
			double zNew = items[i]->getZ() + (items[i]->getVZ() * coEff1 * dt) + (items[i]->getFZ() * coEff3 * dt * dt * m) + memZ[i];
			items[i]->setPos(xNew,yNew,zNew,boxSize);

		}
		return 0;
	}

	int brownianIntegrator::normalStep(double time, double dt, int nParticles, int boxSize, simulation::particle** items, physics::forces* f)
	{
		return 0;
	}

}