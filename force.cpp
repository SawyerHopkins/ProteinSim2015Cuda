#include "force.h"

namespace physics
{

	/********************************************//**
	*---------------MANAGER CONSTRUCTION-------------
	 ***********************************************/

	forces::forces()
	{
		timeDependent=false;
	}

	forces::~forces()
	{
		//Free memory from the Force associated with the IForce Pointer.
		for (std::vector<IForce*>::iterator i = flist.begin(); i != flist.end(); ++i)
		{
			delete[] *i;
		}
		//Free memory of the IForce Pointers.
		delete[] &flist;
		delete[] &timeDependent;
	}

	/********************************************//**
	*-----------------FORCE MANAGEMENT---------------
	 ***********************************************/

	void forces::addForce(IForce* f)
	{
		flist.push_back(f);
		if (f->isTimeDependent())
		{
			timeDependent=true;
		}
	}

	void forces::getAcceleration(int nPart, int boxSize, double time, simulation::cell**** cells, simulation::particle** items)
	{
		//Iterate across all elements in the system.
		for (int index = 0; index < nPart; index++)
		{
			//Resets the force on the particle.
			items[index]->clearForce();

			simulation::particle* p = items[index];
			simulation::cell* itemCell = cells[p->getCX()][p->getCY()][p->getCZ()];

			//Iterates through all forces.
			for (std::vector<IForce*>::iterator it = flist.begin(); it != flist.end(); it++)
			{
				(*it)->getAcceleration(index, nPart, boxSize, time, itemCell, items);
			}
		}
	}

}
