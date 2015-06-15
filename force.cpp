#include "force.h"

namespace physics
{

/*-----------------------------------------*/
/*-------FORCE MANAGER CONSTRUCTION--------*/
/*-----------------------------------------*/

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

/*-----------------------------------------*/
/*--------------FORCE MANAGER--------------*/
/*-----------------------------------------*/

	void forces::addForce(IForce* f)
	{
		flist.push_back(f);
		if (f->isTimeDependent())
		{
			timeDependent=true;
		}
	}

	void forces::getAcceleration(int nPart, int boxSize, int cellScale, double time, simulation::particle** items)
		{
			//Iterate across all elements in the system.
			for (int index = 0; index < nPart; index++)
			{
				//Resets the force on the particle.
				items[index]->clearForce();
				//Iterates through all forces.
				for (std::vector<IForce*>::iterator i = flist.begin(); i != flist.end(); ++i)
				{
					//Gets the acceleration from the force.
					(*i)->getAcceleration(index, nPart, boxSize, cellScale, time, items);
				}
			}
		}

}
