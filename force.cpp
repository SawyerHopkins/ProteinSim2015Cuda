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

	void forces::getAcceleration(int index, double time, simulation::particle** items)
		{
			//Iterate across all elements in the system.
			//for (std::vector<IForce*>::iterator i = flist.begin(); i != flist.end(); ++i)
			//{
			//	double subAcc[3] = {0.0,0.0,0.0};
			//	(*i)->getAcceleration(index,time,pts,subAcc);
			//	//Update the net acceleration.
			//	*(acc+0)+=*(subAcc+0);
			//	*(acc+1)+=*(subAcc+1);
			//	*(acc+2)+=*(subAcc+2);
			//}
		}

}
