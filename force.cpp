#include "force.h"

namespace physics
{

	//Constructor for the force manager.
	forces::forces()
	{
		timeDependent=false;
	}

	//Destructor for the force manager.
	//Be careful with the pointers.
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

	void forces::addForce(IForce* f)
	{
		flist.push_back(f);
		if (f->isTimeDependent())
		{
			timeDependent=true;
		}
	}

	//Calculates the total acceleration to get net force.
	//Uses pointer to acc to update potentials in the I_Integrator.
	void forces::getAcceleration(float index, float time, mathTools::points* pts, float (&acc)[3])
		{
			//Iterate across all elements in the system.
			for (std::vector<IForce*>::iterator i = flist.begin(); i != flist.end(); ++i)
			{
				float subAcc[3] = {0.0,0.0,0.0};
				(*i)->getAcceleration(index,time,pts,subAcc);
				*(acc+0)+=*(subAcc+0);
				*(acc+1)+=*(subAcc+1);
				*(acc+2)+=*(subAcc+2);
			}
		}

}
