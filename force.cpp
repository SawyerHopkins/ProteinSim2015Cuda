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

	//Calculates the total acceleration to get net force.
	//Uses pointer to acc to update potentials in the I_Integrator.
	void forces::getAcceleration(float pos[], float vel[], float t, float (&acc)[3])
		{
			//Iterate across all elements in the system.
			for (std::vector<IForce*>::iterator i = flist.begin(); i != flist.end(); ++i)
			{
				acc[0]=(*i)->getAcceleration(pos[0],vel[0],t);
				acc[1]=(*i)->getAcceleration(pos[1],vel[1],t);
				acc[2]=(*i)->getAcceleration(pos[2],vel[2],t);
			}
		}

	//Get the acceleration from the Coloumb potential.
	float electroStaticForce::getAcceleration(float pos, float vel, float time)
	{
		return 0.0;
	}
}
