#include "force.h"

namespace physics
{

	void forces::getAcceleration(float pos[], float vel[], float t, float (&acc)[3])
		{
			for (std::vector<IForce*>::iterator i = flist.begin(); i != flist.end(); i++)
			{
				acc[0]=(*i)->getAcceleration(pos[0],vel[0],t);
				acc[1]=(*i)->getAcceleration(pos[1],vel[1],t);
				acc[2]=(*i)->getAcceleration(pos[2],vel[2],t);
			}
		}

	float electroStaticForce::getAcceleration(float pos, float vel, float time)
	{
		return 0.0;
	}
}
