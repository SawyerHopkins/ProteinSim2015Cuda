#include "particle.h"

namespace simulation
{

/*-----------------------------------------*/
/*-----------SYSTEM CONSTRUCTION-----------*/
/*-----------------------------------------*/

	particle::particle()
	{
	}

	particle::~particle()
	{
		delete[] &x;
		delete[] &y;
		delete[] &z;
		delete[] &x0;
		delete[] &y0;
		delete[] &z0;
		delete[] &fx;
		delete[] &fy;
		delete[] &fz;
		delete[] &fx0;
		delete[] &fy0;
		delete[] &fz0;
		delete[] &vx;
		delete[] &vy;
		delete[] &vz;
		delete[] &m;
		delete[] &r;
		delete[] &cx;
		delete[] &cy;
		delete[] &cz;
		delete[] &index;
	}


}

