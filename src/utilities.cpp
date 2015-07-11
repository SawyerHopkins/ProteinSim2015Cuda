#include "utilities.h"

using namespace std;

namespace utilities
{

	double util::safeMod(double val, double base)
	{
		//0 mod n is always zero
		if (val == 0)
		{
			return 0.0;
		}
		//if the particle is on the edge of the system.
		else if (val == base)
		{
			return 0.0;
		}
		//if the particle is outside the upper bounds.
		else if (val>base)
		{
			return (val-base);
		}
		//if the particle is outside the lower bounds.
		else if (val<0)
		{
			return (val+base);
		}
		//No problems return value.
		else
		{
			return val;
		}
	}

	double util::safeMod0(double val0, double val, double base)
	{
		//The difference between the two values.
		double dx = val - val0;
		//If the values are further apart than half the system, use PBC.
		if (fabs(dx) > base/2 )
		{
			//Check which direction to implement PBC.
			if (dx < 0)
			{
				return val0-base;
			}
			else
			{
				return val0+base;
			}
		}
		else
		{
			return val0;
		}
		return 0.0;
	}

	void util::loadBar(double x0, int n, long counter, int w)
	{
		/**************************************************************************************//**
		*----------------------------------------SOURCE FROM---------------------------------------
		*------------------------------------------------------------------------------------------
		*---https://www.ross.click/2011/02/creating-a-progress-bar-in-c-or-any-other-console-app---
		*------------------------------------------------------------------------------------------
		******************************************************************************************/
		//if ( (x != n) && (x % (n/100+1) != 0) ) return;

		int x = (int)x0;

		//Choose when to update console.
		if ( (x != n) && (counter % 100 != 0) ) return;

		double ratio  =  x/(double)n;
		int   c      =  ratio * w;

		cout.precision(3);

		cout << setw(3) << (int)(ratio*100) << "% [";
		for (int x=0; x<c; x++) cout << "=";
		for (int x=c; x<w; x++) cout << " ";
		cout << "] - " << x0 << "\r" << flush;
	}

	double util::pbcDist(double X,double Y, double Z,double X1, double Y1,double Z1,double L)
	{

		double r,dx,dy,dz;

		//Check X direction.
		if(fabs(X-X1) <= L/2 )
		{
			dx=fabs(X-X1);
		}
		else
		{
			dx=fabs(X-X1);
			dx=fabs(dx-L);
		}

		//Check Y direction.
		if(fabs(Y-Y1) <= L/2 )
		{
			dy=fabs(Y-Y1);
		}
		else
		{
			dy=fabs(Y-Y1);
			dy=fabs(dy-L);
		}

		//Check Z direction.
		if(fabs(Z-Z1) <= L/2 )
		{
			dz=fabs(Z-Z1);
		}
		else
		{
			dz=fabs(Z-Z1);
			dz=fabs(dz-L);
		}

		//Pythag for the distance.
		r=(dx*dx)+(dy*dy)+(dz*dz);


		return r;

	}

	void util::unitVectorSimple(double dX, double dY, double dZ, double r, double (&acc)[3])
	{
		//Normalize by distance.
		acc[0]=dX/r;
		acc[1]=dY/r;
		acc[2]=dZ/r;
	}

	void util::unitVectorAdv(double X,double Y, double Z,double X1, double Y1,double Z1,double (&acc)[3],double r,int L)
	{
		double dx,dy,dz;

		dx=X1-X; dy=Y1-Y; dz=Z1-Z;

		//Check X PBC.
		if(fabs(dx) > L/2)
		{
			if(dx<0)
			{
				dx=dx+L;
			}
			else
			{
				dx=dx-L;
			}
		}

		//Check Y PBC.
		if(fabs(dy) > L/2)
		{
			if(dy<0)
			{
				dy=dy+L;
			}
			else
			{
				dy=dy-L;
			}
		}

		//Check Z PBC.
		if(fabs(dz) > L/2)
		{
			if(dz<0)
			{
				dz=dz+L;
			}
			else
			{
				dz=dz-L;
			}
		}

		//Normalize by distance.
		dx=dx/r; dy=dy/r; dz=dz/r;
		acc[0]=dx; acc[1]=dy; acc[2]=dz;
	}

}

