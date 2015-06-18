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
		else if (val>base)
		{
			return (val-base);
		}
		else if (val<0)
		{
			return (val+base);
		}
		else
		{
			return val;
		}
	}

	double util::safeMode0(double val0, double val, double base)
	{
		double dx = val - val0;
		if (fabs(dx) > base/2 )
		{
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
		/*-----------------------------------------*/
		/*---------------SOURCE FROM---------------*/
		/*-----------------------------------------*/
		/* https://www.ross.click/2011/02/creating-a-progress-bar-in-c-or-any-other-console-app/*/
		/*-----------------------------------------*/
		//if ( (x != n) && (x % (n/100+1) != 0) ) return;

		int x = (int)x0;

		if ( (x != n) && (counter % 10 != 0) ) return;

		double ratio  =  x/(double)n;
		int   c      =  ratio * w;

		cout.precision(3);

		cout << setw(3) << (int)(ratio*100) << "% [";
		for (int x=0; x<c; x++) cout << "=";
		for (int x=c; x<w; x++) cout << " ";
		cout << "] - " << x0 << "\r" << flush;
	}

	double util::pbcDistAlt(double X,double Y, double Z,double X1, double Y1,double Z1,double L)
	{

		double r,dx,dy,dz;

		if(fabs(X-X1) <= L/2 )
		{
			dx=fabs(X-X1);
		}
		else
		{
			dx=fabs(X-X1);
			dx=fabs(dx-L);
		}


		if(fabs(Y-Y1) <= L/2 )
		{
			dy=fabs(Y-Y1);
		}
		else
		{
			dy=fabs(Y-Y1);
			dy=fabs(dy-L);
		}

		if(fabs(Z-Z1) <= L/2 )
		{
			dz=fabs(Z-Z1);
		}
		else
		{
			dz=fabs(Z-Z1);
			dz=fabs(dz-L);
		}

		r=(dx*dx)+(dy*dy)+(dz*dz);


		return r;

	}

	void util::unitVector(double dX, double dY, double dZ, double r, double (&acc)[3])
	{
		acc[0]=dX/r;
		acc[1]=dY/r;
		acc[2]=dZ/r;
	}

	void util::unitVectorAlt(double X,double Y, double Z,double X1, double Y1,double Z1,double (&acc)[3],double r,int L)
	{
		double dx,dy,dz;

		dx=X1-X; dy=Y1-Y; dz=Z1-Z;
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

		dx=dx/r; dy=dy/r; dz=dz/r;
		acc[0]=dx; acc[1]=dy; acc[2]=dz;
	}

}

