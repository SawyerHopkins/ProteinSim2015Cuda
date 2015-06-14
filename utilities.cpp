#include "utilities.h"

using namespace std;

namespace utilities
{
	//fmod can't do negative numbers so use this.
	double util::safeMod(double val, double base)
	{
		double output = 0.0;
		//0 mod n is always zero
		if (val == 0)
		{
			return 0.0;
		}
		//if value is negative wrap the value around base.
		else if (val < 0)
		{
			output = fmod(fabs(val),base);
			return (base-output);
		}
		//if value is positive we can use the standard fmod.
		else
		{
			return fmod(val,base);
		}
	}

	// Process has done i out of n rounds,
	// and we want a bar of width w and resolution r.
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

	//Gets the distance between to particles considering periodic boundary conditions.
	double util::pbcDist(double v1, double v2, double size)
	{
		//If the particles are further than half the box size away from each other
		//then then they are closer periodically.
		if (fabs(v1-v2) > size/2)
		{
			//Set the periodic distance and reverse the relative sign.
			if (v1 < v2)
			{
				return (size-fabs(v1-v2));
			}
			else
			{
				return -(size-fabs(v1-v2));
			}
			return (v2-v1);
		}
		else 
		{
			return (v1-v2);
		}
	}

	//Normalizes the distances to create a unit vector in &acc[3].
	void util::unitVector(double dX, double dY, double dZ, double r, double (&acc)[3])
	{
		acc[0]=-dX/r;
		acc[1]=-dY/r;
		acc[2]=-dZ/r;
	}

}

