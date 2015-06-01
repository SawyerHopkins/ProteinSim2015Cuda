#include "utilities.h"

using namespace std;

namespace mathTools
{
	//fmod can't do negative numbers so use this.
	float utilities::safeMod(float val, float base)
	{
		float output;
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
	void utilities::loadBar(int x, int n, int w)
	{
		/*-----------------------------------------*/
		/*---------------SOURCE FROM---------------*/
		/*-----------------------------------------*/
		/* https://www.ross.click/2011/02/creating-a-progress-bar-in-c-or-any-other-console-app/*/
		/*-----------------------------------------*/
		if ( (x != n) && (x % (n/100+1) != 0) ) return;
	 
		float ratio  =  x/(float)n;
		int   c      =  ratio * w;
	 
		cout << setw(3) << (int)(ratio*100) << "% [";
		for (int x=0; x<c; x++) cout << "=";
		for (int x=c; x<w; x++) cout << " ";
		cout << "]\r" << flush;
	}

	float utilities::pbcDist(float v1, float v2, float size)
	{
		if (fabs(v1-v2) > size/2)
		{
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

	void utilities::unitVector(float dX, float dY, float dZ, float r, float (&acc)[3])
	{
		acc[0]=-dX/r;
		acc[1]=-dY/r;
		acc[2]=-dZ/r;
	}


}

