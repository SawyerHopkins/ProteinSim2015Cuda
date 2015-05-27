#include <math.h>
#include "utilities.h"

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

}

