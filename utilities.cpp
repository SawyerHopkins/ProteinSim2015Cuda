#include <math.h>
#include "utilities.h"

namespace mathTools
{

	utilities::utilities()
	{
	}

	utilities::~utilities()
	{
	}

	float utilities::safeMod(float val, float base)
	{
		float output;
		if (val == 0)
		{
			return 0.0;
		}
		else if (val < 0)
		{
			output = fmod(fabs(val),base);
			return (base-output);
		}
		else
		{
			return fmod(val,base);
		}
	}

}

