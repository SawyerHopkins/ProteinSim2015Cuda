#ifndef UTILITIES_H
#define UTILITIES_H

namespace mathTools
{

	//Contains some useful math wrappers.
	class utilities
	{
		public:

			//fmod can't do negative numbers so use this.
			static float safeMod(float val, float base);

			//gets the distance between to points while
			//considering periodic boundary conditions.
			static float dist(float x1, float x2);

	};

}

#endif // UTILITIES_H
