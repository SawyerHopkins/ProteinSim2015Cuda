#ifndef UTILITIES_H
#define UTILITIES_H

namespace mathTools
{

	class utilities
	{
		public:
			utilities();
			~utilities();

			static float safeMod(float val, float base);
			static float dist(float x1, float x2);

	};

}

#endif // UTILITIES_H
