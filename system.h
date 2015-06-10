#ifndef SYSTEM_H
#define SYSTEM_H
#include <vector>
#include <iostream>
#include <math.h>
#include "cell.h"

namespace simulation
{

	class system
	{

	private:

		int nParticles;
		float concentration;
		int boxSize;
		std::vector<cell*> cells;

	public:
		system(int nPart, float conc, int scale, float r);
		~system();

	};

}

#endif // SYSTEM_H
