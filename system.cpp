#include "System.h"

namespace simulation
{

	System::System(int nPart, float conc, int scale, float r)
	{

		//Set the number of particles
		nParticles = nPart;

		//Create a box based on desired concentration.
		double vP = nPart*(4.0/3.0)*atan(1)*4*r*r*r;
		boxSize = (int) cbrt(vP / conc);

		//Sets the actual concentration.
		concentration = vP/pow(boxSize,3.0);

		std::cout << "\nSystem concentration: " << concentration << "\n";

		//Calculates the number of cells needed.
		int cellSize = boxSize / scale;
		int numCells = pow(cellSize,3.0);

		//Create the needed cells.
		for (int i=0; i<numCells; i++)
		{
			cells.push_back(new cell);
		}

	}

	System::~System()
	{
	}


}

