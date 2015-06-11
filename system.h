#ifndef SYSTEM_H
#define SYSTEM_H
#include <iostream>
#include <math.h>
#include <random>
#include "cell.h"
#include "particle.h"
#include "integrator.h"

namespace simulation
{

	class system
	{

		private:

			/*-----------------------------------------*/
			/*------------SYSTEM VARIABLES-------------*/
			/*-----------------------------------------*/

			//Information about the system.
			int nParticles;
			float concentration;
			int boxSize;
			int cellSize;
			float temp;

			//Random number seed.
			int seed;

			//System entities.
			cell **** cells;
			particle** particles;

			//System integrator.
			integrators::I_integrator* integrator;

			/*-----------------------------------------*/
			/*---------------SYSTEM INIT---------------*/
			/*-----------------------------------------*/

			//System setup.
			void initCells(int numCells, int scale);
			void initParticles();

			//System tools.
			void maxwellVelocityInit(std::mt19937* gen, std::uniform_real_distribution<double>* distribution);
			void initCheck(std::mt19937* gen, std::uniform_real_distribution<double>* distribution);

			/*-----------------------------------------*/
			/*-------------SYSTEM HANDLING-------------*/
			/*-----------------------------------------*/

			//Particle handling.
			void moveParticle(int index, float x, float y, float z);

		public:

			/*-----------------------------------------*/
			/*-----------SYSTEM CONSTRUCTION-----------*/
			/*-----------------------------------------*/

			//Constructor/Destructor
			system(int nPart, float conc, int scale, float r, float sysTemp, int rnd, integrators::I_integrator* sysInt);
			~system();

	};

}

#endif // SYSTEM_H
