#ifndef FORCE_H
#define FORCE_H
#include <vector>
#include <cmath>
#include <random>
#include "point.h"
#include "utilities.h"

namespace physics
{

/*-----------------------------------------*/
/*-------------FORCE INTERFACE-------------*/
/*-----------------------------------------*/

	//A generic force container.
	class IForce
	{
		public:
			//virtual methods for forces of various parameters.
			virtual void getAcceleration(int index, float time, mathTools::points* pts, float (&acc)[3])=0;
			//Mark if the force is time dependent.
			virtual bool isTimeDependent()=0;
	};

/*-----------------------------------------*/
/*----------INTEGRATOR MANAGEMENT----------*/
/*-----------------------------------------*/

	//Management system for a collection of forces.
	class forces
	{
		private:

			//A vector of all forces in the system.
			std::vector<IForce*> flist;
			//Flagged if flist contains a time dependant force.
			bool timeDependent;

		public:

			//Constructor/Destructor
			forces();
			~forces();

			//Adds a force to the  stack
			void addForce(IForce* f);

			//Calculates the total acceleration
			void getAcceleration(float index, float time, mathTools::points* pts, float (&acc)[3]);
			bool isTimeDependent() { return timeDependent; }
	};

/*-----------------------------------------*/
/*------------LINEAR DRAG FORCE------------*/
/*-----------------------------------------*/

	//Drag force.
	class dragForce : public IForce
	{

		private:

			float gamma;

		public:

			//Constructor/Destructor
			dragForce(float coeff) { gamma = coeff; }
			~dragForce() { delete[] &gamma; };

			//Evaluates the force.
			void getAcceleration(int index, float time, mathTools::points* pts, float (&acc)[3]);
			bool isTimeDependent() { return false; }
	};

/*-----------------------------------------*/
/*-------------AGGREGATE FORCE-------------*/
/*-----------------------------------------*/

	//Drag force.
	class aggForce : public IForce
	{

		private:

			float gamma;
			float cutOff;
			float coEff1;
			float coEff2;

		public:

			//Constructor/Destructor
			aggForce(float coeff, float cut);
			~aggForce() { delete[] &gamma; };

			//Evaluates the force.
			void getAcceleration(int index, float time, mathTools::points* pts, float (&acc)[3]);
			bool isTimeDependent() { return false; }
	};

/*-----------------------------------------*/
/*-------------BROWNIAN FORCE--------------*/
/*-----------------------------------------*/

	//Drag force.
	class brownianForce : public IForce
	{

		private:

			float gamma;
			float sigma;

			float sig1;
			float sig2;
			float corr;
			float rc12;
			float c0;

			float * memX;
			float * memY;
			float * memZ;

			float * memCorrX;
			float * memCorrY;
			float * memCorrZ;

			int memSize;

			std::mt19937* gen;
			std::uniform_real_distribution<double>* distribution;

		public:

			//Constructor/Destructor
			brownianForce(float coEff, float stDev, float t_initial, float dt, int size);
			~brownianForce() { delete[] &gamma; };

			void init(float dt, float t_initial);

			//Evaluates the force.
			void getAcceleration(int index, float time, mathTools::points* pts, float (&acc)[3]);
			bool isTimeDependent() { return false; }
	};

}

#endif // FORCE_H
