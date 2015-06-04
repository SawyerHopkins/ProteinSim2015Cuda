#include "force.h"

namespace physics
{

	//Releases the memory blocks
	brownianForce::~brownianForce()
	{
		delete[] &gamma;
		delete[] &sigma;

		delete[] &sig1;
		delete[] &sig2;
		delete[] &corr;
		delete[] &rc12;
		delete[] &c0;

		delete[] memX;
		delete[] memY;
		delete[] memZ;

		delete[] memCorrX;
		delete[] memCorrY;
		delete[] memCorrZ;

		delete[] &memSize;

		delete[]  gen;
		delete[]  distribution;
	}

	//Creates the brownian motion force.
	brownianForce::brownianForce(float coEff, float stDev, float t_initial, float dt, int size) :
	memX(new float[size]), memY(new float[size]), memZ(new float[size]),
	memCorrX(new float[size]), memCorrY(new float[size]), memCorrZ(new float[size])
	{
		//Set vital variables.
		gamma = coEff;
		sigma = stDev;
		memSize = size;

		//Set secondary variables.
		init(dt,t_initial);

		//Creates random guassian number generator for the brownian dynamics.
		std::random_device rd;
		gen = new std::mt19937(rd());
		distribution = new std::normal_distribution<double>(0.0,sigma);

	}

	//Sets secondary variables.
	//Copied from the old code.
	//How does this work?
	void brownianForce::init(float dt, float t_initial)
	{
		float y = gamma*dt;
		double y1,y2,y3,y4,y5,y6,y7,y8,y9;
		float ty = 2*y;
		float cpn = 0.0;
		float cmn = 0.0;
		float gn = 0.0;

		if (y != 0)
		{
			cpn = 2.0*y-3.0+4.0*exp(-y)-exp(-ty);
			cmn = -2.0*y-3.0+4.0*exp( y)-exp(ty);
			gn  = exp(y)-ty-exp(-y);
			c0 = exp(-y);
		}

		if (y < 0.05)
		{
			y1 = y;
			y2 = y1*y1;
			y3 = y2*y1;
			y4 = y3*y1;
			y5 = y4*y1;
			y6 = y5*y1;
			y7 = y6*y1;
			y8 = y7*y1;
			y9 = y8*y1;

			cpn = +(2.0/3.0)*y3-0.5*y4+(7.0/30.0)*y5-(1.0/12.0)
				*y6+(31.0/1260.0)*y7
				-(1.0/160.0)*y8+(127.0/90720.0)*y9;
			cmn = -(2.0/3.0)*y3-0.5*y4-(7.0/30.)*y5-(1.0/12.0)
				*y6-(31.0/1260.0)*y7
				-(1.0/160.0)*y8-(127.0/90720.0)*y9;
			gn = (1.0/3.0)*y3+(1.0/60.0)*y5;
		}

		if (y==0)
		{
			c0=1.0;
		}

		float gammaSq = gamma*gamma;

		sig1 = std::sqrt( (t_initial*cpn)/gammaSq );
		sig2 = std::sqrt( (-t_initial*cmn)/gammaSq );
		corr = (t_initial/gammaSq) * (gn/(sig1*sig2));
		rc12 = sqrt(1-(corr*corr));

	}

	//Get the acceleration from the Coloumb potential.
	void brownianForce::getAcceleration(int index, float time, mathTools::points* pts, float (&acc)[3])
	{

		//Gets the correlated part of the force from the previous random kick.
		if (time > 0)
		{
			*(memCorrX+index)=sig2 * ((corr * *(memX+index)) + (rc12 * *(memCorrX+index)));
			*(memCorrY+index)=sig2 * ((corr * *(memY+index)) + (rc12 * *(memCorrY+index)));
			*(memCorrZ+index)=sig2 * ((corr * *(memZ+index)) + (rc12 * *(memCorrZ+index)));
		}
		else
		{
			//If t=0 there was no previous kick.
			*(memCorrX+index)=0.0;
			*(memCorrY+index)=0.0;
			*(memCorrZ+index)=0.0;
		}

		//Remember this brownian kick for the next iteration.
		*(memX+index) = (*distribution)(*gen);
		*(memY+index) = (*distribution)(*gen);
		*(memZ+index) = (*distribution)(*gen);

		//Put these two parts of the force together with the appropriate coefficents.
		//See Pathria chapter 15 and/or GUNSTEREN & BERENDSEN 1981.
		acc[0] = (sig1 * *(memX+index)) + (c0 * *(memCorrX+index));

	}
}