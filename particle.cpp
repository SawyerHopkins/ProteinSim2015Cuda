/*The MIT License (MIT)

Copyright (c) <2015> <Sawyer Hopkins>

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.*/

#include "particle.h"

namespace simulation
{

	/********************************************//**
	*--------------SYSTEM CONSTRUCTION---------------
	************************************************/

	particle::particle(int pid)
	{
		name = pid;

		x = 0.0;
		y = 0.0;
		z = 0.0;
		
		x0 = 0.0;
		y0 = 0.0;
		z0 = 0.0;
		
		fx = 0.0;
		fy = 0.0;
		fz = 0.0;
		
		fx0 = 0.0;
		fy0 = 0.0;
		fz0 = 0.0;

		vx = 0.0;
		vy = 0.0;
		vz = 0.0;

		//For debugging.
		cx = -1;
		cy = -1;
		cz = -1;

		r = 0.0;
		m = 0.0;
		
	}

	particle::~particle()
	{
		delete &x;
		delete &y;
		delete &z;
		delete &x0;
		delete &y0;
		delete &z0;
		delete &fx;
		delete &fy;
		delete &fz;
		delete &fx0;
		delete &fy0;
		delete &fz0;
		delete &vx;
		delete &vy;
		delete &vz;
		delete &m;
		delete &r;
		delete &cx;
		delete &cy;
		delete &cz;
		delete &name;
	}

	/********************************************//**
	*---------------SYSTEM MANAGEMENT----------------
	************************************************/

	void particle::setX(double val, double boxSize)
	{
		double x0Temp = x0;
		double xTemp = x;
		x = utilities::util::safeMod(val, boxSize);
		x0 = utilities::util::safeMode0(xTemp,x,boxSize);
		if ((x < 0.0) || (x >= boxSize))
		{
			std::cout << "\nParticle: " << name << " out of bounds.\n";
			std::cout << x << "\n";
			std::cout << xTemp << " - " << x0Temp << "\n";
			std::cout << val << "\n";
			std::cout << fx << "\n";
			exit(100);
		}
	}

	void particle::setY(double val, double boxSize)
	{
		double y0Temp = y0;
		double yTemp = y;
		y = utilities::util::safeMod(val, boxSize);
		y0 = utilities::util::safeMode0(yTemp,y,boxSize);
		if ((y < 0.0) || (y >= boxSize))
		{
			std::cout << "\nParticle: " << name << " out of bounds.\n";
			std::cout << y << "\n";
			std::cout << yTemp << " - " << y0Temp << "\n";
			std::cout << val << "\n";
			std::cout << fy << "\n";
			exit(100);
		}
	}

	void particle::setZ(double val, double boxSize)
	{
		double z0Temp = z0;
		double zTemp = z;
		z = utilities::util::safeMod(val, boxSize);
		z0 = utilities::util::safeMode0(zTemp,z,boxSize);
		if ((z < 0.0) || (z >= boxSize))
		{
			std::cout << "\nParticle: " << name << " out of bounds.\n";
			std::cout << z << "\n";
			std::cout << zTemp << " - " << z0Temp << "\n";
			std::cout << val << "\n";
			std::cout << fz << "\n";
			exit(100);
		}
	}

	void particle::setPos(double xVal, double yVal, double zVal, double boxSize)
	{
		setX(xVal,boxSize);
		setY(yVal,boxSize);
		setZ(zVal,boxSize);
	}

	void particle::updateForce(double xVal, double yVal, double zVal)
	{
		fx += xVal;
		fy += yVal;
		fz += zVal;
	}

	void particle::clearForce()
	{
		fx0 = fx;
		fy0 = fy;
		fz0 = fz;
		fx = 0.0;
		fy = 0.0;
		fz = 0.0;
	}

}

