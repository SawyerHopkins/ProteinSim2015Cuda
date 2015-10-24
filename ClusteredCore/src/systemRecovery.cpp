/*The MIT License (MIT)

Copyright (c) [2015] [Sawyer Hopkins]

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.*/
#include "system.h"

namespace simulation
{

	/********************************************//**
	*-----------------SYSTEM RECOVERY----------------
	 ***********************************************/

	system* system::loadFromFile(configReader::config* cfg, std::string sysState, integrators::I_integrator* sysInt, physics::forces* sysFcs)
	{
		using namespace std;

		int bsize = cfg->getParam<int>("boxSize",0);

		vector<particle*> loadedparts;

		ifstream state;
		state.open(sysState, ios_base::in);

		for(std::string line; std::getline(state, line);)
		{
			istringstream data(line);

			float x, y, z;
			float x0, y0, z0;
			float fx, fy, fz;
			float fx0, fy0, fz0;

			data >> x >> y >> z;
			data >> x0 >> y0 >> z0;
			data >> fx >> fy >> fz;
			data >> fx0 >> fy0 >> fz0;

			particle* temp = new particle(loadedparts.size());
			temp->setPos(x0,y0,z0,bsize);
			temp->updateForce(fx0,fy0,fz0,0,NULL,false);
			temp->nextIter();
			temp->setPos(x,y,z,bsize);
			temp->updateForce(fx,fy,fz,0,NULL,false);

			loadedparts.push_back(temp);
		}

		system* oldSys = new system();
		oldSys->trialName = sysState + "-rewind";
		oldSys->nParticles = loadedparts.size();
		oldSys->concentration = cfg->getParam<int>("Concentration",0);
		oldSys->boxSize = bsize;
		oldSys->cellSize = cfg->getParam<int>("cellSize",0);
		oldSys->cellScale = cfg->getParam<int>("cellScale",0);
		oldSys->temp = cfg->getParam<int>("temp",0);
		oldSys->currentTime = 0;
		oldSys->dTime = cfg->getParam<int>("dTime",0);
		oldSys->outputFreq = cfg->getParam<int>("outputFreq",0);
		oldSys->outXYZ = cfg->getParam<int>("outXYZ",0);
		oldSys->cycleHour = cfg->getParam<int>("cycleHour",0);
		oldSys->seed = cfg->getParam<int>("seed",0);

		oldSys->integrator = sysInt;
		oldSys->sysForces = sysFcs;

		oldSys->particles = loadedparts.data();

		oldSys->initCells(oldSys->cellScale);
		oldSys->writeSystemInit();
		return oldSys;
	}

}