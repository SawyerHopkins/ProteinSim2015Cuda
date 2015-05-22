#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <string.h>
#include <float.h> 
#include <point.h>
#include "verlet.h"
#include "force.h"
#include <iostream>

using namespace std;

int main(int argc, char **argv)
{
	srand (time(NULL));

	integrators::verlet * difeq = new integrators::verlet(0.1);

	mathTools::points * pt = new mathTools::points(5);
	pt->init();

	for (int i = 0; i < pt->arr_size; i++)
	{
		cout << pt->getX(i) << "," << pt->getY(i) << "," << pt->getZ(i) << "\n";
	}

	for (int i =0; i < pt->arr_size; i++)
	{
		float pos[3] = {pt->getX(i), pt->getY(i), pt->getZ(i)};
		float vel[3] = {pt->getVX(i), pt->getVY(i), pt->getVZ(i)};
		difeq->nextPosition(i,pos,vel,pt,NULL);
	}

	printf("hello world\n");

	return 0;
}

float force(void)
{
	return 0.0;
}