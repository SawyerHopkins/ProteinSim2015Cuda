#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <string.h>
#include <float.h> 
#include <point.h>
#include <iostream>

using namespace std;

int main(int argc, char **argv)
{
	srand (time(NULL));
	mathTools::points * pt = new mathTools::points(5);
	pt->init();
	for (int i = 0; i < pt->arr_size; i++)
	{
		cout << pt->getX(i) << "," << pt->getY(i) << "," << pt->getZ(i) << "\n";
	}
	printf("hello world\n");

	return 0;
}

float force(void)
{
	return 0.0;
}