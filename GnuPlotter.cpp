#include "GnuPlotter.h"


using namespace std;

namespace Plotting
{

	void GnuPlotter::plot(int size, float* x, float* y, float* z)
	{
		//http://stackoverflow.com/questions/4445720/how-to-plot-graphs-in-gnuplot-in-real-time-in-c
		pid_t childPid=fork();
		if (childPid==0) 
		{
			writeFile(size, x, y ,z, "system");
			FILE* pipe=popen("gnuplot -persist","w");
			fprintf(pipe, "set datafile separator \",\"\n");
			fprintf(pipe, "splot \"system.txt\" using 1:2:3 with points\n");
			fprintf(pipe,"quit\n");
			fflush(pipe);
			fclose(pipe);
			exit(0);
		}
	}

	void GnuPlotter::writeFile(int size, float* x, float* y, float* z, std::string name)
	{
		ofstream myFile;
		myFile.open(name + ".txt");
		for (int i = 0; i < size; i++)
		{
			myFile << *(x+i) << "," << *(y+i) << "," << *(z+i) << "\n";
		}
		myFile.close();
	}

}

