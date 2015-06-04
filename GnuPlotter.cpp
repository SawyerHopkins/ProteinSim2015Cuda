#include "GnuPlotter.h"


using namespace std;

namespace Plotting
{

	//Sends the system to GNUPlot.
	void GnuPlotter::plot(int size, double* x, double* y, double* z)
	{
		/*-----------------------------------------*/
		/*---------------SOURCE FROM---------------*/
		/*-----------------------------------------*/
		/*http://stackoverflow.com/questions/4445720/how-to-plot-graphs-in-gnuplot-in-real-time-in-c*/
		/*-----------------------------------------*/
		
		//Create a new PID.
		pid_t childPid=fork();
		if (childPid==0) 
		{
			//Write a temp file.
			writeFile(size, x, y ,z, "system");
			//Open GNUPlot session.
			FILE* pipe=popen("gnuplot -persist","w");
			//Tell GNUPlot to read CSV.
			fprintf(pipe, "set datafile separator \",\"\n");
			//Plot the temp file as scattered points.
			fprintf(pipe, "splot \"system.txt\" using 1:2:3 with points\n");
			//Close the session.
			fprintf(pipe,"quit\n");
			//Clean up the stream.
			fflush(pipe);
			fclose(pipe);
			exit(0);
		}
	}

	//Writes the system as CSV.
	void GnuPlotter::writeFile(int size, double* x, double* y, double* z, std::string name)
	{
		//Create a stream to the desired file.
		ofstream myFile;
		myFile.open(name + ".txt");
		//Write each point in the system as a line of csv formatted as: X,Y,Z
		for (int i = 0; i < size; i++)
		{
			myFile << *(x+i) << "," << *(y+i) << "," << *(z+i) << "\n";
		}
		//Close the stream.
		myFile.close();
	}

}

