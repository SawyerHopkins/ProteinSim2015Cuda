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

using namespace std;

/********************************************//**
*-------------FUNCTION DECLARATIONS--------------
*------------------------------------------------
*------------see function for details------------
************************************************/

static inline void greeting();
void runScript();
void runAnalysis(string fileName);

/********************************************//**
*------------------MAIN PROGRAM------------------
************************************************/

/**
 * @brief The program entry point.
 * @param argc Not implemented.
 * @param argv Not implemented.
 * @return 
 */
int main(int argc, char **argv)
{

	//Program welcome.
	greeting();

	//Program flags.
	bool a = false;
	string aName = "";

	int i = 0;
	//Iterate across input arguments.
	while (i < argc)
	{
		//Flag for analysis mode.
		string str(argv[i]);
		if (str.compare("-a")==0)
		{
			a = true;
			//Make sure the next argument exists.
			if ((i+1) < argc)
			{
				//Name of file to analyze.
				i++;
				string file(argv[i]);
				aName = file;
			}
			else
			{
				debugging::error::throwInputError();
			}
		}
		i++;
	}

	//Choose which mode to run.
	if (a)
	{
		runAnalysis(aName);
	}
	else
	{
		runScript();
	}

	//Debug code 0 -> No Error:
	return 0;
}


/********************************************//**
*-----------------AUX FUNCTIONS------------------
************************************************/

/**
 * @brief Output the program name and information.
 */
void greeting()
{
	cout << "---Particle Simulator 2015---\n";
	cout << "----Sawyer Hopkins et al.----\n\n";
}