#ifndef CONFIG_H
#define CONFIG_H
#include <string>
#include <fstream>
#include <sstream>
#include <map>
#include <regex>

namespace configReader
{

	using namespace std;

	class config
	{

		private:

			map<string, string> options;

		public:

			//Header Version.
			const int version = 1;

			config(string fileName);
			~config();

			template<typename T> T getParam(string key);

			bool containsKey(string key) { return options.count(key); }
	};

}

#endif // CONFIG_H
