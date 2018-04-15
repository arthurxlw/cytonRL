/*
Copyright 2018 XIAOLIN WANG (xiaolin.wang@nict.go.jp; arthur.xlw@gmail.com)

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

#ifndef _CYTONLIB_PARAMETERS_BASE_H_
#define _CYTONLIB_PARAMETERS_BASE_H_

#include "xlCLib.h"
#include <string>
#include <vector>
#include <map>
#include <sstream>
#include <iostream>
#include <iostream>
#include <string>
#include <fstream>
#include <vector>
#include <stack>
#include <sstream>
#include <assert.h>
#include <stdlib.h>
#include <algorithm>
#include <math.h>
#include <stdio.h>

namespace xllib{

class ParametersBase
{
protected:
	// struct
	struct Option
	{
		std::string	name;				// option name   (ex: -f --config)
		std::string	value;				// default value (ex: config.ini)
		std::string	help;				// help string   (ex: configuration files)
	};

	std::map<std::string, std::string> opt2val;
	std::vector<std::string> opts;
	std::string helps;
public:

	std::vector<std::string> params;
	bool pipeMode;
	std::ostream* os;

	ParametersBase()
	{
		const Option options[] =
		{
				{"help",  "default",	"explanation, [valid values]"},
				{"","",""}
		};

		addOptions(options);
	}

	virtual void init_members()
	{

	}

	virtual const char* get(const std::string& name)
	{
		std::string tName=std::string("--")+name;
		std::map<std::string, std::string>::iterator iter= opt2val.find(tName);
		const char* res=NULL;
		if(iter==opt2val.end())
		{
			fprintf(stderr, "getOption unkown name: %s\n", name.c_str());
			fprintf(stderr, "Option names: \n");
			for(iter=opt2val.begin(); iter!=opt2val.end(); iter++)
			{
				fprintf(stderr, "  %s\n", iter->first.c_str());
			}
			assert(false);
		}
		else
		{
			res=iter->second.c_str();
		}
		if(!pipeMode)
		{
			fprintf(stderr, "  %s:\t%s\n", name.c_str(), res);
		}
		else
		{
			XLLib::printfln(std::cerr, "  %s:\t%s", name.c_str(), res);
		}
		return res;
	}

	virtual int geti(const std::string& key)
	{
		return atoi(get(key));
	}

	virtual double getf(const std::string& key)
	{
		return atof(get(key));
	}
	virtual void parse( int argc, const char* const* argv )
	{
		if(argc>1)
		{
			std::string t=argv[1];
			pipeMode= (t=="--pipeMode");
			if(pipeMode)
			{
				std::cerr <<"pipeMode\n";
			}

		}
		os=pipeMode?&std::cerr:&std::cout;

		int iStart=!pipeMode?1:2;
		for( int i = iStart; i < argc; )
		{
			const std::string& arg = argv[i];
			if(arg=="--help")
			{
				std::cout<<helps<<"\n";
				exit(0);
			}
			else if(XLLib::stringStartsWith(arg,"-"))
			{
				if(i+1>=argc)
				{
					std::cout<<"No value for option: "<<arg<<"\n";
					assert(false);
				}
				const std::string& val=argv[i+1];

				if(arg=="--cmdConfig")
				{
					std::vector<std::string> tArgs=readCmdConfig(val);
					assert(tArgs.size()%2==0);
					for(int k=0;k<tArgs.size();k+=2)
					{
						const std::string tArg=tArgs.at(k);
						if(!XLLib::stringStartsWith(tArg,"-"))
						{
							std::cout<<"In CmdConfig: "<<arg<<", unkown option: "<<tArg<<"\n";
						}
						updateOptionValue(tArgs.at(k),tArgs.at(k+1));
					}
				}
				else
				{
					updateOptionValue(arg,val);
				}
				i+=2;
			}
			else
			{
				params.push_back(arg);
				i+=1;
			}
		}

		init_members();
	}

	void parseFile(const std::string& configFile)
	{
		std::vector<std::string> argList;
		std::vector<const char*> args;
		readArgs(configFile,argList,args);
		parse( args.size(), &(args[0]) );
	}


	void parseLine(const std::string& params_line)
	{
		std::vector<std::string> argList;
		std::vector<const char*> args;
		line2args(params_line,argList,args);
		parse( args.size(), &(args[0]) );
	}

protected:

	void addOptions(const Option* options)
	{
		std::ostringstream os;
		for( int i = 0;; i++ )
		{
			const Option& option = options[i];
			if( option.name == "" )
			{
				break;
			}
			std::string tName=std::string("--")+option.name;
			if(opt2val.find(tName)!=opt2val.end())
			{
				std::cout << "Duplicate entry for the option: " << option.name << std::endl;
				assert(false);
			}
			opt2val[tName] = option.value;
			os<<tName<<" :\t"<<option.help<<" ("<<option.value<<")"<<"\n";
		}
		helps=helps+os.str();
	}

	static void readArgs(const std::string& fileName,
			std::vector<std::string>& argList, std::vector<const char*>& args)
	{
		std::vector<std::string> lines;
		XLLib::readFile(fileName,lines);
		argList.clear();
		argList.push_back("TMP-EXE");
		for(std::vector<std::string>::iterator iter=lines.begin();iter!=lines.end();iter++)
		{
			std::vector<std::string> items;
			if(!XLLib::stringStartsWith(*iter,"#"))
			{
				XLLib::str2list(*iter,items);
				argList.insert(argList.end(),items.begin(),items.end());
			}
		}
		args.clear();
		for(std::vector<std::string>::iterator iter=argList.begin();iter!=argList.end();iter++)
		{
			const char* tc=(*iter).c_str();
			args.push_back(tc);
		}

	}

	static void line2args(const std::string& line,
			std::vector<std::string>& argList,
			std::vector<const char*>& args)
	{
		argList.clear();
		argList.push_back("TMP-EXE");
		std::vector<std::string> items;
		XLLib::str2list(line,items);
		argList.insert(argList.end(),items.begin(),items.end());

		args.clear();
		for(std::vector<std::string>::iterator iter=argList.begin();iter!=argList.end();iter++)
		{
			const char* tc=(*iter).c_str();
			args.push_back(tc);
		}

	}

	void updateOptionValue(const std::string& opt, const std::string& val)
	{
		std::map<std::string,std::string>::const_iterator it=opt2val.find(opt);
		if(it==opt2val.end()){
			std::cout<<"Unkown option: "<<opt<<"\n";
			std::cout<<"valid options are:\n";
			for(std::map<std::string, std::string>::const_iterator it=opt2val.begin();
					it!=opt2val.end(); it++)
			{
				std::cout<<"  "<<it->first<<" : "<< it->second<<"\n";
			}
			assert(false);
		}

		opt2val[opt]=val;
	}

	std::vector<std::string> readCmdConfig(const std::string& fileName)
	{
		std::vector<std::string> res;
		std::ifstream f( fileName.c_str() );
		if( !f.good() )
		{
			std::cout << "cannot open cmd config file: " << fileName << std::endl;
			assert( false );
		}

		std::string line;
		while(std::getline(f,line))
		{
			if(!XLLib::stringStartsWith(line,"#"))
			{
				std::vector<std::string> items;
				XLLib::str2list(line,items);
				res.insert(res.end(),items.begin(),items.end());
			}
		}
		return res;
	}

};

}
#endif /* PARAMS_H_ */
