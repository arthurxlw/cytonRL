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

#ifndef _CYTONRL_ENVIRONMENT_H_
#define _CYTONRL_ENVIRONMENT_H_

#include "basicHeadsRl.h"
#include <ale/ale_interface.hpp>
#include "HostMatReal.h"
#include "DevMatReal.h"
#include "Variable.h"
#include "opencv2/opencv.hpp"

using namespace cytonLib;

namespace cytonRl
{

class Environment
{
public:

	ALEInterface ale;
	string gameName;
	int randomSeed;
	vector<Action> actions;
	float sumReward;
	int initLives;
	bool screenOn;

	vector<unsigned char> image;
	cv::Mat cvRaw;
	cv::Mat cvImage;

	HostMatUchar imageMat;

	int rawHeight;
	int rawWidth;

	int height;
	int width;

	double scaleH;
	double scaleW;

	vector<string> actDict;

	Environment()
	{
	}

	Environment(const Environment& other)
	{
	}

	void init(int idx=0, int randomSeed_=123);

	void getScreen();

	void reset(bool resetGame=false);

	void setScreen(bool screenOn_);

	bool roundEnd();

	float act(int act);

	const char* getActName(int act);

};

} /* namespace cytonLib */

#endif /* ENVIRONMENT_H_ */
