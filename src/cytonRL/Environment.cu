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

#include "Environment.h"
#include "Global.h"
#include "opencv2/opencv.hpp"
#include "ParamsRl.h"
#include "utils.h"

namespace cytonRl
{

const float B_WEIGHT = 0.114;
const float G_WEIGHT = 0.587;
const float R_WEIGHT = 0.299;


void Environment::init(int idx, int randomSeed_)
{
	gameName=params.env;

	randomSeed=randomSeed_;
	// Get & Set the desired settings
	ale.setInt("random_seed", randomSeed);
	//The default is already 0.25, this is just an example
	ale.setFloat("repeat_action_probability", 0.0);
	ale.setInt("frame_skip", 1);
	screenOn=false;
	ale.setBool("display_screen", screenOn);
	ale.setBool("sound", false);

	// Load the ROM file. (Also resets the system for new settings to
	// take effect.)
	ale.loadROM(gameName);

	// Get the vector of legal actions
	//    ActionVect actions = ale.getLegalActionSet();
	actions=ale.getMinimalActionSet();

	const ALEScreen & aleScreen = ale.getScreen();
	vector<unsigned char> image;
	rawHeight=aleScreen.height();
	height=84;

	rawWidth=aleScreen.width();
	width=84;

	scaleH=(double)height/rawHeight;
	scaleW=(double)width/rawWidth;

	XLLib::printfln("screen raw %d %d , processed %d %d .", rawHeight, rawWidth, height, width);

	cvRaw.create(cv::Size(rawWidth, rawHeight), CV_32FC1);
	imageMat.resize(width, height);

	this->reset();
	initLives=ale.lives();

	//
	actDict.push_back("NOOP");//	   PLAYER_A_NOOP           = 0,
	actDict.push_back("FIRE");//	    PLAYER_A_FIRE           = 1,
	actDict.push_back("UP");//	    PLAYER_A_UP             = 2,
	actDict.push_back("RIGH");//	    PLAYER_A_RIGHT          = 3,
	actDict.push_back("LEFT");//	    PLAYER_A_LEFT           = 4,
	actDict.push_back("DOWN");//	    PLAYER_A_DOWN           = 5,
	actDict.push_back("UPRIGHT");//	    PLAYER_A_UPRIGHT        = 6,
	actDict.push_back("UPLEFT");//	    PLAYER_A_UPLEFT         = 7,
	actDict.push_back("DOWNRIGHT");//	    PLAYER_A_DOWNRIGHT      = 8,
	actDict.push_back("DOWNLEFT");//	    PLAYER_A_DOWNLEFT       = 9,
	actDict.push_back("UPFIRE");//	    PLAYER_A_UPFIRE         = 10,
	actDict.push_back("RIGHTFIRE");//	    PLAYER_A_RIGHTFIRE      = 11,
	actDict.push_back("LEFTFIRE");//	    PLAYER_A_LEFTFIRE       = 12,
	actDict.push_back("DOWNFIRE");//	    PLAYER_A_DOWNFIRE       = 13,
	actDict.push_back("UPRIGHTFIRE");//	    PLAYER_A_UPRIGHTFIRE    = 14,
	actDict.push_back("UPLEFTFIRE");//	    PLAYER_A_UPLEFTFIRE     = 15,
	actDict.push_back("DOWNRIGHTFIRE");//	    PLAYER_A_DOWNRIGHTFIRE  = 16,
	actDict.push_back("DOWNLEFTFIRE");//	    PLAYER_A_DOWNLEFTFIRE   = 17,


	for(int i=0; i<actions.size(); i++)
	{
		int act=actions.at(i);
		XLLib::printfln("action %d %s", i, actDict[act].c_str());
	}
}



void Environment::getScreen()
{
	ale.getScreenRGB(image);

	unsigned char* ptr=&image[0];
	for(int h=0; h<rawHeight; h+=1)
	{
		for(int w=0; w<rawWidth; w+=1)
		{
			Precision r=ptr[0];
			Precision g=ptr[1];
			Precision b=ptr[2];
			ptr += 3;

			Precision val=(r*R_WEIGHT+g*G_WEIGHT+b*B_WEIGHT)/255.0;
			cvRaw.at<float>(h,w)=val;
		}
	}

	cv::resize(cvRaw, cvImage, cv::Size(0, 0), scaleW, scaleH, cv::INTER_LINEAR);

	assert(cvImage.rows==height && cvImage.cols==width);

	uchar* tD=imageMat.data;
	for(int h=0; h<height; h++)
	{
		for(int w=0; w<width; w++)
		{
			*tD=(uchar)(cvImage.at<float>(h,w)*255.0);
			tD+=1;
		}
	}
	assert(tD-imageMat.data == imageMat.length());
}

void Environment::reset(bool resetGame)
{

	if(ale.game_over() || resetGame)
	{
		ale.reset_game();
	}
	int nr=rand()%30;
	Action action=actions.at(0);
	for(int i=0; i<nr; i++)
	{
		ale.act(action);
	}
	initLives=ale.lives();
	sumReward=0;
}


void Environment::setScreen(bool screenOn_)
{
	if(screenOn!=screenOn_)
	{
		screenOn=screenOn_;
		XLLib::printfln("environment.setScreen %d", screenOn);
		ale.setBool("display_screen", screenOn);
		ale.loadROM(gameName);
	}
}

bool Environment::roundEnd()
{
	bool res=ale.game_over() || ale.lives()!=initLives;
	return res;
}

float Environment::act(int act)
{
	Action action=actions.at(act);

	int repeat=4;
	float reward=0;

	for(int i=0; i<repeat; i++)
	{
		reward+=ale.act(action);
		if(ale.game_over() || ale.lives()!=initLives)
		{
			break;
		}
	}
	sumReward+=reward;

	return reward;
}

const char* Environment::getActName(int act)
{
	int action=actions.at(act);
	return actDict.at(action).c_str();
}
} /* namespace cytonLib */
