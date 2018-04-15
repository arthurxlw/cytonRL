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

#include "ParamsRl.h"
#include "Global.h"

namespace cytonRl
{

ReinLearnParams::ReinLearnParams()
{
	const Option options[] = {
			{"batchSize", "32", "the size of batch"},
			{"dueling","1", "using dueling DQN, 0|1"},
			{"eGreedy",  "1.0:0.01:5000000",	"e-Greedy  start_value : end_value : num_steps"},
			{"env", "roms/breakout.bin", "the rom of an Atari game"},
			{"gamma", "0.99", "parameter in the Bellman equation"},
			{"inputFrames",  "4",	"the number of concatnated frames as input"},
			{"learningRate",  "0.0000625",	"the learning rate"},
			{"learnStart",  "50000",	"the step of starting learning"},
			{"loadModel", "", "the file path for loading a model"},
			{"maxEpisodeSteps", "18000", "the maximun number of steps per episode"},
			{"maxSteps","100000000", "the maximun number of training steps"},
			{"mode", "train", "working mode, train|test"},
			{"networkSize", "32:64:64:512", "the ouput dimension of each hidden layer"},
			{"optimizer", "RMSprop", "the optimzier, SGD|RMSprop"},
			{"priorityAlpha",  "0.6",	"the alpha parameter of prioritized experience replay"},
			{"priorityBeta",  "0.4",	"the beta parameter of prioritized experience replay"},
			{"progPeriod",  "10000",	"the period of showing training progress"},
			{"replayMemory",  "1000000",	"the capacity of replay memory"},
			{"saveModel", "model/model", "the file path for saving models"},
			{"savePeriod", "1000000", "the period of saving models"},
			{"showScreen", "0", "whether show screens of playing Atari games, 0|1. Creating or deleting the model/model.screen file can change this setting during the running of the program. Note that hidding screens makes program run faster."},
			{"targetQ",  "30000",	"the period of copy the current network to the target network "},
			{"testEGreedy",  "0.001",	"the e-Greedy threshold of test"},
			{"testEpisodes", "100", "the number of test epidoes"},
			{"testMaxEpisodeSteps", "18000", "the maximun number of steps per test episode"},
			{"testPeriod",  "5000000",	"the period of test"},
			{"updatePeriod",  "4",	"the period of learn a batch"},
			{"","",""}
	};

	addOptions(options);
}

void ReinLearnParams::init_members()
{
	mode=get("mode");
	batchSize=geti("batchSize");
	dueling=geti("dueling");
	env=get("env");
	gamma=getf("gamma");
	inputFrames=geti("inputFrames");
	learningRate=getf("learningRate");
	learnStart=geti("learnStart");
	loadModel=get("loadModel");
	maxEpisodeSteps=geti("maxEpisodeSteps");
	maxSteps=geti("maxSteps");
	optimizer=get("optimizer");
	priorityAlpha=getf("priorityAlpha");
	priorityBeta=getf("priorityBeta");
	progPeriod=geti("progPeriod");
	replayMemory=geti("replayMemory");
	saveModel=get("saveModel");
	savePeriod=geti("savePeriod");
	showScreen=geti("showScreen");
	targetQ=geti("targetQ");
	testEGreedy=getf("testEGreedy");
	testEpisodes=geti("testEpisodes");
	testMaxEpisodeSteps=geti("testMaxEpisodeSteps");
	testPeriod=geti("testPeriod");
	updatePeriod=geti("updatePeriod");
	XLLib::str2ints(get("networkSize"), ":", networkSize);
	string t=get("eGreedy");
	XLLib::str2doubles(t, ":", eGreedy);

}


ReinLearnParams params;

} /* namespace cytonRl */
