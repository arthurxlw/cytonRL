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

#ifndef _CYTONRL_REINFORCELEARNER_H_
#define _CYTONRL_REINFORCELEARNER_H_

#include "NetworkRl.h"
#include "Environment.h"
#include "NumberProbes.h"
#include "NumberProbe.h"
#include "ReplayMemory.h"
#include "SumTree.h"

namespace cytonRl
{


class ReinforceLearner
{
public:
	NetworkRl network;
	NetworkRl networkTarget;

	Variable xBatch;
	Variable xSingle;
	Variable xTestSample;

	Environment env;
	int na;

	ReplayMemory replayMemTrain;
	ReplayMemory replayMemTest;
	SumTree sumTree;
	int episode;
	int episodeStep=0;
	int step;

	NumberProbe<Precision> probeQTrain;
	NumberProbe<Precision> probeQApply;

	HostMatPrec Q;
	HostMatPrec Q1;
	HostMatPrec Q1target;
	HostMatInt targetAction;
	HostMatPrec targetQ;
	HostMatPrec diffs;
	bool testMode;
	Precision randActRatio;

	XLLibTime startTime;
	void init();

	void train();
	void test(bool verbose);

	void work();
	void workTrain();
	void workTest();

protected:
	int getAction(ReplayMemory& replayMem, int frame, bool test);
	Precision learnOneBatch(ReplayMemory& mem);

	void initScreen(const string& screenSignalFile_);
	void setScreen();
	string screenSignalFile;

};

} /* namespace cytonLib */

#endif /* REINFORCELEARNER_H_ */
