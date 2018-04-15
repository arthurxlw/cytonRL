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

#ifndef _CYTONRL_PARAMSRL_H_
#define _CYTONRL_PARAMSRL_H_

#include "ParametersBase.h"
#include "basicHeadsRl.h"

namespace cytonRl
{

class ReinLearnParams: public xllib::ParametersBase
{
public:
	string mode;
	int replayMemory;
	int inputFrames;
	vector<int> networkSize;
	int batchSize;

	Precision gamma;
	Precision priorityAlpha;
	Precision priorityBeta;
	Precision learningRate;
	vector<double> eGreedy;
	string env;
	bool dueling;
	int maxSteps;
	int maxEpisodeSteps;
	int learnStart;
	int updatePeriod;
	int targetQ;
	int progPeriod;
	int eGreedyStartStep;
	string optimizer;
	bool showScreen;

	int testPeriod;
	double testEGreedy;
	int testEpisodes;
	int testMaxEpisodeSteps;

	string saveModel;
	string loadModel;
	int savePeriod;

	ReinLearnParams();
	void init_members();

};

extern ReinLearnParams params;

} /* namespace cytonRl */

#endif /* REINLEARNPARAMS_H_ */
