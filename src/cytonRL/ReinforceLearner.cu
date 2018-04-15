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

#include "ReinforceLearner.h"
#include "ParamsRl.h"
#include "Global.h"
#include "WeightFactory.h"
#include "utils.h"

namespace cytonRl
{

void ReinforceLearner::init()
{
	env.init(0, 0);

	replayMemTrain.init(env.height, env.width, params.inputFrames, params.replayMemory);
	replayMemTest.init(env.height, env.width, params.inputFrames, params.testMaxEpisodeSteps);
	sumTree.init(params.replayMemory);

	xBatch.resize(params.batchSize, params.inputFrames, env.height, env.width, 2);
	xSingle.resize(1, params.inputFrames, env.height, env.width, 2);

	na=env.actions.size();
	network.init(&xBatch, na);
	networkTarget.init(&xBatch, na);
	networkTarget.weightFactory.whole.copyFrom(network.weightFactory.whole);

}



void ReinforceLearner::work()
{
	if(params.mode == "train")
	{
		workTrain();
	}
	else if(params.mode == "test")
	{
		workTest();
	}
	else
	{
		XLLib::printfln("unknown mode %s", params.mode.c_str());
		assert(false);
	}

}

void ReinforceLearner::workTrain()
{
	global.init();
	init();
	initScreen(params.saveModel+".screen");

	if(params.loadModel!="")
	{
		network.weightFactory.load(params.loadModel);
		XLLib::printfln(" loadModel %s", params.loadModel.c_str());
		networkTarget.weightFactory.whole.copyFrom(network.weightFactory.whole);
	}

	startTime=XLLib::startTime();
	ReplayMemory& memory=replayMemTrain;

	int frame=0;
	step=0;
	Step* s;
	XLLibTime startTime=XLLib::startTime();
	NumberProbe<Precision> probeEpiReward;
	bool firstProg=true;
	for(episode=0; ;episode++)	{

		if(step>=params.maxSteps)
		{
			XLLib::printfln("\n reach maxSteps %d %d", step, params.maxSteps);
			break;
		}

		if(	frame>=memory.capacity)
		{
			frame=0;
			XLLib::printfln("replayMemory restart");
		}

		env.reset();

		for(episodeStep=0;  ; episodeStep++)
		{
			if(step%1000==0)
			{
				setScreen();
			}

			if(step>=params.maxSteps)
			{
				XLLib::printfln("\n reach maxSteps %d %d", step, params.maxSteps);
				break;
			}

			if(episodeStep>=params.maxEpisodeSteps)
			{
				XLLib::printfln("\n reach maxEpisodeSteps %d %d", episodeStep, params.maxEpisodeSteps);
				env.reset(true);
				break;
			}

			step +=1;
			if(step%params.progPeriod==0)
			{
				if(firstProg)
				{
					XLLib::printfln("#FORMAT: Total Steps , Total Time(second), Total Episodes,");
					XLLib::printfln("      Reward average reward per episode ( number of episodes ),");
					XLLib::printfln("      QTrain average Q value of training ( number of QTrain samples), ");
					XLLib::printfln("      QApply average Q value of applying (number of QApply samples),");
					XLLib::printfln("      |grad| average norm of gradient (number of gradient samples)");
					firstProg=false;
				}
				XLLib::printf("# step %d, %s, episode %d, reward %s",
						step, XLLib::endTime(startTime).c_str(), episode,
						probeEpiReward.toString().c_str());
				XLLib::printf(", QTrain %s, QApply %s, |grad| %s\n",
										probeQTrain.toString(false).c_str(),
										probeQApply.toString(false).c_str(),
										network.weightFactory.probeNorms.toString(false).c_str()
										);
				probeEpiReward.reset();
				probeQTrain.reset();
				probeQApply.reset();
				network.weightFactory.probeNorms.reset();
			}

			if(step%params.savePeriod==0)
			{
				string tFile=XLLib::stringFormat("%s.%d", params.saveModel.c_str(), step);
				XLLib::printfln("saveModel %s", tFile.c_str());
				network.weightFactory.save(tFile);
			}

			if(step%params.testPeriod==0)
			{
				test(true);
			}

			env.getScreen();
			memory.addImage(frame, env.imageMat);
			sumTree.set(frame, 1.0);

			s=&memory.getStep(frame);
			s->action=getAction(memory, frame, false);
			Precision tReward=env.act(s->action);
			s->reward=std::max((Precision)-1.0, std::min((Precision)1.0, tReward));
			s->terminate=env.roundEnd();

			if( step>=params.learnStart && step%params.updatePeriod==0)
			{
				learnOneBatch(memory);
			}

			frame +=1;
			if(s->terminate || frame>=memory.capacity)
			{
				break;
			}
		}

		probeEpiReward.update(env.sumReward);
	}
}


void ReinforceLearner::test(bool verbose)
{
	XLLibTime startTime1=XLLib::startTime();
	ReplayMemory& memory=replayMemTest;
	int maxSteps=params.maxEpisodeSteps;
	assert(memory.capacity==maxSteps);
	if(verbose)
	{
		XLLib::printfln("start test:");
	}

	vector<double> rewards;
	vector<int> steps;
	for(episode=0; episode<params.testEpisodes ; episode++)
	{
		Precision epiReward=0;
		int step=0;
		for(int round=0; ; round++)
		{
			env.reset(round==0);
			for(int roundStep=0; ; roundStep++)
			{
				if(step%1000==0)
				{
					setScreen();
				}

				env.getScreen();
				memory.addImage(step, env.imageMat);

				Step* s=&memory.getStep(step);
				s->action=getAction(memory, step, true);
				Precision tReward=env.act(s->action);
				s->reward=std::max((Precision)-1.0, std::min((Precision)1.0, tReward));
				s->terminate=env.roundEnd();

				step +=1;
				if(s->terminate || step>=maxSteps)
				{
					break;
				}
			}

			epiReward+=env.sumReward;
			if(env.ale.game_over() || step>=maxSteps)
			{
				break;
			}
		}

		rewards.push_back(epiReward);
		steps.push_back(step);
		if(verbose)
		{
			XLLib::printfln("  epi %d, step %d, reward %.3f", episode, step, epiReward);
		}
	}

	vector<int> indexs;
	XLLib::sortIndex(rewards, indexs);
	double medianR=0;

	XLLib::printfln("sorted rewards:");
	for(int k=0; k<rewards.size(); k++)
	{
		int i=indexs.at(k);
		XLLib::printfln("  %d\t%d\t%.3f", k, steps.at(i), rewards.at(i));
	}

	int im=rewards.size()/2;
	if(rewards.size()%2==0)
	{
		medianR=(rewards.at(indexs.at(im))+rewards.at(indexs.at(im+1)))/2.0;
	}
	else
	{
		medianR=rewards.at(indexs.at(im));
	}

	double sumR=0;
	for(int i=0; i<rewards.size(); i++)
	{
		sumR +=rewards.at(i);
	}
	double avgR=sumR/rewards.size();

	XLLib::printfln("test %s %s, medianReward %.3f, avgReward %.3f", XLLib::endTime(startTime).c_str(),
			XLLib::endTime(startTime1).c_str(), medianR, avgR);

}

void ReinforceLearner::workTest()
{
	global.init();
	init();
	initScreen(params.loadModel+".screen");

	network.weightFactory.load(params.loadModel);
	XLLib::printfln("loadModel %s", params.loadModel.c_str());

	startTime=XLLib::startTime();
	XLLib::printfln("start test ...");
	test(true);

}


int ReinforceLearner::getAction(ReplayMemory& mem, int frame, bool test)
{

	double epsilon=0;
	if(!test)
	{
		int step1=this->step;
		double eStart=params.eGreedy.at(0);
		double eEnd=params.eGreedy.at(1);
		int N=(int)params.eGreedy.at(2);

		epsilon=eEnd;
		if(step1<N)
		{
			epsilon= eStart + (double)step1/N*(eEnd-eStart);
		}
	}
	else
	{
		epsilon=params.testEGreedy;
	}
	if(step%params.progPeriod==0 && test==false)
	{
		XLLib::printfln("train e-Greedy epsilon %.6f", epsilon);
	}
	double roll=(double)rand()/RAND_MAX;
	assert(roll>=0 && roll<=1);
	int na=env.actions.size();
	int res=-1;
	if(roll<epsilon)
	{
		res=rand()%na;
	}
	else
	{
		mem.getInput(frame, xSingle);

		network.forwardQ(&xSingle);
		Q.copyFrom(network.y);
		assert(Q.length()==na);

		Precision maxV=Q.at(0);
		vector<int> maxActions;
		maxActions.push_back(0);
		assert(Q.length()==na);
		for(int i=1; i<na; i++)
		{
			double tv=Q.at(i);
			if(tv>maxV)
			{
				maxV=tv;
				maxActions.clear();
				maxActions.push_back(i);
			}
			else if(tv==maxV)
			{
				maxActions.push_back(i);
			}
		}
		probeQApply.update(maxV);

		int ia=0;
		if(maxActions.size()>=2)
		{
			ia=rand()%maxActions.size();
		}
		res=maxActions.at(ia);
	}

	return res;

}

Precision ReinforceLearner::learnOneBatch( ReplayMemory& mem)
{
	if(step%params.targetQ==0)
	{
		networkTarget.weightFactory.whole.copyFrom(network.weightFactory.whole);
	}

	int n=params.batchSize;
	int c=params.inputFrames;
	int h=env.height;
	int w=env.width;
	assert(xBatch.ni==w && xBatch.nj==n*c*h);
	//sample
	assert(RAND_MAX>mem.len);
	vector<int> frames;
	vector<Precision> weights;
	Precision beta=(1.0 - params.priorityBeta)*( ((Precision)step)/params.maxSteps) + params.priorityBeta;
	if(step%params.progPeriod==0)
	{
		XLLib::printfln("prioritized replay beta %.6f", beta);
	}
	sumTree.sample(n, frames, mem.len, beta, weights);

	for(vector<int>::iterator it=frames.begin(); it!=frames.end(); it++)
	{
		int& frame=*it;
		assert(frame<mem.len);
		if(frame==mem.len-1)
		{
			frame-=1;
		}
	}
	assert(frames.size()==n);


	vector<Step*> steps;
	for(vector<int>::iterator it=frames.begin(); it!=frames.end(); it++)
	{
		steps.push_back(&mem.getStep(*it));
	}

	//targetQ
	for(int i=0; i<n; i++)
	{
		int frame1=frames.at(i)+1;
		DevMatPrec input1=xBatch.range(0, w, c*h*i, c*h*(i+1));
		if(!steps.at(i)->terminate)
		{
			mem.getInput(frame1, input1);
		}
		else
		{
			input1.setZero();
		}
	}

	networkTarget.forwardQ(&xBatch);
	Q1target.copyFrom(networkTarget.y);

	network.forwardQ(&xBatch);
	Q1.copyFrom(network.y);

	targetAction.resize(1,n);
	targetQ.resize(1, n);
	for(int i=0; i<n; i++)
	{
		double tv=0;
		int ti=-1;
		Step& s=*steps.at(i);
		if(!s.terminate)
		{
			ti=XLLib::iMax(&Q1.at(0, i), na);
			tv=Q1target.at(ti, i);
		}
		Precision newQ=s.reward+params.gamma*tv;

		targetAction.at(i)=s.action;
		targetQ.at(i)=newQ;
		probeQTrain.update(newQ);
	}

	//learn
	Precision lambda=params.learningRate;
	for(int i=0; i<n; i++)
	{
		int frame=frames.at(i);
		DevMatPrec input=xBatch.range(0, w, c*h*i, c*h*(i+1));
		mem.getInput(frame, input);
	}
	network.targetAction.copyFrom(targetAction);
	network.targetQ.copyFrom(targetQ);

	network.forwardQ(&xBatch);
	double trainScore=network.learnQ(lambda, diffs, weights);
	for(int k=0; k<frames.size(); k++)
	{
		Precision td=diffs.at(k);
		Precision tp=pow(abs(td)+0.01, params.priorityAlpha);
		sumTree.set(frames.at(k), tp);
	}

	return trainScore;
}

void ReinforceLearner::initScreen(const string& screenSignalFile_)
{
	screenSignalFile=screenSignalFile_;
	if(params.showScreen)
	{
		vector<string> lines;
		lines.push_back("");
		XLLib::writeFile(screenSignalFile,lines);
	}
	else
	{
		XLLib::fileRemove(screenSignalFile);
	}

	setScreen();
}

void ReinforceLearner::setScreen()
{
	bool screenOn=XLLib::fileExists(screenSignalFile);
	env.setScreen(screenOn);
}

} /* namespace cytonLib */
