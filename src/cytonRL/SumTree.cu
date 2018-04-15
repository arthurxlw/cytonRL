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

#include "SumTree.h"
#include "Global.h"
#include "ParamsRl.h"

namespace cytonRl
{

void SumTree::init(int capacity)
{
	data.resize(2*capacity-1, 0.0);
	n=capacity;
}

void SumTree::set(int dataIndex, double value)
{
	int index=dataIndex+n-1;
	data.at(index)=value;
	while(true)
	{
		int parent=(index-1)/2;
		data.at(parent)=data.at(2*parent+1)+data.at(2*parent+2);
		if(parent==0)
		{
			break;
		}
		index=parent;
	}
}

int SumTree::retrieve(double value_, int validLen)
{
	double value=value_;
	assert(value<data.at(0));
	int index=0;
	while(true)
	{
		if(value>data.at(index))
		{
			XLLib::printfln("error at sumTree.retrieve! %g %g %d %g", value_, value, index, data.at(index));
			assert(false);
			value=data.at(index);
		}


		if(index>=n-1)
		{
			break;
		}
		int left=2*index+1;
		double leftValue=data.at(left);
		if(value<=leftValue)
		{
			index=left;
		}
		else
		{
			value-=leftValue;
			index=left+1;
		}
	}
	int dataIndex=index-n+1;
	assert(dataIndex<validLen);
	assert(dataIndex<n);
	return dataIndex;

}

void SumTree::sample(int batchSize, vector<int>& dataIndexes, int validLen, Precision beta, vector<Precision>& weights)
{
	double total=data.at(0);
	double d=total/batchSize;
	weights.clear();
	Precision maxW=0;
	for(int i=0; i<batchSize; i++)
	{
		double tRand=rand();
		tRand/=RAND_MAX;
		double roll= tRand+i;

		double roll1=roll*d;
		int dataIndex=retrieve(roll1, validLen);
		Precision dataVal=data.at(dataIndex+n-1);
		Precision tw=pow(dataVal, -beta);
		weights.push_back(tw);
		maxW=std::max(maxW, tw);
		dataIndexes.push_back(dataIndex);
	}

	for(int i=0; i<weights.size(); i++)
	{
		weights.at(i) /= maxW;
	}
}

} /* namespace cytonRl */
