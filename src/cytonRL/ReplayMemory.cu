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

#include "ReplayMemory.h"
#include "utils.h"
#include "ParamsRl.h"

namespace cytonRl
{

void ReplayMemory::init(int height_, int width_, int inputFrames_, int capacity_)
{
	height=height_;
	width=width_;
	inputFrames=inputFrames_;
	capacity=capacity_;

	buffer.resize(width*height,capacity);
	steps.resize(capacity);

	len=0;
}


void ReplayMemory::addImage(int frame, HostMatUchar& mat)
{
	assert(frame<capacity);
	len=std::max(len, frame+1);

	DevMatUchar tImage=buffer.range(0, -1, frame, frame+1);
	tImage.reshape(width, height);
	tImage.copyFrom(mat);

}

void ReplayMemory::getInput(int frame, DevMatPrec& mat)
{
	assert(frame<len);
	assert(buffer.ni==height*width);
	assert(mat.length()==height*width*inputFrames);

	int frameStart0=std::max(frame-inputFrames+1, 0);
	int frameStart=frameStart0;
	for(int tFrame=frame-1; tFrame>=frameStart0; tFrame--)
	{
		Step& s=getStep(tFrame);
		if(s.terminate)
		{
			frameStart=tFrame+1;
			break;
		}

	}

	DevMatUchar tInput=buffer.range(0, height*width, frameStart, frame+1);
	int kStart=frameStart - (frame-inputFrames+1);
	if(kStart==0)
	{
		mat.convertFrom(tInput, 1.0/255);
	}
	else
	{
		mat.reshape(height*width, mat.length()/(height*width));
		mat.range(0, -1, 0, kStart ).setZero();
		mat.range(0, -1, kStart, inputFrames).convertFrom(tInput, 1.0/255);
	}
}

void ReplayMemory::getImage(int frame, DevMatPrec& mat)
{
	DevMatUchar tImage=buffer.range(0, -1, frame, frame+1);
	mat.convertFrom(tImage, 1.0/255);
}

Step& ReplayMemory::getStep(int frame)
{
	if(frame>=this->len)
	{
		XLLib::printfln("replayMemory.getStep fail, %d %d", this->len, frame);
	}
	assert(frame<this->len);
	return steps.at(frame);
}

} /* namespace cytonRl */
