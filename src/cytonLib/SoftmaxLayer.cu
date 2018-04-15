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

#include "SoftmaxLayer.h"

namespace cytonLib {


Variable* SoftmaxLayer::init(string tag_, Variable* x_)
{
	tag=tag_;
	this->x=x_;
	y.resize(*x);
	return &y;
}


void SoftmaxLayer::forward()
{
	y.resize(*x);
	checkError(cudnnSoftmaxForward(global.cudnnHandle, CUDNN_SOFTMAX_ACCURATE,
			CUDNN_SOFTMAX_MODE_INSTANCE,
			&global.one, x->desc, x->data, &global.zero, y.desc, y.data));
}

void SoftmaxLayer::backward()
{
	if(!sparse)
	{
		checkError(cudnnSoftmaxBackward(global.cudnnHandle, CUDNN_SOFTMAX_ACCURATE,
				CUDNN_SOFTMAX_MODE_INSTANCE,
				&global.one, y.desc, y.data, y.desc, y.grad.data,
				&global.zero, x->desc, x->grad.data));
	}

}

__global__
void softmax_backwardSparse_kernel(int* targets, Precision* y, Precision* dx, Precision* likehood, Precision scale,
		int ni, int nj, bool ignore0)
{
	int j=blockIdx.y;
	int i=blockDim.x*blockIdx.x+threadIdx.x;

	if(i<ni)
	{
		int target=targets[j];

		int offset=i+ni*j;
		Precision* tY=y+offset;

		Precision tDx=0;
		if(target !=0 || !ignore0)
		{

			if( i == target){
				tDx=1-*tY;
				Precision tLog;
#ifdef PREICISION_DOUBLE
				tLog=log(*tY);
#else
				tLog=logf(*tY+1.0e-20);
#endif
				likehood[j]=tLog;
			}
			else
			{
				tDx=-*tY;
			}
		}
		else
		{
			tDx=0;
			if( i==target)
			{
				likehood[j]=0;
			}
		}

		if(dx!=NULL)
		{
			dx[offset]=tDx*scale;
		}

	}
}


Precision SoftmaxLayer::backwardSparse(int* target, Precision scale, bool ignore0)
{
	int ni=y.ni;
	int nj=y.nj;
	likehood.resize(1,nj);

	dim3 grid(ceil(ni, blockSize),nj);
	softmax_backwardSparse_kernel<<<grid, blockSize>>>(target, y.data, x->grad.data, likehood.data,
			scale, ni, nj, ignore0);
	checkError(cudaGetLastError());

	likehood_host.copyFrom(likehood);
	likehoodSum=likehood_host.sum();
	return likehoodSum;
}


__global__
void softmax_backwardSmooth_kernel(int* targets, Precision* y, Precision* dy, Precision* likehood, Precision scale,
		int ni, int nj, bool ignore0, Precision epsilon0, Precision epsilon1)
{
	int j=blockIdx.y;
	int i=blockDim.x*blockIdx.x+threadIdx.x;

	if(i<ni)
	{
		int target=targets[j];

		int offset=i+ni*j;
		Precision* tY=y+offset;

		Precision tDy= 1.0/(*tY+1.0e-20);
		if(target !=0 || !ignore0)
		{
			if( i == target){
				tDy *= epsilon1;
				Precision tLog;
#ifdef PREICISION_DOUBLE
				tLog=log(*tY);
#else
				tLog=logf(*tY+1.0e-20);
#endif
				likehood[j]=tLog;
			}
			else
			{
				tDy *= epsilon0;
			}
		}
		else
		{
			tDy=0;
			if( i==target)
			{
				likehood[j]=0;
			}
		}

		if(dy!=NULL)
		{
			dy[offset]=tDy*scale;
		}

	}
}


Precision SoftmaxLayer::backwardSmoothPrepare(int* target, Precision scale, bool ignore0, Precision epsilon)
{
	int ni=y.ni;
	int nj=y.nj;
	likehood.resize(1,nj);

	if(epsilon>0)
	{
		sparse=false;

		Precision epsilon0=epsilon/ni;
		Precision epsilon1=1-epsilon+epsilon0;

		dim3 grid(ceil(ni, blockSize),nj);
		softmax_backwardSmooth_kernel<<<grid, blockSize>>>(target, y.data, y.grad.data, likehood.data,
				scale, ni, nj, ignore0, epsilon0, epsilon1);
		checkError(cudaGetLastError());
	}
	else
	{
		sparse=true;

		dim3 grid(ceil(ni, blockSize),nj);
		softmax_backwardSparse_kernel<<<grid, blockSize>>>(target, y.data, x->grad.data, likehood.data,
				scale, ni, nj, ignore0);
		checkError(cudaGetLastError());
	}

	likehood_host.copyFrom(likehood);
	likehoodSum=likehood_host.sum();

	return likehoodSum;
}



} /* namespace cytonLib */
