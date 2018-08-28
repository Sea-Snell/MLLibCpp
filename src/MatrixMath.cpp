#include "MatrixMath.hpp"
#include "Node.hpp"
#include <limits>

MatMul::MatMul(Node* a, Node* b){
	inputs.push_back(a);
	inputs.push_back(b);
	a->outputs.push_back(this);
	b->outputs.push_back(this);
	a->outCount += 1;
	b->outCount += 1;
	name = "MatMul";
}

void MatMul::getDimentions(){
	if(getCount != 0){
		getCount = (getCount + 1) % outCount;
		return;
	}

	inputs[0]->getDimentions();
	inputs[1]->getDimentions();

	if (inputs[0]->resultDims.rank == 1 && inputs[1]->resultDims.rank == 1){
		resultDims.rank = 0;
		resultDims.size = 1;
		resultDims.dimentions = {};
	}
	else if (inputs[0]->resultDims.rank == 1){
		resultDims.rank = 1;
		resultDims.size = inputs[1]->resultDims.dimentions[1];
		resultDims.dimentions = {inputs[1]->resultDims.dimentions[1]};

		GROUP_SIZE = 16;
		globalSize = resultDims.size + (GROUP_SIZE - resultDims.size % GROUP_SIZE);
		if (resultDims.size % GROUP_SIZE == 0){
			globalSize -= GROUP_SIZE;
		}
	}
	else if (inputs[1]->resultDims.rank == 1){
		resultDims.rank = 1;
		resultDims.size = inputs[0]->resultDims.dimentions[0];
		resultDims.dimentions = {inputs[0]->resultDims.dimentions[0]};

		GROUP_SIZE = 16;
		globalSize = resultDims.size + (GROUP_SIZE - resultDims.size % GROUP_SIZE);
		if (resultDims.size % GROUP_SIZE == 0){
			globalSize -= GROUP_SIZE;
		}
	}
	else{
		resultDims.rank = 2;
		resultDims.size = inputs[0]->resultDims.dimentions[0] * inputs[1]->resultDims.dimentions[1];
		resultDims.dimentions = {inputs[0]->resultDims.dimentions[0], inputs[1]->resultDims.dimentions[1]};

		GROUP_SIZE = 1024;
		blockSide = 32;
		workPerBlockSideSquared = 16;
		heightSize = resultDims.dimentions[0] + (blockSide - resultDims.dimentions[0] % blockSide);
		if (resultDims.dimentions[0] % blockSide == 0){
			heightSize -= blockSide;
		}
		widthSize = resultDims.dimentions[1] + (blockSide - resultDims.dimentions[1] % blockSide);
		if (resultDims.dimentions[1] % blockSide == 0){
			widthSize -= blockSide;
		}
		blocksWide = widthSize / blockSide;
	}

	resultDims.setBuf();

	result = {};
	for (int i = 0; i < timeSteps; i++){
		result.push_back(cl::Buffer(context, CL_MEM_READ_WRITE, sizeof(float) * resultDims.size));
	}

	getCount = (getCount + 1) % outCount;
}

void MatMul::getValue(){
	if(getCount != 0){
		getCount = (getCount + 1) % outCount;
		if (getCount == 0){
			currentTime = (currentTime + 1) % timeSteps;
		}
		return;
	}
	inputs[0]->getValue();
	inputs[1]->getValue();
	
	if (inputs[0]->resultDims.rank == 1 && inputs[1]->resultDims.rank == 1){
		matMul1x1(cl::EnqueueArgs(queue, cl::NullRange, cl::NDRange(resultDims.size), cl::NullRange), inputs[0]->resultDims.dimBuf, inputs[0]->result[currentTime % inputs[0]->timeSteps], inputs[1]->result[currentTime % inputs[1]->timeSteps], result[currentTime]);
	}
	else if (inputs[0]->resultDims.rank == 1){
		matMul1x2(cl::EnqueueArgs(queue, cl::NullRange, cl::NDRange(globalSize), cl::NDRange(GROUP_SIZE)), inputs[0]->result[currentTime % inputs[0]->timeSteps], inputs[1]->resultDims.dimBuf, inputs[1]->result[currentTime % inputs[1]->timeSteps], result[currentTime]);
	}
	else if (inputs[1]->resultDims.rank == 1){
		matMul2x1(cl::EnqueueArgs(queue, cl::NullRange, cl::NDRange(globalSize), cl::NDRange(GROUP_SIZE)), inputs[0]->resultDims.dimBuf, inputs[0]->result[currentTime % inputs[0]->timeSteps], inputs[1]->result[currentTime % inputs[1]->timeSteps], result[currentTime]);
	}
	else{
		matMul2x2(cl::EnqueueArgs(queue, cl::NullRange, cl::NDRange((widthSize * heightSize) / workPerBlockSideSquared), cl::NDRange(GROUP_SIZE / workPerBlockSideSquared)), inputs[0]->resultDims.dimBuf, inputs[0]->result[currentTime % inputs[0]->timeSteps], inputs[1]->resultDims.dimBuf, inputs[1]->result[currentTime % inputs[1]->timeSteps], result[currentTime], blocksWide);
	}
	getCount = (getCount + 1) % outCount;
	if (getCount == 0){
		currentTime = (currentTime + 1) % timeSteps;
	}
}

void MatMul::deriveDimentions(GPUDimentions* tempSeed){
	getCount = (getCount + 1) % outCount;
	seedDimAdd(tempSeed);

	if (getCount == 0){
		outDims.push_back(inputs[0]->resultDims);
		outDims.push_back(inputs[1]->resultDims);
		out.push_back(cl::Buffer(context, CL_MEM_READ_WRITE, sizeof(float) * outDims[0].size));
		out.push_back(cl::Buffer(context, CL_MEM_READ_WRITE, sizeof(float) * outDims[1].size));

		if (inputs[0]->resultDims.rank == 1){
			GROUP_SIZE_DERIVATIVE_0 = 16;
			globalSizeDerivative0 = outDims[0].size + (GROUP_SIZE_DERIVATIVE_0 - outDims[0].size % GROUP_SIZE_DERIVATIVE_0);
			if (outDims[0].size % GROUP_SIZE_DERIVATIVE_0 == 0){
				globalSizeDerivative0 -= GROUP_SIZE_DERIVATIVE_0;
			}

			GROUP_SIZE_DERIVATIVE_1 = 1024;
			blockSideDerivative1 = 32;
			heightSizeDerivative1 = outDims[1].dimentions[0] + (blockSideDerivative1 - outDims[1].dimentions[0] % blockSideDerivative1);
			if (outDims[1].dimentions[0] % blockSideDerivative1 == 0){
				heightSizeDerivative1 -= blockSideDerivative1;
			}
			widthSizeDerivative1 = outDims[1].dimentions[1] + (blockSideDerivative1 - outDims[1].dimentions[1] % blockSideDerivative1);
			if (outDims[1].dimentions[1] % blockSideDerivative1 == 0){
				widthSizeDerivative1 -= blockSideDerivative1;
			}
			blocksWideDerivative1 = widthSizeDerivative1 / blockSideDerivative1;
		}
		else if (inputs[1]->resultDims.rank == 1){
			GROUP_SIZE_DERIVATIVE_0 = 1024;
			blockSideDerivative0 = 32;
			heightSizeDerivative0 = outDims[0].dimentions[0] + (blockSideDerivative0 - outDims[0].dimentions[0] % blockSideDerivative0);
			if (outDims[0].dimentions[0] % blockSideDerivative0 == 0){
				heightSizeDerivative0 -= blockSideDerivative0;
			}
			widthSizeDerivative0 = outDims[0].dimentions[1] + (blockSideDerivative0 - outDims[0].dimentions[1] % blockSideDerivative0);
			if (outDims[0].dimentions[1] % blockSideDerivative0 == 0){
				widthSizeDerivative0 -= blockSideDerivative0;
			}
			blocksWideDerivative0 = widthSizeDerivative0 / blockSideDerivative0;

			GROUP_SIZE_DERIVATIVE_1 = 16;
			globalSizeDerivative1 = outDims[1].size + (GROUP_SIZE_DERIVATIVE_1 - outDims[1].size % GROUP_SIZE_DERIVATIVE_1);
			if (outDims[1].size % GROUP_SIZE_DERIVATIVE_1 == 0){
				globalSizeDerivative1 -= GROUP_SIZE_DERIVATIVE_1;
			}
		}
		else if (inputs[0]->resultDims.rank == 2 && inputs[1]->resultDims.rank == 2){
			GROUP_SIZE_DERIVATIVE_0 = 1024;
			blockSideDerivative0 = 32;
			workPerBlockSideSquaredDerivative0 = 16;
			heightSizeDerivative0 = outDims[0].dimentions[0] + (blockSideDerivative0 - outDims[0].dimentions[0] % blockSideDerivative0);
			if (outDims[0].dimentions[0] % blockSideDerivative0 == 0){
				heightSizeDerivative0 -= blockSideDerivative0;
			}
			widthSizeDerivative0 = outDims[0].dimentions[1] + (blockSideDerivative0 - outDims[0].dimentions[1] % blockSideDerivative0);
			if (outDims[0].dimentions[1] % blockSideDerivative0 == 0){
				widthSizeDerivative0 -= blockSideDerivative0;
			}
			blocksWideDerivative0 = widthSizeDerivative0 / blockSideDerivative0;

			GROUP_SIZE_DERIVATIVE_1 = 1024;
			blockSideDerivative1 = 32;
			workPerBlockSideSquaredDerivative1 = 16;
			heightSizeDerivative1 = outDims[1].dimentions[0] + (blockSideDerivative1 - outDims[1].dimentions[0] % blockSideDerivative1);
			if (outDims[1].dimentions[0] % blockSideDerivative1 == 0){
				heightSizeDerivative1 -= blockSideDerivative1;
			}
			widthSizeDerivative1 = outDims[1].dimentions[1] + (blockSideDerivative1 - outDims[1].dimentions[1] % blockSideDerivative1);
			if (outDims[1].dimentions[1] % blockSideDerivative1 == 0){
				widthSizeDerivative1 -= blockSideDerivative1;
			}
			blocksWideDerivative1 = widthSizeDerivative1 / blockSideDerivative1;
		}

		inputs[0]->deriveDimentions(&outDims[0]);
		inputs[1]->deriveDimentions(&outDims[1]);
	}
}

void MatMul::derive(){
	getCount = (getCount + 1) % outCount;
	if (getCount == 0){
		int realTime = (timeSteps - 1) - currentTime;

		if (typeid(*inputs[0]) != typeid(Constant)){
			if (inputs[0]->getCount == 0 && (inputs[0]->timeSteps != 1 || (inputs[0]->timeSteps == 1 && currentTime == 0))){
				zeroBuffer(cl::EnqueueArgs(queue, cl::NullRange, cl::NDRange(inputs[0]->seedDims.size), cl::NullRange), inputs[0]->seed);
			}
			if (inputs[0]->resultDims.rank == 1 && inputs[1]->resultDims.rank == 1){
				matMul1x1Derivative(cl::EnqueueArgs(queue, cl::NullRange, cl::NDRange(outDims[0].size), cl::NullRange), inputs[1]->result[realTime % inputs[1]->timeSteps], seed, out[0]);
			}
			else if (inputs[0]->resultDims.rank == 1){
				matMul1x2Derivative0(cl::EnqueueArgs(queue, cl::NullRange, cl::NDRange(globalSizeDerivative0), cl::NDRange(GROUP_SIZE_DERIVATIVE_0)), inputs[1]->resultDims.dimBuf, inputs[1]->result[realTime % inputs[1]->timeSteps], seedDims.dimBuf, seed, out[0]);
			}
			else if (inputs[1]->resultDims.rank == 1){
				matMul2x1Derivative0(cl::EnqueueArgs(queue, cl::NullRange, cl::NDRange(widthSizeDerivative0 * heightSizeDerivative0), cl::NDRange(GROUP_SIZE_DERIVATIVE_0)), inputs[1]->result[realTime % inputs[1]->timeSteps], seedDims.dimBuf, seed, outDims[0].dimBuf, out[0], blocksWideDerivative0);
			}
			else{
				matMul2x2Derivative0(cl::EnqueueArgs(queue, cl::NullRange, cl::NDRange((widthSizeDerivative0 * heightSizeDerivative0) / workPerBlockSideSquaredDerivative0), cl::NDRange(GROUP_SIZE_DERIVATIVE_0 / workPerBlockSideSquaredDerivative0)), inputs[1]->resultDims.dimBuf, inputs[1]->result[realTime % inputs[1]->timeSteps], seedDims.dimBuf, seed, outDims[0].dimBuf, out[0], blocksWideDerivative0);
			}
			explodeUp(cl::EnqueueArgs(queue, cl::NullRange, cl::NDRange(inputs[0]->seedDims.size), cl::NullRange), outDims[0].dimBuf, out[0], inputs[0]->seedDims.dimBuf, inputs[0]->seed);
			inputs[0]->derive();
		}
		if (typeid(*inputs[1]) != typeid(Constant)){
			if (inputs[1]->getCount == 0 && (inputs[1]->timeSteps != 1 || (inputs[1]->timeSteps == 1 && currentTime == 0))){
				zeroBuffer(cl::EnqueueArgs(queue, cl::NullRange, cl::NDRange(inputs[1]->seedDims.size), cl::NullRange), inputs[1]->seed);
			}
			if (inputs[0]->resultDims.rank == 1 && inputs[1]->resultDims.rank == 1){
				matMul1x1Derivative(cl::EnqueueArgs(queue, cl::NullRange, cl::NDRange(outDims[1].size), cl::NullRange), inputs[0]->result[realTime % inputs[0]->timeSteps], seed, out[1]);
			}
			else if (inputs[0]->resultDims.rank == 1){
				matMul1x2Derivative1(cl::EnqueueArgs(queue, cl::NullRange, cl::NDRange(widthSizeDerivative1 * heightSizeDerivative1), cl::NDRange(GROUP_SIZE_DERIVATIVE_1)), inputs[0]->result[realTime % inputs[0]->timeSteps], seedDims.dimBuf, seed, outDims[1].dimBuf, out[1], blocksWideDerivative1);
			}
			else if (inputs[1]->resultDims.rank == 1){
				matMul2x1Derivative1(cl::EnqueueArgs(queue, cl::NullRange, cl::NDRange(globalSizeDerivative1), cl::NDRange(GROUP_SIZE_DERIVATIVE_1)), inputs[0]->resultDims.dimBuf, inputs[0]->result[realTime % inputs[0]->timeSteps], seedDims.dimBuf, seed, out[1]);
			}
			else{
				matMul2x2Derivative1(cl::EnqueueArgs(queue, cl::NullRange, cl::NDRange((widthSizeDerivative1 * heightSizeDerivative1) / workPerBlockSideSquaredDerivative1), cl::NDRange(GROUP_SIZE_DERIVATIVE_1 / workPerBlockSideSquaredDerivative1)), inputs[0]->resultDims.dimBuf, inputs[0]->result[realTime % inputs[0]->timeSteps], seedDims.dimBuf, seed, outDims[1].dimBuf, out[1], blocksWideDerivative1);
			}
			explodeUp(cl::EnqueueArgs(queue, cl::NullRange, cl::NDRange(inputs[1]->seedDims.size), cl::NullRange), outDims[1].dimBuf, out[1], inputs[1]->seedDims.dimBuf, inputs[1]->seed);
			inputs[1]->derive();
		}

		currentTime = (currentTime + 1) % timeSteps;
	}
}


Trans::Trans(Node* a){
	inputs.push_back(a);
	a->outputs.push_back(this);
	a->outCount += 1;
	name = "Trans";
}

void Trans::getDimentions(){
	if(getCount != 0){
		getCount = (getCount + 1) % outCount;
		return;
	}

	inputs[0]->getDimentions();

	resultDims.rank = inputs[0]->resultDims.rank;
	resultDims.size = inputs[0]->resultDims.size;
	resultDims.dimentions = {};
	resultDims.dimentions.push_back(inputs[0]->resultDims.dimentions[1]);
	resultDims.dimentions.push_back(inputs[0]->resultDims.dimentions[0]);

	resultDims.setBuf();

	result = {};
	for (int i = 0; i < timeSteps; i++){
		result.push_back(cl::Buffer(context, CL_MEM_READ_WRITE, sizeof(float) * resultDims.size));
	}

	getCount = (getCount + 1) % outCount;
}

void Trans::getValue(){
	if(getCount != 0){
		getCount = (getCount + 1) % outCount;
		if (getCount == 0){
			currentTime = (currentTime + 1) % timeSteps;
		}
		return;
	}
	inputs[0]->getValue();
	trans(cl::EnqueueArgs(queue, cl::NullRange, cl::NDRange(resultDims.size), cl::NullRange), inputs[0]->resultDims.dimBuf, inputs[0]->result[currentTime % inputs[0]->timeSteps], result[currentTime]);
	getCount = (getCount + 1) % outCount;
	if (getCount == 0){
		currentTime = (currentTime + 1) % timeSteps;
	}
}

void Trans::deriveDimentions(GPUDimentions* tempSeed){
	getCount = (getCount + 1) % outCount;
	seedDimAdd(tempSeed);

	if (getCount == 0){
		if (seedDims.rank == 0){
			outDims.push_back(seedDims);
		}
		else{
			outDims.push_back(inputs[0]->resultDims);
		}
		out.push_back(cl::Buffer(context, CL_MEM_READ_WRITE, sizeof(float) * outDims[0].size));
		inputs[0]->deriveDimentions(&outDims[0]);
	}
}

void Trans::derive(){
	getCount = (getCount + 1) % outCount;
	if (getCount == 0){
		if (typeid(*inputs[0]) != typeid(Constant)){
			if (inputs[0]->getCount == 0 && (inputs[0]->timeSteps != 1 || (inputs[0]->timeSteps == 1 && currentTime == 0))){
				zeroBuffer(cl::EnqueueArgs(queue, cl::NullRange, cl::NDRange(inputs[0]->seedDims.size), cl::NullRange), inputs[0]->seed);
			}
			if (seedDims.rank == 0){
				explodeUp(cl::EnqueueArgs(queue, cl::NullRange, cl::NDRange(inputs[0]->seedDims.size), cl::NullRange), seedDims.dimBuf, seed, inputs[0]->seedDims.dimBuf, inputs[0]->seed);
			}
			else if (seedDims.rank == 1){
				transDerivative1(cl::EnqueueArgs(queue, cl::NullRange, cl::NDRange(outDims[0].size), cl::NullRange), seed, outDims[0].dimBuf, out[0]);
				explodeUp(cl::EnqueueArgs(queue, cl::NullRange, cl::NDRange(inputs[0]->seedDims.size), cl::NullRange), outDims[0].dimBuf, out[0], inputs[0]->seedDims.dimBuf, inputs[0]->seed);
			}
			else{
				transDerivative2(cl::EnqueueArgs(queue, cl::NullRange, cl::NDRange(outDims[0].size), cl::NullRange), seedDims.dimBuf, seed, out[0]);
				explodeUp(cl::EnqueueArgs(queue, cl::NullRange, cl::NDRange(inputs[0]->seedDims.size), cl::NullRange), outDims[0].dimBuf, out[0], inputs[0]->seedDims.dimBuf, inputs[0]->seed);
			}
			inputs[0]->derive();
		}

		currentTime = (currentTime + 1) % timeSteps;
	}
}



Sum::Sum(Node* a, int dimentionVal){
	dimention = dimentionVal;
	inputs.push_back(a);
	a->outputs.push_back(this);
	a->outCount += 1;
	name = "Sum";
}

void Sum::getDimentions(){
	if(getCount != 0){
		getCount = (getCount + 1) % outCount;
		return;
	}

	inputs[0]->getDimentions();

	resultDims.rank = inputs[0]->resultDims.rank - 1;
	resultDims.size = inputs[0]->resultDims.size / inputs[0]->resultDims.dimentions[dimention];
	resultDims.dimentions = {};
	for (int i = 0; i < inputs[0]->resultDims.rank; i++){
		if (i != dimention){
			resultDims.dimentions.push_back(inputs[0]->resultDims.dimentions[i]);
		}
	}

	preSum = 1;
	for (int i = dimention + 1; i < inputs[0]->resultDims.rank; i++){
		preSum *= inputs[0]->resultDims.dimentions[i];
	}

	GROUP_SIZE = 128;
	int dimSize = inputs[0]->resultDims.dimentions[dimention] + (GROUP_SIZE - inputs[0]->resultDims.dimentions[dimention] % GROUP_SIZE);
	if (inputs[0]->resultDims.dimentions[dimention] % GROUP_SIZE == 0){
		dimSize -= GROUP_SIZE;
	}
	blocksWide = dimSize / GROUP_SIZE;
	globalSize = (inputs[0]->resultDims.size / inputs[0]->resultDims.dimentions[dimention]) * dimSize;

	resultDims.setBuf();

	result = {};
	resedue = {};
	for (int i = 0; i < timeSteps; i++){
		result.push_back(cl::Buffer(context, CL_MEM_READ_WRITE, sizeof(float) * resultDims.size));
		resedue.push_back(cl::Buffer(context, CL_MEM_READ_WRITE, sizeof(float) * ((inputs[0]->resultDims.size / inputs[0]->resultDims.dimentions[dimention]) * blocksWide)));
	}

	getCount = (getCount + 1) % outCount;
}

void Sum::getValue(){
	if(getCount != 0){
		getCount = (getCount + 1) % outCount;
		if (getCount == 0){
			currentTime = (currentTime + 1) % timeSteps;
		}
		return;
	}
	inputs[0]->getValue();
	sum_Pt1(cl::EnqueueArgs(queue, cl::NullRange, cl::NDRange(globalSize), cl::NDRange(GROUP_SIZE)), inputs[0]->result[currentTime % inputs[0]->timeSteps], resedue[currentTime], inputs[0]->resultDims.dimentions[dimention], preSum, blocksWide);
	sum_Pt2(cl::EnqueueArgs(queue, cl::NullRange, cl::NDRange(inputs[0]->resultDims.size / inputs[0]->resultDims.dimentions[dimention]), cl::NullRange), resedue[currentTime], result[currentTime], blocksWide);
	getCount = (getCount + 1) % outCount;
	if (getCount == 0){
		currentTime = (currentTime + 1) % timeSteps;
	}
}

void Sum::deriveDimentions(GPUDimentions* tempSeed){
	getCount = (getCount + 1) % outCount;
	seedDimAdd(tempSeed);

	if (getCount == 0){
		if (seedDims.rank < inputs[0]->resultDims.rank - dimention){
			outDims.push_back(seedDims);
		}
		else{
			GPUDimentions tempDims = GPUDimentions();
			tempDims.rank = seedDims.rank + 1;
			tempDims.size = seedDims.size * inputs[0]->resultDims.dimentions[dimention];
			tempDims.dimentions = {};
			for (int i = 0; i < tempDims.rank; i++){
				tempDims.dimentions.push_back(inputs[0]->resultDims.dimentions[inputs[0]->resultDims.rank - tempDims.rank + i]);
			}
			tempDims.setBuf();
			outDims.push_back(tempDims);
		}
		out.push_back(cl::Buffer(context, CL_MEM_READ_WRITE, sizeof(float) * outDims[0].size));
		inputs[0]->deriveDimentions(&outDims[0]);
	}
}

void Sum::derive(){
	getCount = (getCount + 1) % outCount;
	if (getCount == 0){
		if (typeid(*inputs[0]) != typeid(Constant)){
			if (inputs[0]->getCount == 0 && (inputs[0]->timeSteps != 1 || (inputs[0]->timeSteps == 1 && currentTime == 0))){
				zeroBuffer(cl::EnqueueArgs(queue, cl::NullRange, cl::NDRange(inputs[0]->seedDims.size), cl::NullRange), inputs[0]->seed);
			}
			if (seedDims.rank < inputs[0]->resultDims.rank - dimention){
				explodeUp(cl::EnqueueArgs(queue, cl::NullRange, cl::NDRange(inputs[0]->seedDims.size), cl::NullRange), seedDims.dimBuf, seed, inputs[0]->seedDims.dimBuf, inputs[0]->seed);
			}
			else{
				sumDerivative(cl::EnqueueArgs(queue, cl::NullRange, cl::NDRange(outDims[0].size), cl::NullRange), seedDims.dimBuf, seed, out[0], inputs[0]->resultDims.dimentions[dimention], preSum);
				explodeUp(cl::EnqueueArgs(queue, cl::NullRange, cl::NDRange(inputs[0]->seedDims.size), cl::NullRange), outDims[0].dimBuf, out[0], inputs[0]->seedDims.dimBuf, inputs[0]->seed);
			}
			inputs[0]->derive();
		}

		currentTime = (currentTime + 1) % timeSteps;
	}
}

string Sum::describe(){
	return name + "(" + inputs[0]->describe() + ", " + to_string(dimention) + ")";
}




void Mean::getValue(){
	if(getCount != 0){
		getCount = (getCount + 1) % outCount;
		if (getCount == 0){
			currentTime = (currentTime + 1) % timeSteps;
		}
		return;
	}
	inputs[0]->getValue();
	sum_Pt1(cl::EnqueueArgs(queue, cl::NullRange, cl::NDRange(globalSize), cl::NDRange(GROUP_SIZE)), inputs[0]->result[currentTime % inputs[0]->timeSteps], resedue[currentTime], inputs[0]->resultDims.dimentions[dimention], preSum, blocksWide);
	mean_Pt2(cl::EnqueueArgs(queue, cl::NullRange, cl::NDRange(inputs[0]->resultDims.size / inputs[0]->resultDims.dimentions[dimention]), cl::NullRange), resedue[currentTime], result[currentTime], inputs[0]->resultDims.dimentions[dimention], blocksWide);
	getCount = (getCount + 1) % outCount;
	if (getCount == 0){
		currentTime = (currentTime + 1) % timeSteps;
	}
}

void Mean::derive(){
	getCount = (getCount + 1) % outCount;
	if (getCount == 0){
		if (typeid(*inputs[0]) != typeid(Constant)){
			if (inputs[0]->getCount == 0 && (inputs[0]->timeSteps != 1 || (inputs[0]->timeSteps == 1 && currentTime == 0))){
				zeroBuffer(cl::EnqueueArgs(queue, cl::NullRange, cl::NDRange(inputs[0]->seedDims.size), cl::NullRange), inputs[0]->seed);
			}
			if (seedDims.rank < inputs[0]->resultDims.rank - dimention){
				meanDerivativeSmallSeed(cl::EnqueueArgs(queue, cl::NullRange, cl::NDRange(outDims[0].size), cl::NullRange), seed, out[0], inputs[0]->resultDims.dimentions[dimention]);
				explodeUp(cl::EnqueueArgs(queue, cl::NullRange, cl::NDRange(inputs[0]->seedDims.size), cl::NullRange), outDims[0].dimBuf, out[0], inputs[0]->seedDims.dimBuf, inputs[0]->seed);
			}
			else{
				meanDerivative(cl::EnqueueArgs(queue, cl::NullRange, cl::NDRange(outDims[0].size), cl::NullRange), seedDims.dimBuf, seed, out[0], inputs[0]->resultDims.dimentions[dimention], preSum);
				explodeUp(cl::EnqueueArgs(queue, cl::NullRange, cl::NDRange(inputs[0]->seedDims.size), cl::NullRange), outDims[0].dimBuf, out[0], inputs[0]->seedDims.dimBuf, inputs[0]->seed);
			}
			inputs[0]->derive();
		}

		currentTime = (currentTime + 1) % timeSteps;
	}
}



Max::Max(Node* a, int dimentionVal){
	inputs.push_back(a);
	a->outputs.push_back(this);
	a->outCount += 1;
	name = "Max";
	dimention = dimentionVal;
}

void Max::getDimentions(){
	if(getCount != 0){
		getCount = (getCount + 1) % outCount;
		return;
	}

	inputs[0]->getDimentions();

	resultDims.rank = inputs[0]->resultDims.rank - 1;
	resultDims.size = inputs[0]->resultDims.size / inputs[0]->resultDims.dimentions[dimention];
	resultDims.dimentions = {};
	for (int i = 0; i < inputs[0]->resultDims.rank; i++){
		if (i != dimention){
			resultDims.dimentions.push_back(inputs[0]->resultDims.dimentions[i]);
		}
	}

	preSum = 1;
	for (int i = dimention + 1; i < inputs[0]->resultDims.rank; i++){
		preSum *= inputs[0]->resultDims.dimentions[i];
	}

	resultDims.setBuf();

	result = {};
	idx = {};
	for (int i = 0; i < timeSteps; i++){
		result.push_back(cl::Buffer(context, CL_MEM_READ_WRITE, sizeof(float) * resultDims.size));
		idx.push_back(cl::Buffer(context, CL_MEM_READ_WRITE, sizeof(int) * inputs[0]->resultDims.size));
	}

	getCount = (getCount + 1) % outCount;
}

void Max::getValue(){
	if(getCount != 0){
		getCount = (getCount + 1) % outCount;
		if (getCount == 0){
			currentTime = (currentTime + 1) % timeSteps;
		}
		return;
	}
	inputs[0]->getValue();
	max_(cl::EnqueueArgs(queue, cl::NullRange, cl::NDRange(resultDims.size), cl::NullRange), inputs[0]->result[currentTime % inputs[0]->timeSteps], result[currentTime], idx[currentTime], inputs[0]->resultDims.dimentions[dimention], preSum);
	getCount = (getCount + 1) % outCount;
	if (getCount == 0){
		currentTime = (currentTime + 1) % timeSteps;
	}
}

void Max::deriveDimentions(GPUDimentions* tempSeed){
	getCount = (getCount + 1) % outCount;
	seedDimAdd(tempSeed);

	if (getCount == 0){
		outDims.push_back(inputs[0]->resultDims);
		out.push_back(cl::Buffer(context, CL_MEM_READ_WRITE, sizeof(float) * outDims[0].size));
		inputs[0]->deriveDimentions(&outDims[0]);
	}
}

void Max::derive(){
	getCount = (getCount + 1) % outCount;
	if (getCount == 0){
		int realTime = (timeSteps - 1) - currentTime;

		if (typeid(*inputs[0]) != typeid(Constant)){
			if (inputs[0]->getCount == 0 && (inputs[0]->timeSteps != 1 || (inputs[0]->timeSteps == 1 && currentTime == 0))){
				zeroBuffer(cl::EnqueueArgs(queue, cl::NullRange, cl::NDRange(inputs[0]->seedDims.size), cl::NullRange), inputs[0]->seed);
			}
			if (seedDims.rank < inputs[0]->resultDims.rank - dimention){
				maxDerivativeSmallSeed(cl::EnqueueArgs(queue, cl::NullRange, cl::NDRange(outDims[0].size), cl::NullRange), seedDims.dimBuf, seed, out[0], idx[realTime]);
				explodeUp(cl::EnqueueArgs(queue, cl::NullRange, cl::NDRange(inputs[0]->seedDims.size), cl::NullRange), outDims[0].dimBuf, out[0], inputs[0]->seedDims.dimBuf, inputs[0]->seed);
			}
			else{
				maxDerivative(cl::EnqueueArgs(queue, cl::NullRange, cl::NDRange(outDims[0].size), cl::NullRange), seedDims.dimBuf, seed, out[0], idx[realTime], inputs[0]->resultDims.dimentions[dimention], preSum);
				explodeUp(cl::EnqueueArgs(queue, cl::NullRange, cl::NDRange(inputs[0]->seedDims.size), cl::NullRange), outDims[0].dimBuf, out[0], inputs[0]->seedDims.dimBuf, inputs[0]->seed);
			}
			inputs[0]->derive();
		}

		currentTime = (currentTime + 1) % timeSteps;
	}
}

string Max::describe(){
	return name + "(" + inputs[0]->describe() + ", " + to_string(dimention) + ")";
}

void Min::getValue(){
	if(getCount != 0){
		getCount = (getCount + 1) % outCount;
		if (getCount == 0){
			currentTime = (currentTime + 1) % timeSteps;
		}
		return;
	}
	inputs[0]->getValue();
	min_(cl::EnqueueArgs(queue, cl::NullRange, cl::NDRange(resultDims.size), cl::NullRange), inputs[0]->result[currentTime % inputs[0]->timeSteps], result[currentTime], idx[currentTime], inputs[0]->resultDims.dimentions[dimention], preSum);
	getCount = (getCount + 1) % outCount;
	if (getCount == 0){
		currentTime = (currentTime + 1) % timeSteps;
	}
}


