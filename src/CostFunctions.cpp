#include "CostFunctions.hpp"

CostFunction::CostFunction(Node* hypothesis, Node* y, int dimentionVal = 0){
	dimention = dimentionVal;
	inputs.push_back(hypothesis);
	inputs.push_back(y);
	hypothesis->outputs.push_back(this);
	y->outputs.push_back(this);
	hypothesis->outCount += 1;
	y->outCount += 1;
}

void CostFunction::getDimentions(){
	if(getCount != 0){
		getCount = (getCount + 1) % outCount;
		return;
	}

	inputs[0]->getDimentions();
	inputs[1]->getDimentions();

	resultDims.rank = 0;
	resultDims.size = 1;
	resultDims.dimentions = {};

	resultDims.setBuf();

	GROUP_SIZE = 128;
	numGroups = inputs[0]->resultDims.size / GROUP_SIZE;
	if (numGroups * GROUP_SIZE != inputs[0]->resultDims.size){
		numGroups += 1;
	}
	globalSize = inputs[0]->resultDims.size + (GROUP_SIZE - inputs[0]->resultDims.size % GROUP_SIZE);
	if (inputs[0]->resultDims.size % GROUP_SIZE == 0){
		globalSize -= GROUP_SIZE;
	}

	result = cl::Buffer(context, CL_MEM_READ_WRITE, sizeof(float) * resultDims.size);
	resedue = cl::Buffer(context, CL_MEM_READ_WRITE, sizeof(float) * numGroups);
	getCount = (getCount + 1) % outCount;
}

void CostFunction::deriveDimentions(GPUDimentions* tempSeed){
	getCount = (getCount + 1) % outCount;
	seedDimAdd(tempSeed);

	if (getCount == 0){
		outDims.push_back(inputs[0]->resultDims);
		out.push_back(cl::Buffer(context, CL_MEM_READ_WRITE, sizeof(float) * outDims[0].size));
		inputs[0]->deriveDimentions(&outDims[0]);
	}
}

string CostFunction::describe(){
	return name + "(" + inputs[0]->describe() + ", " + inputs[1]->describe() + ", " + to_string(dimention) + ")";
}








void MeanSquared::getDimentions(){
	if(getCount != 0){
		getCount = (getCount + 1) % outCount;
		return;
	}

	inputs[0]->getDimentions();
	inputs[1]->getDimentions();

	resultDims.rank = 0;
	resultDims.size = 1;
	resultDims.dimentions = {};

	resultDims.setBuf();

	GROUP_SIZE = 128;
	numGroups = inputs[0]->resultDims.size / GROUP_SIZE;
	if (numGroups * GROUP_SIZE != inputs[0]->resultDims.size){
		numGroups += 1;
	}

	globalSize = inputs[0]->resultDims.size + (GROUP_SIZE - inputs[0]->resultDims.size % GROUP_SIZE);
	if (inputs[0]->resultDims.size % GROUP_SIZE == 0){
		globalSize -= GROUP_SIZE;
	}

	result = cl::Buffer(context, CL_MEM_READ_WRITE, sizeof(float) * resultDims.size);
	differenceMemo = cl::Buffer(context, CL_MEM_READ_WRITE, sizeof(float) * inputs[0]->resultDims.size);
	resedue = cl::Buffer(context, CL_MEM_READ_WRITE, sizeof(float) * numGroups);
	getCount = (getCount + 1) % outCount;
}

void MeanSquared::getValue(){
	if(getCount != 0){
		getCount = (getCount + 1) % outCount;
		return;
	}
	inputs[0]->getValue();
	meanSquaredPt1(cl::EnqueueArgs(queue, cl::NullRange, cl::NDRange(globalSize), cl::NDRange(GROUP_SIZE)), inputs[0]->resultDims.dimBuf, inputs[0]->result, inputs[1]->result, differenceMemo, resedue);
	meanFullResedue(cl::EnqueueArgs(queue, cl::NullRange, cl::NDRange(resultDims.size), cl::NullRange), resedue, result, inputs[0]->resultDims.dimentions[dimention], numGroups);
	getCount = (getCount + 1) % outCount;
}

void MeanSquared::derive(){
	getCount = (getCount + 1) % outCount;
	if (getCount == 0){
		if (typeid(*inputs[0]) != typeid(Constant)){
			if (inputs[0]->getCount == 0){
				zeroBuffer(cl::EnqueueArgs(queue, cl::NullRange, cl::NDRange(inputs[0]->seedDims.size), cl::NullRange), inputs[0]->seed);
			}
			meanSquaredDerivative(cl::EnqueueArgs(queue, cl::NullRange, cl::NDRange(outDims[0].size), cl::NullRange), seed, differenceMemo, out[0], inputs[0]->resultDims.dimentions[dimention]);
			explodeUp(cl::EnqueueArgs(queue, cl::NullRange, cl::NDRange(inputs[0]->seedDims.size), cl::NullRange), outDims[0].dimBuf, out[0], inputs[0]->seedDims.dimBuf, inputs[0]->seed);
			inputs[0]->derive();
		}
	}
}






void CrossEntropy::getValue(){
	if(getCount != 0){
		getCount = (getCount + 1) % outCount;
		return;
	}
	inputs[0]->getValue();
	crossEntropyPt1(cl::EnqueueArgs(queue, cl::NullRange, cl::NDRange(globalSize), cl::NDRange(GROUP_SIZE)), inputs[0]->resultDims.dimBuf, inputs[0]->result, inputs[1]->result, resedue);
	meanFullResedue(cl::EnqueueArgs(queue, cl::NullRange, cl::NDRange(resultDims.size), cl::NullRange), resedue, result, inputs[0]->resultDims.dimentions[dimention], numGroups);
	getCount = (getCount + 1) % outCount;
}

void CrossEntropy::derive(){
	getCount = (getCount + 1) % outCount;
	if (getCount == 0){
		if (typeid(*inputs[0]) != typeid(Constant)){
			if (inputs[0]->getCount == 0){
				zeroBuffer(cl::EnqueueArgs(queue, cl::NullRange, cl::NDRange(inputs[0]->seedDims.size), cl::NullRange), inputs[0]->seed);
			}
			crossEntropyDerivative(cl::EnqueueArgs(queue, cl::NullRange, cl::NDRange(outDims[0].size), cl::NullRange), seed, inputs[0]->result, inputs[1]->result, out[0], inputs[0]->resultDims.dimentions[dimention]);
			explodeUp(cl::EnqueueArgs(queue, cl::NullRange, cl::NDRange(inputs[0]->seedDims.size), cl::NullRange), outDims[0].dimBuf, out[0], inputs[0]->seedDims.dimBuf, inputs[0]->seed);
			inputs[0]->derive();
		}
	}
}






CrossEntropySoftmax::CrossEntropySoftmax(Node* hypothesis, Node* y, int dimentionVal, int meanDimentionVal): CostFunction(hypothesis, y, dimentionVal){
	meanDimention = meanDimentionVal;
	name = "CrossEntropySoftmax";
}

void CrossEntropySoftmax::getDimentions(){
	if(getCount != 0){
		getCount = (getCount + 1) % outCount;
		return;
	}

	inputs[0]->getDimentions();
	inputs[1]->getDimentions();

	resultDims.rank = 0;
	resultDims.size = 1;
	resultDims.dimentions = {};

	resultDims.setBuf();

	if (dimention == -1){
		dimention = inputs[0]->resultDims.rank - 1;
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
	numGroups = (inputs[0]->resultDims.size / inputs[0]->resultDims.dimentions[dimention]) * blocksWide;

	result = cl::Buffer(context, CL_MEM_READ_WRITE, sizeof(float) * resultDims.size);
	resedue = cl::Buffer(context, CL_MEM_READ_WRITE, sizeof(float) * numGroups);
	resultResedue = cl::Buffer(context, CL_MEM_READ_WRITE, sizeof(float) * (inputs[0]->resultDims.size / inputs[0]->resultDims.dimentions[dimention]));
	softmaxMemo = cl::Buffer(context, CL_MEM_READ_WRITE, sizeof(float) * inputs[0]->resultDims.size);
	getCount = (getCount + 1) % outCount;
}

void CrossEntropySoftmax::getValue(){
	if(getCount != 0){
		getCount = (getCount + 1) % outCount;
		return;
	}
	inputs[0]->getValue();
	softmaxPt1(cl::EnqueueArgs(queue, cl::NullRange, cl::NDRange(globalSize), cl::NDRange(GROUP_SIZE)), inputs[0]->result, resedue, inputs[0]->resultDims.dimentions[dimention], preSum, blocksWide);
	softmaxPt2(cl::EnqueueArgs(queue, cl::NullRange, cl::NDRange(inputs[0]->resultDims.size / inputs[0]->resultDims.dimentions[dimention]), cl::NullRange), resedue, resultResedue, blocksWide);
	softmaxPt3(cl::EnqueueArgs(queue, cl::NullRange, cl::NDRange(globalSize), cl::NDRange(GROUP_SIZE)), inputs[0]->result, resultResedue, resedue, inputs[0]->resultDims.dimentions[dimention], preSum, blocksWide);
	softmaxPt4(cl::EnqueueArgs(queue, cl::NullRange, cl::NDRange(inputs[0]->resultDims.size / inputs[0]->resultDims.dimentions[dimention]), cl::NullRange), resedue, resultResedue, blocksWide);
	crossEntropySoftmaxPt5(cl::EnqueueArgs(queue, cl::NullRange, cl::NDRange(globalSize), cl::NDRange(GROUP_SIZE)), inputs[0]->result, inputs[1]->result, resultResedue, resedue, softmaxMemo, inputs[0]->resultDims.dimentions[dimention], preSum, blocksWide);
	meanFullResedue(cl::EnqueueArgs(queue, cl::NullRange, cl::NDRange(resultDims.size), cl::NullRange), resedue, result, inputs[0]->resultDims.dimentions[meanDimention], numGroups);
	getCount = (getCount + 1) % outCount;
}

void CrossEntropySoftmax::derive(){
	getCount = (getCount + 1) % outCount;
	if (getCount == 0){
		if (typeid(*inputs[0]) != typeid(Constant)){
			if (inputs[0]->getCount == 0){
				zeroBuffer(cl::EnqueueArgs(queue, cl::NullRange, cl::NDRange(inputs[0]->seedDims.size), cl::NullRange), inputs[0]->seed);
			}
			crossEntropySoftmaxDerivative(cl::EnqueueArgs(queue, cl::NullRange, cl::NDRange(outDims[0].size), cl::NullRange), seed, softmaxMemo, inputs[1]->result, out[0], inputs[0]->resultDims.dimentions[meanDimention]);
			explodeUp(cl::EnqueueArgs(queue, cl::NullRange, cl::NDRange(inputs[0]->seedDims.size), cl::NullRange), outDims[0].dimBuf, out[0], inputs[0]->seedDims.dimBuf, inputs[0]->seed);
			inputs[0]->derive();
		}
	}
}

string CrossEntropySoftmax::describe(){
	return name + "(" + inputs[0]->describe() + ", " + inputs[1]->describe() + ", " + to_string(dimention) + ", " + to_string(meanDimention) + ")";
}

