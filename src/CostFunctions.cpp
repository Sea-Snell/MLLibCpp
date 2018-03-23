#include "CostFunctions.hpp"

MeanSquared::MeanSquared(Node* hypothesis, Node* y, int dimentionVal){
	dimention = dimentionVal;
	GROUP_SIZE = 128;
	inputs.push_back(hypothesis);
	inputs.push_back(y);
	hypothesis->outputs.push_back(this);
	y->outputs.push_back(this);
	hypothesis->outCount += 1;
	y->outCount += 1;
	name = "MeanSquared";
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

	int numGroups = inputs[0]->resultDims.size / GROUP_SIZE;
	if (numGroups * GROUP_SIZE != inputs[0]->resultDims.size){
		numGroups += 1;
	}

	result = cl::Buffer(context, CL_MEM_READ_WRITE, sizeof(float) * resultDims.size);
	differenceMemo = cl::Buffer(context, CL_MEM_READ_WRITE, sizeof(float) * inputs[0]->resultDims.size);
	diffSquaredResedue = cl::Buffer(context, CL_MEM_READ_WRITE, sizeof(float) * numGroups);
	getCount = (getCount + 1) % outCount;
}

void MeanSquared::getValue(){
	if(getCount != 0){
		getCount = (getCount + 1) % outCount;
		return;
	}
	inputs[0]->getValue();
	inputs[1]->getValue();

	int globalSize = inputs[0]->resultDims.size + (GROUP_SIZE - inputs[0]->resultDims.size % GROUP_SIZE);
	if (inputs[0]->resultDims.size % GROUP_SIZE == 0){
		globalSize -= GROUP_SIZE;
	}
	meanSquared(cl::EnqueueArgs(queue, cl::NullRange, cl::NDRange(globalSize), cl::NDRange(GROUP_SIZE)), inputs[0]->resultDims.dimBuf, inputs[0]->result, inputs[1]->result, differenceMemo, diffSquaredResedue, result, inputs[0]->resultDims.dimentions[dimention]);
	getCount = (getCount + 1) % outCount;
}

void MeanSquared::deriveDimentions(GPUDimentions* tempSeed){
	getCount = (getCount + 1) % outCount;
	seedDimAdd(tempSeed);

	if (getCount == 0){
		outDims.push_back(inputs[0]->resultDims);
		out.push_back(cl::Buffer(context, CL_MEM_READ_WRITE, sizeof(float) * outDims[0].size));
		inputs[0]->deriveDimentions(&outDims[0]);
	}
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

string MeanSquared::describe(){
	return name + "(" + inputs[0]->describe() + ", " + inputs[1]->describe() + ", " + to_string(dimention) + ")";
}





CrossEntropy::CrossEntropy(Node* hypothesis, Node* y, int dimentionVal){
	dimention = dimentionVal;
	GROUP_SIZE = 128;
	inputs.push_back(hypothesis);
	inputs.push_back(y);
	hypothesis->outputs.push_back(this);
	y->outputs.push_back(this);
	hypothesis->outCount += 1;
	y->outCount += 1;
	name = "CrossEntropy";
}

void CrossEntropy::getDimentions(){
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

	int numGroups = inputs[0]->resultDims.size / GROUP_SIZE;
	if (numGroups * GROUP_SIZE != inputs[0]->resultDims.size){
		numGroups += 1;
	}

	result = cl::Buffer(context, CL_MEM_READ_WRITE, sizeof(float) * resultDims.size);
	crossResultResedue = cl::Buffer(context, CL_MEM_READ_WRITE, sizeof(float) * numGroups);
	getCount = (getCount + 1) % outCount;
}

void CrossEntropy::getValue(){
	if(getCount != 0){
		getCount = (getCount + 1) % outCount;
		return;
	}
	inputs[0]->getValue();
	inputs[1]->getValue();
	
	int globalSize = inputs[0]->resultDims.size + (GROUP_SIZE - inputs[0]->resultDims.size % GROUP_SIZE);
	if (inputs[0]->resultDims.size % GROUP_SIZE == 0){
		globalSize -= GROUP_SIZE;
	}
	crossEntropy(cl::EnqueueArgs(queue, cl::NullRange, cl::NDRange(globalSize), cl::NDRange(GROUP_SIZE)), inputs[0]->resultDims.dimBuf, inputs[0]->result, inputs[1]->result, crossResultResedue, result, inputs[0]->resultDims.dimentions[dimention]);
	getCount = (getCount + 1) % outCount;
}

void CrossEntropy::deriveDimentions(GPUDimentions* tempSeed){
	getCount = (getCount + 1) % outCount;
	seedDimAdd(tempSeed);

	if (getCount == 0){
		outDims.push_back(inputs[0]->resultDims);
		out.push_back(cl::Buffer(context, CL_MEM_READ_WRITE, sizeof(float) * outDims[0].size));
		inputs[0]->deriveDimentions(&outDims[0]);
	}
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

string CrossEntropy::describe(){
	return name + "(" + inputs[0]->describe() + ", " + inputs[1]->describe() + ", " + to_string(dimention) + ")";
}

// CrossEntropySoftmax::CrossEntropySoftmax(Node* hypothesis, Node* y, int dimentionVal, int meanDimentionVal){
// 	outCount = 0;
// 	inputs.push_back(hypothesis);
// 	inputs.push_back(y);
// 	hypothesis->outCount += 1;
// 	y->outCount += 1;
// 	name = "CrossEntropySoftmax";
// 	dimention = dimentionVal;
// 	meanDimention = meanDimentionVal;
// 	dCallCount = 0;
// 	gCallCount = 0;
// }

// NumObject CrossEntropySoftmax::getValue(int t, int tf){
// 	gCallCount += 1;
// 	if(gCallCount > 1){
// 		if(gCallCount >= outCount){
// 			gCallCount = 0;
// 		}
// 		return derivativeMemo[t];
// 	}
// 	if(gCallCount >= outCount){
// 		gCallCount = 0;
// 	}

// 	NumObject hypothesis = inputs[0]->getValue(t, tf);
// 	NumObject y = inputs[1]->getValue(t, tf);

// 	if(dimention == -1){
// 		dimention = hypothesis.rank - 1;
// 	}

// 	vector<int> newDimentions;
// 	for(int i = 0; i < hypothesis.rank; i++){
// 		if (i != dimention){
// 			newDimentions.push_back(hypothesis.dimentions[i]);
// 		}
// 	}
// 	double lowVal = -numeric_limits<double>::infinity();
// 	NumObject maxVals = NumObject(newDimentions, lowVal);

// 	int preSum = 1;
// 	for(int i = 0; i < dimention; i++){
// 		preSum *= hypothesis.dimentions[i];
// 	}
// 	int postSum = hypothesis.values.size() / (preSum * hypothesis.dimentions[dimention]);

// 	for(int i = 0; i < preSum; i++){
// 		for(int x = 0; x < hypothesis.dimentions[dimention]; x++){
// 			for(int z = 0; z < postSum; z++){
// 					maxVals.values[i * postSum + z] = max(maxVals.values[i * postSum + z], hypothesis.values[i * postSum * hypothesis.dimentions[dimention] + postSum * x + z]);
// 			}
// 		}
// 	}

// 	NumObject temp = NumObject(maxVals.dimentions, 0.0);
// 	for(int i = 0; i < preSum; i++){
// 		for(int x = 0; x < hypothesis.dimentions[dimention]; x++){
// 			for(int z = 0; z < postSum; z++){
// 				temp.values[i * postSum + z] += exp(hypothesis.values[i * postSum * hypothesis.dimentions[dimention] + x * postSum + z] - maxVals.values[i * postSum + z]);
// 			}
// 		}
// 	}

// 	vector<NumObject> items1 = {maxVals, temp};
// 	NumObject temp2 = mapVals(this, &CrossEntropySoftmax::operation1, items1);

// 	NumObject ans = NumObject(hypothesis.dimentions);
// 	if(t == 0){
// 		softmaxMemo.clear();
// 		softmaxMemo.resize(tf + 1);
// 	}
// 	softmaxMemo[t] = NumObject(hypothesis.dimentions);
// 	double temp3;
// 	for(int i = 0; i < preSum; i++){
// 		for(int x = 0; x < hypothesis.dimentions[dimention]; x++){
// 			for(int z = 0; z < postSum; z++){
// 				temp3 = hypothesis.values[i * postSum * hypothesis.dimentions[dimention] + x * postSum + z] - temp2.values[i * postSum + z];
// 				softmaxMemo[t].values.push_back(exp(temp3));
// 				ans.values.push_back(y.values[i * postSum * hypothesis.dimentions[dimention] + x * postSum + z] * temp3);
// 			}
// 		}
// 	}

// 	NumObject ans2 = reduceSumByDimention(ans, hypothesis.rank);
// 	ans2.values[0] /= -hypothesis.dimentions[meanDimention];

// 	return memoize(ans2, t, tf);
// }

// double CrossEntropySoftmax::operation1(vector<double>& a){
// 	return a[0] + log(a[1]);
// }

// void CrossEntropySoftmax::derive(NumObject& seed, int t, int tf){
// 	if(sumSeed(seed)){
// 		if (typeid(*inputs[0]) != typeid(Constant)){
// 			vector<NumObject> items1 = {tempSeed, NumObject(1.0 / inputs[0]->derivativeMemo[t].dimentions[meanDimention]), softmaxMemo[t], inputs[1]->derivativeMemo[t]};
// 			NumObject eval1 = mapVals(this, &CrossEntropySoftmax::deriveOperation1, items1);
// 			inputs[0]->derive(eval1, t, tf);
// 		}
// 	}
// }

// double CrossEntropySoftmax::deriveOperation1(vector<double>& a){
// 	return a[0] * a[1] * (a[2] - a[3]);
// }

// string CrossEntropySoftmax::describe(){
// 	return name + "(" + inputs[0]->describe() + ", " + inputs[1]->describe() + ", " + to_string(dimention) + ", " + to_string(meanDimention) + ")";
// }

