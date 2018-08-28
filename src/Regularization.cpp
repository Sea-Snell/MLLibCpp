#include "Regularization.hpp"
#include <random>


// L2::L2(Node* cost, vector<Node*> weights, int dataSize, double parameterVal){
// 	outCount = 0;
// 	inputs.push_back(cost);
// 	cost->outCount += 1;
// 	for(int i = 0; i < weights.size(); i++){
// 		inputs.push_back(weights[i]);
// 		weights[i]->outCount += 1;
// 	}
// 	name = "L2";
// 	parameter = parameterVal;
// 	size = dataSize;
// 	dCallCount = 0;
// 	gCallCount = 0;
// }

// NumObject L2::getValue(int t, int tf){
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

// 	NumObject cost = inputs[0]->getValue(t, tf);

// 	if(parameter == 0.0){
// 		return memoize(cost, t, tf);
// 	}

// 	vector<NumObject> weights;
// 	for(int i = 1; i < inputs.size(); i++){
// 		weights.push_back(inputs[i]->getValue(t, tf));
// 	}

// 	double sum = 0.0;
// 	for(int i = 0; i < weights.size(); i++){
// 		for(int x = 0; x < weights[i].values.size(); x++){
// 			sum += weights[i].values[x] * weights[i].values[x];
// 		}
// 	}

// 	sum *= parameter / (2.0 * size);

// 	NumObject ans = NumObject(sum + cost.values[0]);
// 	return memoize(ans, t, tf);
// }

// void L2::derive(NumObject& seed, int t, int tf){
// 	if(sumSeed(seed)){
// 		inputs[0]->derive(tempSeed, t, tf);

// 		if(parameter != 0.0){	
// 			for(int i = 1; i < inputs.size(); i++){
// 				vector<NumObject> items = {inputs[i]->derivativeMemo[t], tempSeed, parameter / size};
// 				NumObject ans = mapVals(this, &L2::operation, items);

// 				inputs[i]->derive(ans, t, tf);
// 			}
// 		}
// 	}
// }

// double L2::operation(vector<double>& a){
// 	return a[0] * a[1] * a[2];
// }

// string L2::describe(){
// 	string ans = name + "(";
// 	for(int i = 0; i < inputs.size(); i++){
// 		ans += inputs[i]->describe();
// 		ans += ", ";
// 	}
// 	ans += to_string(size) + ", " + to_string(parameter) + ")";
// 	return ans;
// }

// NumObject L1::getValue(int t, int tf){
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

// 	NumObject cost = inputs[0]->getValue(t, tf);

// 	if(parameter == 0.0){
// 		return memoize(cost, t, tf);
// 	}

// 	vector<NumObject> weights;
// 	for(int i = 1; i < inputs.size(); i++){
// 		weights.push_back(inputs[i]->getValue());
// 	}

// 	double sum = 0.0;
// 	for(int i = 0; i < weights.size(); i++){
// 		for(int x = 0; x < weights[i].values.size(); x++){
// 			sum += abs(weights[i].values[x]);
// 		}
// 	}

// 	sum *= parameter / (2.0 * size);

// 	NumObject ans = NumObject(sum + cost.values[0]);
// 	return memoize(ans, t, tf);
// }

// void L1::derive(NumObject& seed, int t, int tf){
// 	if(sumSeed(seed)){
// 		inputs[0]->derive(tempSeed, t, tf);

// 		if(parameter != 0.0){	
// 			for(int i = 1; i < inputs.size(); i++){
// 				vector<NumObject> items = {inputs[i]->derivativeMemo[t], tempSeed, parameter / size};
// 				NumObject ans = mapVals(this, &L1::operation, items);

// 				inputs[i]->derive(ans, t, tf);
// 			}
// 		}
// 	}
// }

// double L1::operation(vector<double>& a){
// 	if(a[0] > 0){
// 		return a[1] * a[2];
// 	}
// 	if(a[0] < 0){
// 		return -a[1] * a[2];
// 	}
// 	return 0.0;
// }

// void maxNorm(NumObject& weight, int dimention, double c){
// 	int preSum = 1;
// 	for(int i = 0; i < dimention; i++){
// 		preSum *= weight.dimentions[i];
// 	}
// 	int postSum = weight.values.size() / (preSum * weight.dimentions[dimention]);

// 	for(int i = 0; i < preSum; i++){
// 		for(int z = 0; z < postSum; z++){
// 			double length = 0.0;
// 			for(int x = 0; x < weight.dimentions[dimention]; x++){
// 				int idx = i * postSum * weight.dimentions[dimention] + x * postSum + z;
// 				length += weight.values[idx] * weight.values[idx];
// 			}
// 			if(length > c * c){
// 				for(int x = 0; x < weight.dimentions[dimention]; x++){
// 					int idx = i * postSum * weight.dimentions[dimention] + x * postSum + z;
// 					weight.values[idx] = c * (weight.values[idx] / sqrt(length));
// 				}
// 			}
// 		}
// 	}
// }


Dropout::Dropout(Node* a, float probabilityVal){
	inputs.push_back(a);
	a->outputs.push_back(this);
	a->outCount += 1;
	name = "Dropout";
	probability = probabilityVal;
}

void Dropout::updateDrop(){
  	random_device rd;
    mt19937 gen(rd());
    uniform_real_distribution<float> distribution(0.0, 1.0);

    for (int x = 0; x < timeSteps; x++){
    	vector<float> dropped = {};
    	dropped.reserve(droppedDims.size);
		for(int i = 0; i < droppedDims.size; i++){
			if(distribution(gen) > probability){
				dropped.push_back(1.0);
			}
			else{
				dropped.push_back(0.0);
			}
		}
		queue.enqueueWriteBuffer(droppedBuf[x], CL_TRUE, 0, sizeof(float) * droppedDims.size, &dropped[0]);
	}
}


void Dropout::getDimentions(){
	if(getCount != 0){
		getCount = (getCount + 1) % outCount;
		return;
	}

	inputs[0]->getDimentions();

	resultDims.rank = inputs[0]->resultDims.rank;
	resultDims.size = inputs[0]->resultDims.size;
	resultDims.dimentions = inputs[0]->resultDims.dimentions;

	droppedDims.rank = 1;
	droppedDims.size = resultDims.dimentions[resultDims.rank - 1];
	droppedDims.dimentions = {resultDims.dimentions[resultDims.rank - 1]};

	resultDims.setBuf();
	droppedDims.setBuf();

	result = {};
	droppedBuf = {};
	for (int i = 0; i < timeSteps; i++){
		result.push_back(cl::Buffer(context, CL_MEM_READ_WRITE, sizeof(float) * resultDims.size));
		droppedBuf.push_back(cl::Buffer(context, CL_MEM_READ_WRITE, sizeof(float) * droppedDims.size));
	}

	getCount = (getCount + 1) % outCount;
}

void Dropout::getValue(){
	if(getCount != 0){
		getCount = (getCount + 1) % outCount;
		if (getCount == 0){
			currentTime = (currentTime + 1) % timeSteps;
		}
		return;
	}
	inputs[0]->getValue();
	multiply(cl::EnqueueArgs(queue, cl::NullRange, cl::NDRange(resultDims.size), cl::NullRange), droppedDims.dimBuf, droppedBuf[currentTime], inputs[0]->resultDims.dimBuf, inputs[0]->result[currentTime % inputs[0]->timeSteps], result[currentTime]);
	getCount = (getCount + 1) % outCount;
	if (getCount == 0){
		currentTime = (currentTime + 1) % timeSteps;
	}
}

void Dropout::deriveDimentions(GPUDimentions* tempSeed){
	getCount = (getCount + 1) % outCount;
	seedDimAdd(tempSeed);

	if (getCount == 0){
		outDims.push_back(getMaxDimentions(vector<GPUDimentions*>{&seedDims, &droppedDims}));
		out.push_back(cl::Buffer(context, CL_MEM_READ_WRITE, sizeof(float) * outDims[0].size));
		inputs[0]->deriveDimentions(&outDims[0]);
	}
}

void Dropout::derive(){
	getCount = (getCount + 1) % outCount;
	if (getCount == 0){
		int realTime = (timeSteps - 1) - currentTime;

		if (typeid(*inputs[0]) != typeid(Constant)){
			if (inputs[0]->getCount == 0 && (inputs[0]->timeSteps != 1 || (inputs[0]->timeSteps == 1 && currentTime == 0))){
				zeroBuffer(cl::EnqueueArgs(queue, cl::NullRange, cl::NDRange(inputs[0]->seedDims.size), cl::NullRange), inputs[0]->seed);
			}
			multiply(cl::EnqueueArgs(queue, cl::NullRange, cl::NDRange(outDims[0].size), cl::NullRange), droppedDims.dimBuf, droppedBuf[realTime], seedDims.dimBuf, seed, out[0]);
			explodeUp(cl::EnqueueArgs(queue, cl::NullRange, cl::NDRange(inputs[0]->seedDims.size), cl::NullRange), outDims[0].dimBuf, out[0], inputs[0]->seedDims.dimBuf, inputs[0]->seed);
			inputs[0]->derive();
		}

		currentTime = (currentTime + 1) % timeSteps;
	}
}

string Dropout::describe(){
	return name + "(" + inputs[0]->describe() + ", " + to_string(probability) + ")";
}

