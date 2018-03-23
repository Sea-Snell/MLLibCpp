#include "Activations.hpp"


void Sigmoid::getValue(){
	if(getCount != 0){
		getCount = (getCount + 1) % outCount;
		return;
	}
	inputs[0]->getValue();
	sigmoid(cl::EnqueueArgs(queue, cl::NullRange, cl::NDRange(resultDims.size), cl::NullRange), inputs[0]->result, result);
	getCount = (getCount + 1) % outCount;
}

void Sigmoid::derive(){
	getCount = (getCount + 1) % outCount;
	if (getCount == 0){
		if (typeid(*inputs[0]) != typeid(Constant)){
			if (inputs[0]->getCount == 0){
				zeroBuffer(cl::EnqueueArgs(queue, cl::NullRange, cl::NDRange(inputs[0]->seedDims.size), cl::NullRange), inputs[0]->seed);
			}
			sigmoidDerivative(cl::EnqueueArgs(queue, cl::NullRange, cl::NDRange(outDims[0].size), cl::NullRange), result, seedDims.dimBuf, seed, out[0]);
			explodeUp(cl::EnqueueArgs(queue, cl::NullRange, cl::NDRange(inputs[0]->seedDims.size), cl::NullRange), outDims[0].dimBuf, out[0], inputs[0]->seedDims.dimBuf, inputs[0]->seed);
			inputs[0]->derive();
		}
	}
}

void ReLU::getValue(){
	if(getCount != 0){
		getCount = (getCount + 1) % outCount;
		return;
	}
	inputs[0]->getValue();
	reLU(cl::EnqueueArgs(queue, cl::NullRange, cl::NDRange(resultDims.size), cl::NullRange), inputs[0]->result, result);
	getCount = (getCount + 1) % outCount;
}

void ReLU::derive(){
	getCount = (getCount + 1) % outCount;
	if (getCount == 0){
		if (typeid(*inputs[0]) != typeid(Constant)){
			if (inputs[0]->getCount == 0){
				zeroBuffer(cl::EnqueueArgs(queue, cl::NullRange, cl::NDRange(inputs[0]->seedDims.size), cl::NullRange), inputs[0]->seed);
			}
			reLUDerivative(cl::EnqueueArgs(queue, cl::NullRange, cl::NDRange(outDims[0].size), cl::NullRange), inputs[0]->result, seedDims.dimBuf, seed, out[0]);
			explodeUp(cl::EnqueueArgs(queue, cl::NullRange, cl::NDRange(inputs[0]->seedDims.size), cl::NullRange), outDims[0].dimBuf, out[0], inputs[0]->seedDims.dimBuf, inputs[0]->seed);
			inputs[0]->derive();
		}
	}
}

void LeakyReLU::getValue(){
	if(getCount != 0){
		getCount = (getCount + 1) % outCount;
		return;
	}
	inputs[0]->getValue();
	leakyReLU(cl::EnqueueArgs(queue, cl::NullRange, cl::NDRange(resultDims.size), cl::NullRange), inputs[0]->result, result);
	getCount = (getCount + 1) % outCount;
}

void LeakyReLU::derive(){
	getCount = (getCount + 1) % outCount;
	if (getCount == 0){
		if (typeid(*inputs[0]) != typeid(Constant)){
			if (inputs[0]->getCount == 0){
				zeroBuffer(cl::EnqueueArgs(queue, cl::NullRange, cl::NDRange(inputs[0]->seedDims.size), cl::NullRange), inputs[0]->seed);
			}
			leakyReLUDerivative(cl::EnqueueArgs(queue, cl::NullRange, cl::NDRange(outDims[0].size), cl::NullRange), inputs[0]->result, seedDims.dimBuf, seed, out[0]);
			explodeUp(cl::EnqueueArgs(queue, cl::NullRange, cl::NDRange(inputs[0]->seedDims.size), cl::NullRange), outDims[0].dimBuf, out[0], inputs[0]->seedDims.dimBuf, inputs[0]->seed);
			inputs[0]->derive();
		}
	}
}

void Gaussian::getValue(){
	if(getCount != 0){
		getCount = (getCount + 1) % outCount;
		return;
	}
	inputs[0]->getValue();
	gaussian(cl::EnqueueArgs(queue, cl::NullRange, cl::NDRange(resultDims.size), cl::NullRange), inputs[0]->result, result);
	getCount = (getCount + 1) % outCount;
}

void Gaussian::derive(){
	getCount = (getCount + 1) % outCount;
	if (getCount == 0){
		if (typeid(*inputs[0]) != typeid(Constant)){
			if (inputs[0]->getCount == 0){
				zeroBuffer(cl::EnqueueArgs(queue, cl::NullRange, cl::NDRange(inputs[0]->seedDims.size), cl::NullRange), inputs[0]->seed);
			}
			gaussianDerivative(cl::EnqueueArgs(queue, cl::NullRange, cl::NDRange(outDims[0].size), cl::NullRange), inputs[0]->result, result, seedDims.dimBuf, seed, out[0]);
			explodeUp(cl::EnqueueArgs(queue, cl::NullRange, cl::NDRange(inputs[0]->seedDims.size), cl::NullRange), outDims[0].dimBuf, out[0], inputs[0]->seedDims.dimBuf, inputs[0]->seed);
			inputs[0]->derive();
		}
	}
}

// Softmax::Softmax(Node* a, int dimentionVal){
// 	outCount = 0;
// 	inputs.push_back(a);
// 	a->outCount += 1;
// 	name = "Softmax";
// 	dimention = dimentionVal;
// 	dCallCount = 0;
// 	gCallCount = 0;
// }

// NumObject Softmax::getValue(int t, int tf){
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

// 	NumObject a = inputs[0]->getValue(t, tf);

// 	if(dimention == -1){
// 		dimention = a.rank - 1;
// 	}

// 	vector<int> newDimentions;
// 	for(int i = 0; i < a.rank; i++){
// 		if (i != dimention){
// 			newDimentions.push_back(a.dimentions[i]);
// 		}
// 	}
// 	double lowVal = -numeric_limits<double>::infinity();
// 	NumObject maxVals = NumObject(newDimentions, lowVal);

// 	int preSum = 1;
// 	for(int i = 0; i < dimention; i++){
// 		preSum *= a.dimentions[i];
// 	}
// 	int postSum = a.values.size() / (preSum * a.dimentions[dimention]);

// 	for(int i = 0; i < preSum; i++){
// 		for(int x = 0; x < a.dimentions[dimention]; x++){
// 			for(int z = 0; z < postSum; z++){
// 					maxVals.values[i * postSum + z] = max(maxVals.values[i * postSum + z], a.values[i * postSum * a.dimentions[dimention] + postSum * x + z]);
// 			}
// 		}
// 	}

// 	NumObject temp = NumObject(maxVals.dimentions, 0.0);
// 	for(int i = 0; i < preSum; i++){
// 		for(int x = 0; x < a.dimentions[dimention]; x++){
// 			for(int z = 0; z < postSum; z++){
// 				temp.values[i * postSum + z] += exp(a.values[i * postSum * a.dimentions[dimention] + x * postSum + z] - maxVals.values[i * postSum + z]);
// 			}
// 		}
// 	}

// 	vector<NumObject> items1 = {maxVals, temp};
// 	NumObject temp2 = mapVals(this, &Softmax::operation1, items1);

// 	NumObject ans = NumObject(a.dimentions);
// 	for(int i = 0; i < preSum; i++){
// 		for(int x = 0; x < a.dimentions[dimention]; x++){
// 			for(int z = 0; z < postSum; z++){
// 				ans.values.push_back(exp(a.values[i * postSum * a.dimentions[dimention] + x * postSum + z] - temp2.values[i * postSum + z]));
// 			}
// 		}
// 	}

// 	return memoize(ans, t, tf);
// }

// double Softmax::operation1(vector<double>& a){
// 	return a[0] + log(a[1]);
// }

// void Softmax::derive(NumObject& seed, int t, int tf){
// 	if(sumSeed(seed)){
// 		if (typeid(*inputs[0]) != typeid(Constant)){
// 			vector<NumObject> items1 = {tempSeed, derivativeMemo[t]};
// 			NumObject eval1 = mapVals(this, &Softmax::deriveOperation1, items1);

// 			vector<int> newDimentions;
// 			for(int i = 0; i < inputs[0]->derivativeMemo[t].rank; i++){
// 				if (i != dimention){
// 					newDimentions.push_back(inputs[0]->derivativeMemo[t].dimentions[i]);
// 				}
// 			}

// 			int preSum = 1;
// 			for(int i = 0; i < dimention; i++){
// 				preSum *= eval1.dimentions[i];
// 			}
// 			int postSum = eval1.values.size() / (preSum * eval1.dimentions[dimention]);

// 			NumObject sums = NumObject(newDimentions, 0.0);
// 			for(int i = 0; i < preSum; i++){
// 				for(int x = 0; x < eval1.dimentions[dimention]; x++){
// 					for(int z = 0; z < postSum; z++){
// 						sums.values[i * postSum + z] += eval1.values[i * postSum * eval1.dimentions[dimention] + x * postSum + z];
// 					}
// 				}
// 			}

// 			NumObject ans = NumObject(eval1.dimentions);
// 			for(int i = 0; i < preSum; i++){
// 				for(int x = 0; x < eval1.dimentions[dimention]; x++){
// 					for(int z = 0; z < postSum; z++){
// 						ans.values.push_back(derivativeMemo[t].values[i * postSum * eval1.dimentions[dimention] + x * postSum + z] * (tempSeed.values[(i * postSum * eval1.dimentions[dimention] + x * postSum + z) % tempSeed.values.size()] - sums.values[i * postSum + z]));
// 					}
// 				}
// 			}

// 			inputs[0]->derive(ans, t, tf);
// 		}
// 	}
// }

// double Softmax::deriveOperation1(vector<double>& a){
// 	return a[0] * a[1];
// }


void TanH::getValue(){
	if(getCount != 0){
		getCount = (getCount + 1) % outCount;
		return;
	}
	inputs[0]->getValue();
	tanH(cl::EnqueueArgs(queue, cl::NullRange, cl::NDRange(resultDims.size), cl::NullRange), inputs[0]->result, result);
	getCount = (getCount + 1) % outCount;
}

void TanH::derive(){
	getCount = (getCount + 1) % outCount;
	if (getCount == 0){
		if (typeid(*inputs[0]) != typeid(Constant)){
			if (inputs[0]->getCount == 0){
				zeroBuffer(cl::EnqueueArgs(queue, cl::NullRange, cl::NDRange(inputs[0]->seedDims.size), cl::NullRange), inputs[0]->seed);
			}
			tanHDerivative(cl::EnqueueArgs(queue, cl::NullRange, cl::NDRange(outDims[0].size), cl::NullRange), result, seedDims.dimBuf, seed, out[0]);
			explodeUp(cl::EnqueueArgs(queue, cl::NullRange, cl::NDRange(inputs[0]->seedDims.size), cl::NullRange), outDims[0].dimBuf, out[0], inputs[0]->seedDims.dimBuf, inputs[0]->seed);
			inputs[0]->derive();
		}
	}
}

void Softsign::getValue(){
	if(getCount != 0){
		getCount = (getCount + 1) % outCount;
		return;
	}
	inputs[0]->getValue();
	softsign(cl::EnqueueArgs(queue, cl::NullRange, cl::NDRange(resultDims.size), cl::NullRange), inputs[0]->result, result);
	getCount = (getCount + 1) % outCount;
}

void Softsign::derive(){
	getCount = (getCount + 1) % outCount;
	if (getCount == 0){
		if (typeid(*inputs[0]) != typeid(Constant)){
			if (inputs[0]->getCount == 0){
				zeroBuffer(cl::EnqueueArgs(queue, cl::NullRange, cl::NDRange(inputs[0]->seedDims.size), cl::NullRange), inputs[0]->seed);
			}
			softsignDerivative(cl::EnqueueArgs(queue, cl::NullRange, cl::NDRange(outDims[0].size), cl::NullRange), inputs[0]->result, seedDims.dimBuf, seed, out[0]);
			explodeUp(cl::EnqueueArgs(queue, cl::NullRange, cl::NDRange(inputs[0]->seedDims.size), cl::NullRange), outDims[0].dimBuf, out[0], inputs[0]->seedDims.dimBuf, inputs[0]->seed);
			inputs[0]->derive();
		}
	}
}


