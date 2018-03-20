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

// double ReLU::operation(vector<double>& a){
// 	if (a[0] >= 0){
// 		return a[0];
// 	}
// 	return 0.0;
// }

// void ReLU::derive(NumObject& seed, int t, int tf){
// 	if(sumSeed(seed)){
// 		if (typeid(*inputs[0]) != typeid(Constant)){
// 			vector<NumObject> items1 = {tempSeed, inputs[0]->derivativeMemo[t]};
// 			NumObject eval1 = mapVals(this, &ReLU::deriveOperation1, items1);
// 			inputs[0]->derive(eval1, t, tf);
// 		}
// 	}
// }

// double ReLU::deriveOperation1(vector<double>& a){
// 	if(a[1] >= 0){
// 		return a[0];
// 	}
// 	return 0.0;
// }

// double LeakyReLU::operation(vector<double>& a){
// 	if (a[0] >= 0){
// 		return a[0];
// 	}
// 	return 0.01 * a[0];
// }

// void LeakyReLU::derive(NumObject& seed, int t, int tf){
// 	if(sumSeed(seed)){
// 		if (typeid(*inputs[0]) != typeid(Constant)){
// 			vector<NumObject> items1 = {tempSeed, inputs[0]->derivativeMemo[t]};
// 			NumObject eval1 = mapVals(this, &LeakyReLU::deriveOperation1, items1);
// 			inputs[0]->derive(eval1, t, tf);
// 		}
// 	}
// }

// double LeakyReLU::deriveOperation1(vector<double>& a){
// 	if(a[1] >= 0){
// 		return a[0];
// 	}
// 	return 0.01 * a[0];
// }

// double Gaussian::operation(vector<double>& a){
// 	return exp(-a[0] * a[0]);
// }

// void Gaussian::derive(NumObject& seed, int t, int tf){
// 	if(sumSeed(seed)){
// 		if (typeid(*inputs[0]) != typeid(Constant)){
// 			vector<NumObject> items1 = {tempSeed, derivativeMemo[t], inputs[0]->derivativeMemo[t]};
// 			NumObject eval1 = mapVals(this, &Gaussian::deriveOperation1, items1);
// 			inputs[0]->derive(eval1, t, tf);
// 		}
// 	}
// }

// double Gaussian::deriveOperation1(vector<double>& a){
// 	return a[0] * -2.0 * a[2] * a[1];
// }

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

// double TanH::operation(vector<double>& a){
// 	return 2.0 / (1.0 + exp(-2.0 * a[0])) - 1.0;
// }

// void TanH::derive(NumObject& seed, int t, int tf){
// 	if(sumSeed(seed)){
// 		if (typeid(*inputs[0]) != typeid(Constant)){
// 			vector<NumObject> items1 = {tempSeed, derivativeMemo[t]};
// 			NumObject eval1 = mapVals(this, &TanH::deriveOperation1, items1);
// 			inputs[0]->derive(eval1, t, tf);
// 		}
// 	}
// }

// double TanH::deriveOperation1(vector<double>& a){
// 	return a[0] * (1.0 - a[1] * a[1]);
// }

// double Softsign::operation(vector<double>& a){
// 	return a[0] / (1.0 + abs(a[0]));
// }

// void Softsign::derive(NumObject& seed, int t, int tf){
// 	if(sumSeed(seed)){
// 		if (typeid(*inputs[0]) != typeid(Constant)){
// 			vector<NumObject> items1 = {tempSeed, inputs[0]->derivativeMemo[t]};
// 			NumObject eval1 = mapVals(this, &Softsign::deriveOperation1, items1);
// 			inputs[0]->derive(eval1, t, tf);
// 		}
// 	}
// }

// double Softsign::deriveOperation1(vector<double>& a){
// 	return a[0] / ((1.0 + abs(a[1])) * (1.0 + abs(a[1])));
// }


