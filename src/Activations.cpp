#include "Activations.hpp"
#include "MapVals.hpp"
#include "HelperFunctions.hpp"

double Sigmoid::operation(vector<double>& a){
	return 1.0 / (1.0 + exp(-a[0]));
}

void Sigmoid::derive(NumObject& seed){
	if (typeid(*inputs[0]) != typeid(Constant)){
		vector<NumObject> items1 = {seed, derivativeMemo};
		NumObject eval1 = mapVals(this, &Sigmoid::deriveOperation1, items1);
		inputs[0]->derive(eval1);
	}
}

double Sigmoid::deriveOperation1(vector<double>& a){
	return a[0] * a[1] * (1.0 - a[1]);
}

double ReLU::operation(vector<double>& a){
	if (a[0] >= 0){
		return a[0];
	}
	return 0.0;
}

void ReLU::derive(NumObject& seed){
	if (typeid(*inputs[0]) != typeid(Constant)){
		vector<NumObject> items1 = {seed, inputs[0]->derivativeMemo};
		NumObject eval1 = mapVals(this, &ReLU::deriveOperation1, items1);
		inputs[0]->derive(eval1);
	}
}

double ReLU::deriveOperation1(vector<double>& a){
	if(a[1] >= 0){
		return a[0];
	}
	return 0.0;
}

double LeakyReLU::operation(vector<double>& a){
	if (a[0] >= 0){
		return a[0];
	}
	return 0.01 * a[0];
}

void LeakyReLU::derive(NumObject& seed){
	if (typeid(*inputs[0]) != typeid(Constant)){
		vector<NumObject> items1 = {seed, inputs[0]->derivativeMemo};
		NumObject eval1 = mapVals(this, &LeakyReLU::deriveOperation1, items1);
		inputs[0]->derive(eval1);
	}
}

double LeakyReLU::deriveOperation1(vector<double>& a){
	if(a[1] >= 0){
		return a[0];
	}
	return 0.01 * a[0];
}

double Gaussian::operation(vector<double>& a){
	return exp(-a[0] * a[0]);
}

void Gaussian::derive(NumObject& seed){
	if (typeid(*inputs[0]) != typeid(Constant)){
		vector<NumObject> items1 = {seed, derivativeMemo, inputs[0]->derivativeMemo};
		NumObject eval1 = mapVals(this, &Gaussian::deriveOperation1, items1);
		inputs[0]->derive(eval1);
	}
}

double Gaussian::deriveOperation1(vector<double>& a){
	return a[0] * -2.0 * a[2] * a[1];
}

Softmax::Softmax(Node* a, int dimentionVal){
	inputs.push_back(a);
	name = "Softmax";
	dimention = dimentionVal;
}

NumObject Softmax::getValue(){
	NumObject a = inputs[0]->getValue();

	if(dimention == -1){
		dimention = a.rank - 1;
	}

	vector<int> newDimentions;
	for(int i = 0; i < a.rank; i++){
		if (i != dimention){
			newDimentions.push_back(a.dimentions[i]);
		}
	}
	double lowVal = -numeric_limits<double>::infinity();
	NumObject maxVals = NumObject(newDimentions, lowVal);

	int preSum = 1;
	for(int i = 0; i < dimention; i++){
		preSum *= a.dimentions[i];
	}
	int postSum = a.values.size() / (preSum * a.dimentions[dimention]);

	for(int i = 0; i < preSum; i++){
		for(int x = 0; x < a.dimentions[dimention]; x++){
			for(int z = 0; z < postSum; z++){
					maxVals.values[i * postSum + z] = max(maxVals.values[i * postSum + z], a.values[i * postSum * a.dimentions[dimention] + postSum * x + z]);
			}
		}
	}

	NumObject temp = NumObject(maxVals.dimentions, 0.0);
	for(int i = 0; i < preSum; i++){
		for(int x = 0; x < a.dimentions[dimention]; x++){
			for(int z = 0; z < postSum; z++){
				temp.values[i * postSum + z] += exp(a.values[i * postSum * a.dimentions[dimention] + x * postSum + z] - maxVals.values[i * postSum + z]);
			}
		}
	}

	vector<NumObject> items1 = {maxVals, temp};
	NumObject temp2 = mapVals(this, &Softmax::operation1, items1);

	NumObject ans = NumObject(a.dimentions);
	for(int i = 0; i < preSum; i++){
		for(int x = 0; x < a.dimentions[dimention]; x++){
			for(int z = 0; z < postSum; z++){
				ans.values.push_back(exp(a.values[i * postSum * a.dimentions[dimention] + x * postSum + z] - temp2.values[i * postSum + z]));
			}
		}
	}

	return memoize(ans);
}

double Softmax::operation1(vector<double>& a){
	return a[0] + log(a[1]);
}

void Softmax::derive(NumObject& seed){
	if (typeid(*inputs[0]) != typeid(Constant)){
		vector<NumObject> items1 = {seed, derivativeMemo};
		NumObject eval1 = mapVals(this, &Softmax::deriveOperation1, items1);

		vector<int> newDimentions;
		for(int i = 0; i < inputs[0]->derivativeMemo.rank; i++){
			if (i != dimention){
				newDimentions.push_back(inputs[0]->derivativeMemo.dimentions[i]);
			}
		}

		int preSum = 1;
		for(int i = 0; i < dimention; i++){
			preSum *= eval1.dimentions[i];
		}
		int postSum = eval1.values.size() / (preSum * eval1.dimentions[dimention]);

		NumObject sums = NumObject(newDimentions, 0.0);
		for(int i = 0; i < preSum; i++){
			for(int x = 0; x < eval1.dimentions[dimention]; x++){
				for(int z = 0; z < postSum; z++){
					sums.values[i * postSum + z] += eval1.values[i * postSum * eval1.dimentions[dimention] + x * postSum + z];
				}
			}
		}

		NumObject ans = NumObject(eval1.dimentions);
		for(int i = 0; i < preSum; i++){
			for(int x = 0; x < eval1.dimentions[dimention]; x++){
				for(int z = 0; z < postSum; z++){
					ans.values.push_back(derivativeMemo.values[i * postSum * eval1.dimentions[dimention] + x * postSum + z] * (seed.values[(i * postSum * eval1.dimentions[dimention] + x * postSum + z) % seed.values.size()] - sums.values[i * postSum + z]));
				}
			}
		}

		inputs[0]->derive(ans);
	}
}

double Softmax::deriveOperation1(vector<double>& a){
	return a[0] * a[1];
}

double TanH::operation(vector<double>& a){
	return 2.0 / (1.0 + exp(-2.0 * a[0])) - 1.0;
}

void TanH::derive(NumObject& seed){
	if (typeid(*inputs[0]) != typeid(Constant)){
		vector<NumObject> items1 = {seed, derivativeMemo};
		NumObject eval1 = mapVals(this, &TanH::deriveOperation1, items1);
		inputs[0]->derive(eval1);
	}
}

double TanH::deriveOperation1(vector<double>& a){
	return a[0] * (1.0 - a[1] * a[1]);
}


