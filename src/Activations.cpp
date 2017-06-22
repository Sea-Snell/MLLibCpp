#include "Activations.hpp"
#include "HelperFunctions.hpp"

double Sigmoid::operation(double a){
	return 1.0 / (1.0 + exp(-a));
}

void Sigmoid::derive(vector<double>& seed){
	if (typeid(*inputs[0]) != typeid(Constant)){
		for(int i = 0; i < outSize; i++){
			ans[i] = seed[i % seedSize] * derivativeMemo[i] * (1.0 - derivativeMemo[i]);
		}
		inputs[0]->derive(ans);
	}
}

void Sigmoid::deriveDimentions(vector<int>& seedDimentionsVal){
	seedDimentions = seedDimentionsVal;
	seedRank = seedDimentions.size();
	seedSize = 1;
	for(int i = 0; i < seedRank; i++){
		seedSize *= seedDimentions[i];
	}

	ans.clear();
	ans.resize(outSize, 0.0);

	inputs[0]->deriveDimentions(outDimentions);
}

double ReLU::operation(double a){
	if (a >= 0){
		return a;
	}
	return 0.0;
}

void ReLU::derive(vector<double>& seed){
	if (typeid(*inputs[0]) != typeid(Constant)){
		for(int i = 0; i < outSize; i++){
			ans[i] = seed[i % seedSize] * (int)(inputs[0]->derivativeMemo[i] >= 0.0);
		}
		inputs[0]->derive(ans);
	}
}

double LeakyReLU::operation(double a){
	if (a >= 0){
		return a;
	}
	return 0.01 * a;
}

void LeakyReLU::derive(vector<double>& seed){
	if (typeid(*inputs[0]) != typeid(Constant)){
		for(int i = 0; i < outSize; i++){
			if(inputs[0]->derivativeMemo[i] >= 0.0){
				ans[i] = seed[i % seedSize];
			}
			else{
				ans[i] = seed[i % seedSize] * 0.01;
			}
		}
		inputs[0]->derive(ans);
	}
}

double Gaussian::operation(double a){
	return exp(-a * a);
}

void Gaussian::derive(vector<double>& seed){
	if (typeid(*inputs[0]) != typeid(Constant)){
		for(int i = 0; i < outSize; i++){
			ans[i] = -2.0 * seed[i % seedSize] * derivativeMemo[i] * inputs[0]->derivativeMemo[i];
		}
		inputs[0]->derive(ans);
	}
}

Softmax::Softmax(Node* a, int dimentionVal){
	inputs.push_back(a);
	name = "Softmax";
	dimention = dimentionVal;
}

void Softmax::getValue(){
	inputs[0]->getValue();

	double lowVal = -numeric_limits<double>::infinity();
	for(int i = 0; i < newSize; i++){
		maxVals[i] = lowVal;
	}

	for(int i = 0; i < preSum; i++){
		for(int x = 0; x < inputs[0]->outDimentions[dimention]; x++){
			for(int z = 0; z < postSum; z++){
				maxVals[i * postSum + z] = max(maxVals[i * postSum + z], inputs[0]->derivativeMemo[i * postSum * inputs[0]->outDimentions[dimention] + postSum * x + z]);
			}
		}
	}

	for(int i = 0; i < preSum; i++){
		for(int z = 0; z < postSum; z++){
			double sum = 0.0;
			for(int x = 0; x < inputs[0]->outDimentions[dimention]; x++){
				sum += exp(inputs[0]->derivativeMemo[i * postSum * inputs[0]->outDimentions[dimention] + x * postSum + z] - maxVals[i * postSum + z]);
			}
			temp[i * postSum + z] = sum;
		}
	}

	for(int i = 0; i < newSize; i++){
		temp2[i] = maxVals[i] + log(temp[i]);
	}

	for(int i = 0; i < preSum; i++){
		for(int x = 0; x < inputs[0]->outDimentions[dimention]; x++){
			for(int z = 0; z < postSum; z++){
				derivativeMemo[i * postSum * inputs[0]->outDimentions[dimention] + x * postSum + z] = exp(inputs[0]->derivativeMemo[i * postSum * inputs[0]->outDimentions[dimention] + x * postSum + z] - temp2[i * postSum + z]);
			}
		}
	}
}

void Softmax::getValueDimentions(){
	inputs[0]->getValueDimentions();

	outDimentions = inputs[0]->outDimentions;
	outSize = inputs[0]->outSize;
	outRank = inputs[0]->outRank;

	derivativeMemo.clear();
	derivativeMemo.resize(outSize, 0.0);

	if(dimention == -1){
		dimention = inputs[0]->outRank - 1;
	}

	newDimentions.clear();
	newSize = 1;
	for(int i = 0; i < inputs[0]->outRank; i++){
		if (i != dimention){
			newSize *= inputs[0]->outDimentions[i];
			newDimentions.push_back(inputs[0]->outDimentions[i]);
		}
	}

	double lowVal = -numeric_limits<double>::infinity();
	maxVals.clear();
	maxVals.resize(newSize, lowVal);

	preSum = 1;
	for(int i = 0; i < dimention; i++){
		preSum *= inputs[0]->outDimentions[i];
	}
	postSum = inputs[0]->outSize / (preSum * inputs[0]->outDimentions[dimention]);

	temp.clear();
	temp.resize(newSize, 0.0);

	temp2.clear();
	temp2.resize(newSize, 0.0);
}

void Softmax::derive(vector<double>& seed){
	if (typeid(*inputs[0]) != typeid(Constant)){

		for(int i = 0; i < outSize; i++){
			temp3[i] = seed[i % seedSize] * derivativeMemo[i];
		}

		for(int i = 0; i < preSum; i++){
			for(int z = 0; z < postSum; z++){
				double sum = 0.0;
				for(int x = 0; x < outDimentions[dimention]; x++){
					sum += temp3[i * postSum * outDimentions[dimention] + x * postSum + z];
				}
				sums[i * postSum + z] = sum;
			}
		}

		for(int i = 0; i < preSum; i++){
			for(int x = 0; x < outDimentions[dimention]; x++){
				for(int z = 0; z < postSum; z++){
					ans[i * postSum * outDimentions[dimention] + x * postSum + z] = derivativeMemo[i * postSum * outDimentions[dimention] + x * postSum + z] * (seed[(i * postSum * outDimentions[dimention] + x * postSum + z) % seedSize] - sums[i * postSum + z]);
				}
			}
		}

		inputs[0]->derive(ans);
	}
}

void Softmax::deriveDimentions(vector<int>& seedDimentionsVal){
	seedDimentions = seedDimentionsVal;
	seedRank = seedDimentions.size();
	seedSize = 1;
	for(int i = 0; i < seedRank; i++){
		seedSize *= seedDimentions[i];
	}

	temp3.clear();
	temp3.resize(outSize, 0.0);

	sums.clear();
	sums.resize(newSize, 0.0);

	ans.clear();
	ans.resize(outSize, 0.0);

	inputs[0]->deriveDimentions(outDimentions);
}

double TanH::operation(double a){
	return 2.0 / (1.0 + exp(-2.0 * a)) - 1.0;
}

void TanH::derive(vector<double>& seed){
	if (typeid(*inputs[0]) != typeid(Constant)){
		for(int i = 0; i < outSize; i++){
			ans[i] = seed[i % seedSize] * (1.0 - derivativeMemo[i] * derivativeMemo[i]);
		}
		inputs[0]->derive(ans);
	}
}


