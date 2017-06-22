#include "Regularization.hpp"
#include <random>


L2::L2(Node* cost, vector<Node*> weights, int dataSize, double parameterVal){
	inputs.push_back(cost);
	for(int i = 0; i < weights.size(); i++){
		inputs.push_back(weights[i]);
	}
	name = "L2";
	parameter = parameterVal;
	size = dataSize;
}

void L2::getValue(){
	for(int i = 0; i < inputSize; i++){
		inputs[i]->getValue();
	}


	if(parameter == 0.0){
		derivativeMemo[0] = inputs[0]->derivativeMemo[0];
	}

	double sum = 0.0;
	for(int i = 1; i < inputSize; i++){
		for(int x = 0; x < inputs[i]->outSize; x++){
			sum += inputs[i]->derivativeMemo[x] * inputs[i]->derivativeMemo[x];
		}
	}

	sum *= parameter / (2.0 * size);

	derivativeMemo[0] = sum + inputs[0]->derivativeMemo[0];
}

void L2::getValueDimentions(){
	inputs[0]->getValueDimentions();

	outDimentions = {};
	outRank = 0;
	outSize = 1;

	derivativeMemo.clear();
	derivativeMemo.resize(1, 0.0);

	inputSize = inputs.size();

}

void L2::derive(vector<double>& seed){
	inputs[0]->derive(seed);

	if(parameter != 0.0){	
		double temp1 = parameter / size;
		for(int i = 1; i < inputs.size(); i++){
			for(int x = 0; x < inputs[i]->outSize; x++){
				ans[i][x] = inputs[i]->derivativeMemo[x] * seed[0] * temp1;
			}
			inputs[i]->derive(ans[i]);
		}
	}
}

void L2::deriveDimentions(vector<int>& seedDimentionsVal){
	seedDimentions = seedDimentionsVal;
	seedRank = seedDimentions.size();
	seedSize = 1;
	for(int i = 0; i < seedRank; i++){
		seedSize *= seedDimentions[i];
	}

	inputs[0]->deriveDimentions(seedDimentions);

	ans.clear();
	for(int i = 1; i < inputSize; i++){
		vector<double> temp;
		temp.resize(inputs[i]->outSize, 0.0);
		ans.push_back(temp);
		inputs[i]->deriveDimentions(inputs[i]->outDimentions);
	}
}

string L2::describe(){
	string ans = name + "(";
	for(int i = 0; i < inputs.size(); i++){
		ans += inputs[i]->describe();
		ans += ", ";
	}
	ans += to_string(size) + ", " + to_string(parameter) + ")";
	return ans;
}

void L1::getValue(){
	for(int i = 0; i < inputSize; i++){
		inputs[i]->getValue();
	}


	if(parameter == 0.0){
		derivativeMemo[0] = inputs[0]->derivativeMemo[0];
	}

	double sum = 0.0;
	for(int i = 1; i < inputSize; i++){
		for(int x = 0; x < inputs[i]->outSize; x++){
			sum += abs(inputs[i]->derivativeMemo[x]);
		}
	}

	sum *= parameter / (2.0 * size);

	derivativeMemo[0] = sum + inputs[0]->derivativeMemo[0];
}

void L1::derive(vector<double>& seed){
	inputs[0]->derive(seed);

	if(parameter != 0.0){	
		double temp1 = parameter / size;
		for(int i = 1; i < inputs.size(); i++){
			for(int x = 0; x < inputs[i]->outSize; x++){
				if(inputs[i]->derivativeMemo[x] > 0){
					ans[i][x] = seed[0] * temp1;
				}
				else if(inputs[i]->derivativeMemo[x] < 0){
					ans[i][x] = -seed[0] * temp1;
				}
				else{
					ans[i][x] = 0.0;
				}
			}
			inputs[i]->derive(ans[i]);
		}
	}
}

void maxNorm(Constant& weight, int dimention, double c){
	int preSum = 1;
	for(int i = 0; i < dimention; i++){
		preSum *= weight.outDimentions[i];
	}
	int postSum = weight.outSize / (preSum * weight.outDimentions[dimention]);

	for(int i = 0; i < preSum; i++){
		for(int z = 0; z < postSum; z++){
			double length = 0.0;
			for(int x = 0; x < weight.outDimentions[dimention]; x++){
				int idx = i * postSum * weight.outDimentions[dimention] + x * postSum + z;
				length += weight.derivativeMemo[idx] * weight.derivativeMemo[idx];
			}
			if(length > c * c){
				for(int x = 0; x < weight.outDimentions[dimention]; x++){
					int idx = i * postSum * weight.outDimentions[dimention] + x * postSum + z;
					weight.derivativeMemo[idx] = c * (weight.derivativeMemo[idx] / sqrt(length));
				}
			}
		}
	}
}


Dropout::Dropout(Node* a, int dimentionVal, double probabilityVal){
	inputs.push_back(a);
	dimention = dimentionVal;
	probability = probabilityVal;
	name = "Dropout";
	training = true;
}

void Dropout::getValue(){
	inputs[0]->getValue();

	if(training == true){
		updateDrop(inputs[0]->outDimentions[dimention]);

		for(int i = 0; i < preSum; i++){
			for(int x = 0; x < inputs[0]->outDimentions[dimention]; x++){
				for(int z = 0; z < postSum; z++){
					int tempIdx = i * postSum * inputs[0]->outDimentions[dimention] + x * postSum + z;
					derivativeMemo[tempIdx] = inputs[0]->derivativeMemo[tempIdx] * dropped[x];
				}
			}
		}
	}
	else{
		for(int i = 0; i < outSize; i++){
			derivativeMemo[i] = inputs[0]->derivativeMemo[i] * probability;
		}
	}
}

void Dropout::getValueDimentions(){
	inputs[0]->getValueDimentions();

	outSize = inputs[0]->outSize;
	outDimentions = inputs[0]->outDimentions;
	outRank = inputs[0]->outRank;

	preSum = 1;
	for(int i = 0; i < dimention; i++){
		preSum *= inputs[0]->outDimentions[i];
	}
	postSum = outSize / (preSum * inputs[0]->outDimentions[dimention]);

	derivativeMemo.clear();
	derivativeMemo.resize(outSize, 0.0);
}

void Dropout::updateDrop(int size){
	srand(time(NULL));
  	random_device rd;
    mt19937 gen(rd());
    uniform_real_distribution<double> distribution(0.0, 1.0);

	dropped.clear();
	for(int i = 0; i < size; i++){
		if(distribution(gen) > probability){
			dropped.push_back(1.0);
		}
		else{
			dropped.push_back(0.0);
		}
	}
}

void Dropout::derive(vector<double>& seed){
	if(seedRank < (inputs[0]->outRank - dimention)){
		for(int x = 0; x < outDimentions[dimention]; x++){
			for(int z = 0; z < postSum; z++){
				ans[x * postSum + z] = seed[(x * postSum + z) % seedSize] * dropped[x];
			}
		}
	}
	else{
		for(int i = 0; i < preSum1; i++){
			for(int x = 0; x < seedDimentions[dimention - (outRank - seedRank)]; x++){
				for(int z = 0; z < postSum1; z++){
					ans[i * postSum * seedDimentions[dimention - (outRank - seedRank)] + x * postSum + z] = seed[i * postSum * seedDimentions[dimention - (outRank - seedRank)] + x * postSum + z] * dropped[x];
				}
			}
		}
	}

	inputs[0]->derive(ans);
}

void Dropout::deriveDimentions(vector<int>& seedDimentionsVal){
	seedDimentions = seedDimentionsVal;
	seedRank = seedDimentions.size();
	seedSize = 1;
	for(int i = 0; i < seedRank; i++){
		seedSize *= seedDimentions[i];
	}

	if(seedRank < (inputs[0]->outRank - dimention)){
		ans.clear();
		ans.resize(outSize / preSum, 0.0);

		vector<int> tempDimentions;
		for(int i = dimention; i < outRank; i++){
			tempDimentions.push_back(outDimentions[i]);
		}

		preSum1 = 1;
		for(int i = 0; i < (dimention - (outRank - seedRank)); i++){
			preSum1 *= seedDimentions[i];
		}
		postSum1 = seedSize / (preSum * seedDimentions[dimention - (outRank - seedRank)]);


		inputs[0]->deriveDimentions(tempDimentions);
	}
	else{
		ans.clear();
		ans.resize(outSize, 0.0);
		inputs[0]->deriveDimentions(outDimentions);
	}
}

string Dropout::describe(){
	return name + "(" + inputs[0]->describe() + ", " + to_string(dimention) + ", " + to_string(probability) + ")";
}

