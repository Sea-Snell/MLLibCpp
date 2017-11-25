#include "Regularization.hpp"
#include "MapVals.hpp"
#include <random>


L2::L2(Node* cost, vector<Node*> weights, int dataSize, double parameterVal){
	outCount = 0;
	inputs.push_back(cost);
	cost->outCount += 1;
	for(int i = 0; i < weights.size(); i++){
		inputs.push_back(weights[i]);
		weights[i]->outCount += 1;
	}
	name = "L2";
	parameter = parameterVal;
	size = dataSize;
	dCallCount = 0;
	gCallCount = 0;
}

NumObject L2::getValue(int t, int tf){
	gCallCount += 1;
	if(gCallCount > 1){
		if(gCallCount >= outCount){
			gCallCount = 0;
		}
		return derivativeMemo[t];
	}
	if(gCallCount >= outCount){
		gCallCount = 0;
	}

	NumObject cost = inputs[0]->getValue(t, tf);

	if(parameter == 0.0){
		return memoize(cost, t, tf);
	}

	vector<NumObject> weights;
	for(int i = 1; i < inputs.size(); i++){
		weights.push_back(inputs[i]->getValue(t, tf));
	}

	double sum = 0.0;
	for(int i = 0; i < weights.size(); i++){
		for(int x = 0; x < weights[i].values.size(); x++){
			sum += weights[i].values[x] * weights[i].values[x];
		}
	}

	sum *= parameter / (2.0 * size);

	NumObject ans = NumObject(sum + cost.values[0]);
	return memoize(ans, t, tf);
}

void L2::derive(NumObject& seed, int t, int tf){
	if(sumSeed(seed)){
		inputs[0]->derive(tempSeed, t, tf);

		if(parameter != 0.0){	
			for(int i = 1; i < inputs.size(); i++){
				vector<NumObject> items = {inputs[i]->derivativeMemo[t], tempSeed, parameter / size};
				NumObject ans = mapVals(this, &L2::operation, items);

				inputs[i]->derive(ans, t, tf);
			}
		}
	}
}

double L2::operation(vector<double>& a){
	return a[0] * a[1] * a[2];
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

NumObject L1::getValue(int t, int tf){
	gCallCount += 1;
	if(gCallCount > 1){
		if(gCallCount >= outCount){
			gCallCount = 0;
		}
		return derivativeMemo[t];
	}
	if(gCallCount >= outCount){
		gCallCount = 0;
	}

	NumObject cost = inputs[0]->getValue(t, tf);

	if(parameter == 0.0){
		return memoize(cost, t, tf);
	}

	vector<NumObject> weights;
	for(int i = 1; i < inputs.size(); i++){
		weights.push_back(inputs[i]->getValue());
	}

	double sum = 0.0;
	for(int i = 0; i < weights.size(); i++){
		for(int x = 0; x < weights[i].values.size(); x++){
			sum += abs(weights[i].values[x]);
		}
	}

	sum *= parameter / (2.0 * size);

	NumObject ans = NumObject(sum + cost.values[0]);
	return memoize(ans, t, tf);
}

void L1::derive(NumObject& seed, int t, int tf){
	if(sumSeed(seed)){
		inputs[0]->derive(tempSeed, t, tf);

		if(parameter != 0.0){	
			for(int i = 1; i < inputs.size(); i++){
				vector<NumObject> items = {inputs[i]->derivativeMemo[t], tempSeed, parameter / size};
				NumObject ans = mapVals(this, &L1::operation, items);

				inputs[i]->derive(ans, t, tf);
			}
		}
	}
}

double L1::operation(vector<double>& a){
	if(a[0] > 0){
		return a[1] * a[2];
	}
	if(a[0] < 0){
		return -a[1] * a[2];
	}
	return 0.0;
}

void maxNorm(NumObject& weight, int dimention, double c){
	int preSum = 1;
	for(int i = 0; i < dimention; i++){
		preSum *= weight.dimentions[i];
	}
	int postSum = weight.values.size() / (preSum * weight.dimentions[dimention]);

	for(int i = 0; i < preSum; i++){
		for(int z = 0; z < postSum; z++){
			double length = 0.0;
			for(int x = 0; x < weight.dimentions[dimention]; x++){
				int idx = i * postSum * weight.dimentions[dimention] + x * postSum + z;
				length += weight.values[idx] * weight.values[idx];
			}
			if(length > c * c){
				for(int x = 0; x < weight.dimentions[dimention]; x++){
					int idx = i * postSum * weight.dimentions[dimention] + x * postSum + z;
					weight.values[idx] = c * (weight.values[idx] / sqrt(length));
				}
			}
		}
	}
}


Dropout::Dropout(Node* a, int dimentionVal, double probabilityVal){
	outCount = 0;
	inputs.push_back(a);
	a->outCount += 1;
	dimention = dimentionVal;
	probability = probabilityVal;
	name = "Dropout";
	training = true;
	dCallCount = 0;
	gCallCount = 0;
}

NumObject Dropout::getValue(int t, int tf){
	gCallCount += 1;
	if(gCallCount > 1){
		if(gCallCount >= outCount){
			gCallCount = 0;
		}
		return derivativeMemo[t];
	}
	if(gCallCount >= outCount){
		gCallCount = 0;
	}

	NumObject a = inputs[0]->getValue(t, tf);

	if(training == true){
		if(t == 0){
			updateDrop(a.dimentions[dimention]);
		}
	}
	else{
		NumObject ans = NumObject(a.dimentions);
		for(int i = 0; i < a.values.size(); i++){
			ans.values.push_back(a.values[i] * probability);
		}
		return memoize(ans, t, tf);
	}

	int preSum = 1;
	for(int i = 0; i < dimention; i++){
		preSum *= a.dimentions[i];
	}
	int postSum = a.values.size() / (preSum * a.dimentions[dimention]);

	NumObject ans = NumObject(a.dimentions);
	for(int i = 0; i < preSum; i++){
		for(int x = 0; x < a.dimentions[dimention]; x++){
			for(int z = 0; z < postSum; z++){
				ans.values.push_back(a.values[i * postSum * a.dimentions[dimention] + x * postSum + z] * dropped[x]);
			}
		}
	}

	return memoize(ans, t, tf);
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

void Dropout::derive(NumObject& seed, int t, int tf){
	if(sumSeed(seed)){
		NumObject a = inputs[0]->derivativeMemo[t];

		if(tempSeed.rank < (a.rank - dimention)){
			int postSum = 1;
			vector<int> newDimentions;
			newDimentions.push_back(a.dimentions[dimention]);
			for(int i = dimention + 1; i < a.rank; i++){
				postSum *= a.dimentions[i];
				newDimentions.push_back(a.dimentions[i]);
			}

			NumObject ans = NumObject(newDimentions);
			for(int x = 0; x < a.dimentions[dimention]; x++){
				for(int z = 0; z < postSum; z++){
					ans.values.push_back(tempSeed.values[(x * postSum + z) % tempSeed.values.size()] * dropped[x]);
				}
			}

			inputs[0]->derive(ans, t, tf);
		}
		else{
			int preSum = 1;
			for(int i = 0; i < (dimention - (a.rank - tempSeed.rank)); i++){
				preSum *= tempSeed.dimentions[i];
			}
			int postSum = tempSeed.values.size() / (preSum * tempSeed.dimentions[dimention - (a.rank - tempSeed.rank)]);

			NumObject ans = NumObject(tempSeed.dimentions);
			for(int i = 0; i < preSum; i++){
				for(int x = 0; x < tempSeed.dimentions[dimention - (a.rank - tempSeed.rank)]; x++){
					for(int z = 0; z < postSum; z++){
						ans.values.push_back(tempSeed.values[i * postSum * tempSeed.dimentions[dimention - (a.rank - seed.rank)] + x * postSum + z] * dropped[x]);
					}
				}
			}

			inputs[0]->derive(ans, t, tf);
		}
	}
}

string Dropout::describe(){
	return name + "(" + inputs[0]->describe() + ", " + to_string(dimention) + ", " + to_string(probability) + ")";
}

