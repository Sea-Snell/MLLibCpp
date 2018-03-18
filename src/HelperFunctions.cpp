#include "HelperFunctions.hpp"
#include <time.h>
#include <cmath>
#include <iostream>
#include <fstream>
#include <iomanip>
#include <random>

NumObject getValue(Node* expression){
	int tempOutCount = expression->outCount;
	expression->outCount = 1;
	expression->getValue();
	NumObject result = NumObject(expression->resultDims.dimentions, 0.0);
	queue.enqueueReadBuffer(expression->result, CL_TRUE, 0, sizeof(float) * expression->resultDims.size, &result.values[0]);
	expression->outCount = tempOutCount;
	return result;
}

void initalize(Node* expression){
	int tempOutCount = expression->outCount;
	expression->outCount = 1;
	expression->clean();
	expression->getDimentions();
	GPUDimentions initialDims = GPUDimentions(0, 1, vector<int>{});
	expression->deriveDimentions(&initialDims);
	expression->outCount = tempOutCount;

	float tempVal = 1.0;
	queue.enqueueWriteBuffer(expression->seed, CL_TRUE, 0, sizeof(float) * 1, &tempVal);
}

void derive(Node* expression){
	int tempOutCount = expression->outCount;
	expression->outCount = 1;
	expression->getValue();
	expression->derive();
	expression->outCount = tempOutCount;
}

void clearHistory(Node* expression){
	expression->outCount = 0;
	expression->outputs = {};
}

NumObject showSeed(Node* expression){
	NumObject seed = NumObject(expression->seedDims.dimentions, 0.0);
	queue.enqueueReadBuffer(expression->seed, CL_TRUE, 0, sizeof(float) * expression->seedDims.size, &seed.values[0]);
	return seed;
}

NumObject showValue(Node* expression){
	NumObject result = NumObject(expression->resultDims.dimentions, 0.0);
	queue.enqueueReadBuffer(expression->result, CL_TRUE, 0, sizeof(float) * expression->resultDims.size, &result.values[0]);
	return result;
}

// NumObject getValueTime(Node* expression, vector<vector<NumObject>>& timeVals, vector<Constant*>& sub){
// 	NumObject costTotal;
// 	int timeFinal = timeVals[0].size() - 1;
// 	for(int i = 0; i <= timeFinal; i++){
// 		for(int x = 0; x < timeVals.size(); x++){
// 			sub[x]->value = timeVals[x][i];
// 		}
// 		NumObject cost = expression->getValue(i, timeFinal);

// 		if(i == 0){
// 			costTotal = cost;
// 		}
// 		else{
// 			for(int x = 0; x < costTotal.values.size(); x++){
// 				costTotal.values[x] += cost.values[x];
// 			}
// 		}
// 	}

// 	for(int x = 0; x < costTotal.values.size(); x++){
// 		costTotal.values[x] /= (timeFinal + 1.0);
// 	}

// 	return costTotal;
// }

// NumObject deriveTime(Node* expression, vector<vector<NumObject>>& timeVals, vector<Constant*>& sub){
// 	NumObject costTotal;
// 	int timeFinal = timeVals[0].size() - 1;
// 	for(int i = 0; i <= timeFinal; i++){
// 		for(int x = 0; x < timeVals.size(); x++){
// 			sub[x]->value = timeVals[x][i];
// 		}
// 		NumObject cost = expression->getValue(i, timeFinal);

// 		if(i == 0){
// 			costTotal = cost;
// 		}
// 		else{
// 			for(int x = 0; x < costTotal.values.size(); x++){
// 				costTotal.values[x] += cost.values[x];
// 			}
// 		}
// 	}

// 	for(int x = 0; x < costTotal.values.size(); x++){
// 		costTotal.values[x] /= (timeFinal + 1.0);
// 	}

// 	for(int i = timeFinal; i >= 0 ; i--){
// 		NumObject seed = NumObject(1.0 / (timeFinal + 1.0));
// 		expression->derive(seed, i, timeFinal);
// 	}
// 	return costTotal;
// }

NumObject gaussianRandomNums(vector<int> dimentions, double mean, double stdDev){
	srand(time(NULL));

	random_device rd;
    mt19937 gen(rd());
    normal_distribution<double> d(mean, stdDev);

	if (dimentions.size() == 0){
		return NumObject(d(gen));
	}

	NumObject ans = NumObject(dimentions);

	int total = 1;
	for(int i = 0; i < dimentions.size(); i++){
		total *= dimentions[i];
	}

	for(int i = 0; i < total; i++){
		ans.values.push_back(d(gen));
	}
	return ans;
}

NumObject trunGaussianRandomNums(vector<int> dimentions, double mean, double stdDev){
	srand(time(NULL));

	random_device rd;
    mt19937 gen(rd());
    normal_distribution<double> d(mean, stdDev);

	if (dimentions.size() == 0){
		double temp = d(gen);
		while (abs(temp - mean) > stdDev * 2){
			temp = d(gen);
		}
		return NumObject(temp);
	}

	NumObject ans = NumObject(dimentions);

	int total = 1;
	for(int i = 0; i < dimentions.size(); i++){
		total *= dimentions[i];
	}

	for(int i = 0; i < total; i++){
		double temp = d(gen);
		while (abs(temp - mean) > stdDev * 2){
			temp = d(gen);
		}
		ans.values.push_back(temp);
	}
	return ans;
}

NumObject uniformRandomNums(vector<int> dimentions, double low, double high){
	srand(time(NULL));

	random_device rd;
    mt19937 gen(rd());
  	uniform_real_distribution<double> d(low, high);

	if (dimentions.size() == 0){
		return NumObject(d(gen));
	}

	NumObject ans = NumObject(dimentions);

	int total = 1;
	for(int i = 0; i < dimentions.size(); i++){
		total *= dimentions[i];
	}

	for(int i = 0; i < total; i++){
		ans.values.push_back(d(gen));
	}
	return ans;
}

// NumObject equal(NumObject& a, NumObject& b){
// 	NumObject ans = NumObject(a.dimentions, 0);
// 	for(int i = 0; i < a.values.size(); i++){
// 		if (a.values[i] == b.values[i]){
// 			ans.values[i] = 1;
// 		}
// 	}
// 	return ans;
// }

// float compareDerivatives(Node* expression, Variable* variable, float delta){
// 	derive(expression);
// 	vector<float> baseDeriv = showSeed(variable).values;


// 	float diff = 0.0;
// 	float sumVal = 0.0;

// 	for (int i = 0; i < variable->resultDims.size; i++){
// 		variable->value.values[i] += delta;
// 		initalize(expression);
// 		float val1 = getValue(expression).values[0];


// 		variable->value.values[i] -= 2.0 * delta;
// 		initalize(expression);
// 		float val2 = getValue(expression).values[0];

// 		float approxDeriv = (val1 - val2) / (2 * delta);
// 		diff += (approxDeriv - baseDeriv[i]) * (approxDeriv - baseDeriv[i]);
// 		sumVal += (approxDeriv + baseDeriv[i]) * (approxDeriv + baseDeriv[i]);

// 		variable->value.values[i] += delta;
// 		initalize(expression);
// 	}

// 	return sqrt(diff) / sqrt(sumVal);
// }

// vector<NumObject> compareDerivativesTime(Node* expression, vector<vector<NumObject>>& timeVals, vector<Constant*>& sub, vector<Variable*>& variables, int n){
// 	deriveTime(expression, timeVals, sub);

// 	vector<NumObject> a;
// 	for(int i = 0; i < variables.size(); i++){
// 		a.push_back(variables[i]->derivative);
// 	}

// 	numDeriveTime(expression, timeVals, sub, variables, n);

// 	vector<NumObject> b;
// 	for(int i = 0; i < variables.size(); i++){
// 		b.push_back(variables[i]->derivative);
// 	}

// 	vector<NumObject> ans;
// 	for(int i = 0; i < b.size(); i++){
// 		NumObject temp = NumObject(b[i].dimentions);
// 		for(int x = 0; x < b[i].values.size(); x++){
// 			temp.values.push_back(abs(a[i].values[x] - b[i].values[x]));
// 		}
// 		ans.push_back(temp);
// 	}
// 	return ans;
// }

// NumObject oneHot(NumObject items, int low, int high){
// 	NumObject ans = NumObject(vector<int>{items.dimentions[0], (high - low) + 1});
// 	for(int i = 0; i < items.values.size(); i++){
// 		for(int x = 0; x < (high - low) + 1; x++){
// 			if(items.values[i] == x + low){
// 				ans.values.push_back(1.0);
// 			}
// 			else{
// 				ans.values.push_back(0.0);
// 			}
// 		}
// 	}
// 	return ans;
// }

// void saveData(NumObject data, string name){
// 	ofstream newFile;
// 	newFile.open(name, ios::out | ios::trunc);

// 	newFile << data.rank << endl;

// 	for(int i = 0; i < data.rank; i++){
// 		newFile << data.dimentions[i] << endl;
// 	}

// 	for(int i = 0; i < data.values.size(); i++){
// 		newFile << fixed << setprecision(6) << data.values[i] << endl;
// 	}

// 	newFile.close();
// }

// NumObject loadData(string name){
// 	ifstream newFile;
// 	newFile.open(name, ios::in);

// 	int rank;
// 	vector<int> dimentions;
// 	vector<double> values;
// 	int temp1;
// 	double temp2;

// 	newFile >> rank;

// 	for(int i = 0; i < rank; i++){
// 		newFile >> temp1;
// 		dimentions.push_back(temp1);
// 	}

// 	while(newFile >> temp2){
// 		values.push_back(temp2);
// 	}

// 	newFile.close();

// 	return NumObject(values, dimentions);
// }



// Set::Set(Node* source, Variable* goal){
// 	outCount = 0;
// 	inputs.push_back(source);
// 	inputs.push_back(goal);
// 	source->outCount += 1;
// 	goal->outCount += 1;
// 	name = "->";
// 	dCallCount = 0;
// 	gCallCount = 0;
// }

// NumObject Set::getValue(int t, int tf){
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

// 	NumObject ans = inputs[0]->getValue(t, tf);
// 	inputs[1]->getValue(t, tf);
// 	dynamic_cast<Variable*>(inputs[1])->value = ans;
// 	return memoize(ans, t, tf);
// }

// void Set::derive(NumObject& seed, int t, int tf){
// 	if(sumSeed(seed)){
// 		inputs[1]->derive(tempSeed, t, tf);
// 		inputs[0]->derive(dynamic_cast<Variable*>(inputs[1])->derivative, t, tf);
// 	}
// }

// string Set::describe(){
// 	return "(" + inputs[0]->describe() + " " + name + " " + inputs[1]->describe() + ")";
// }



// Gate::Gate(Node* a){
// 	closed = false;
// 	outCount = 0;
// 	inputs.push_back(a);
// 	a->outCount += 1;
// 	dCallCount = 0;
// }

// NumObject Gate::getValue(int t, int tf){
// 	NumObject a = inputs[0]->getValue(t, tf);
// 	return memoize(a, t, tf);
// }

// void Gate::derive(NumObject& seed, int t, int tf){
// 	if(closed == false){
// 		if(sumSeed(seed)){
// 			inputs[0]->derive(tempSeed, t, tf);
// 		}
// 	}
// }

// string Gate::describe(){
// 	return inputs[0]->describe();
// }

// void Gate::closeGate(){
// 	if(closed == false){
// 		closed = true;
// 		inputs[0]->outCount -= 1;
// 	}
// }

// void Gate::closeGatePropagate(Node* a){
// 	a->outCount -= 1
// 	for(int i = 0; i < a->inputs.size(); i++){
// 		closeGatePropagate(a->inputs[i]);
// 	}
// }

// void Gate::openGate(){
// 	if(closed == true){
// 		closed = false;
// 		inputs[0]->outCount += 1;
// 	}
// }


