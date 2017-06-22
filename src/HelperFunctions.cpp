#include "HelperFunctions.hpp"
#include <time.h>
#include <cmath>
#include <iostream>
#include <fstream>
#include <iomanip>
#include <random>

Constant getValue(Node* expression){
	expression->getValue();
	return Constant(expression->derivativeMemo, expression->outDimentions);
}

void derive(Node* expression, vector<Variable*>& variables){
	resetDerivative(variables);
	expression->getValue();

	vector<double> seed = {1.0};
	expression->derive(seed);
}

void resetDerivative(vector<Variable*>& variables){
	for(int i = 0; i < variables.size(); i++){
		for(int x = 0; x < variables[i]->outSize; x++){
			variables[i]->derivative[x] = 0.0;
		}
	}
}

void initalize(Node* expression){
	expression->getValueDimentions();
	vector<int> seed = {};
	expression->deriveDimentions(seed);
}

Constant gaussianRandomNums(vector<int> dimentions, double mean, double stdDev){
	srand(time(NULL));

	random_device rd;
    mt19937 gen(rd());
    normal_distribution<double> d(mean, stdDev);

	if (dimentions.size() == 0){
		return Constant(d(gen));
	}

	vector<double> ans;

	int total = 1;
	for(int i = 0; i < dimentions.size(); i++){
		total *= dimentions[i];
	}

	for(int i = 0; i < total; i++){
		ans.push_back(d(gen));
	}
	return Constant(ans, dimentions);
}

Constant trunGaussianRandomNums(vector<int> dimentions, double mean, double stdDev){
	srand(time(NULL));

	random_device rd;
    mt19937 gen(rd());
    normal_distribution<double> d(mean, stdDev);

	if (dimentions.size() == 0){
		double temp = d(gen);
		while (abs(temp - mean) > stdDev * 2){
			temp = d(gen);
		}
		return Constant(temp);
	}

	vector<double> ans;

	int total = 1;
	for(int i = 0; i < dimentions.size(); i++){
		total *= dimentions[i];
	}

	for(int i = 0; i < total; i++){
		double temp = d(gen);
		while (abs(temp - mean) > stdDev * 2){
			temp = d(gen);
		}
		ans.push_back(temp);
	}
	return Constant(ans, dimentions);
}

Constant uniformRandomNums(vector<int> dimentions, double low, double high){
	srand(time(NULL));

	random_device rd;
    mt19937 gen(rd());
  	uniform_real_distribution<double> d(low, high);

	if (dimentions.size() == 0){
		return Constant(d(gen));
	}

	vector<double> ans;

	int total = 1;
	for(int i = 0; i < dimentions.size(); i++){
		total *= dimentions[i];
	}

	for(int i = 0; i < total; i++){
		ans.push_back(d(gen));
	}
	return Constant(ans, dimentions);
}

Constant equal(Constant& a, Constant& b){
	vector<double> ans;
	ans.resize(a.outSize, 0.0);

	for(int i = 0; i < a.outSize; i++){
		if(a.derivativeMemo[i] == b.derivativeMemo[i]){
			ans[i] = 1.0;
		}
	}

	return Constant(ans, a.outDimentions);
}

void numDerive(Node* expression, vector<Variable*>& variables, int n){
	double delta = 0.00001;
	int maxIdx = n;

	expression->getValue();
	vector<double> a = expression->derivativeMemo;

	for(int i = 0; i < variables.size(); i++){
		vector<double> ans;
		ans.reserve(variables[i]->outSize);

		if(n == -1){
			maxIdx = variables[i]->outSize;
		}

		for(int x = 0; x < maxIdx; x++){
			variables[i]->derivativeMemo[x] += delta;
			expression->getValue();
			vector<double> b = expression->derivativeMemo;
			ans.push_back((b[0] - a[0]) / delta);
			variables[i]->derivativeMemo[x] -= delta;
		}

		if(n != -1){
			for(int x = 0; x < (variables[i]->outSize - maxIdx); x++){
				ans.push_back(0.0);
			}
		}

		variables[i]->derivative = ans;
	}
}

vector<Constant> compareDerivatives(Node* expression, vector<Variable*>& variables, int n){
	derive(expression, variables);

	vector<vector<double>> a;
	for(int i = 0; i < variables.size(); i++){
		a.push_back(variables[i]->derivative);
	}

	numDerive(expression, variables, n);

	vector<vector<double>> b;
	for(int i = 0; i < variables.size(); i++){
		b.push_back(variables[i]->derivative);
	}

	vector<Constant> ans;
	for(int i = 0; i < b.size(); i++){
		vector<double> temp;
		temp.reserve(variables[i]->outSize);
		for(int x = 0; x < variables[i]->outSize; x++){
			temp.push_back(abs(a[i][x] - b[i][x]));
		}
		ans.push_back(Constant(temp, variables[i]->outDimentions));
	}
	return ans;
}

Constant oneHot(Constant items, int low, int high){
	vector<int> newDimentions;
	for(int i = 0; i < items.outRank; i++){
		newDimentions.push_back(items.outDimentions[i]);
	}
	newDimentions.push_back(high - low + 1);

	vector<double> ans;
	ans.reserve(items.outSize * (high - low + 1));
	for(int i = 0; i < items.outSize; i++){
		for(int x = 0; x < (high - low) + 1; x++){
			if(items.derivativeMemo[i] == x + low){
				ans.push_back(1.0);
			}
			else{
				ans.push_back(0.0);
			}
		}
	}
	return Constant(ans, newDimentions);
}

void saveData(Constant data, string name){
	ofstream newFile;
	newFile.open(name, ios::out | ios::trunc);

	newFile << data.outRank << endl;

	for(int i = 0; i < data.outRank; i++){
		newFile << data.outDimentions[i] << endl;
	}

	for(int i = 0; i < data.outSize; i++){
		newFile << fixed << setprecision(6) << data.derivativeMemo[i] << endl;
	}

	newFile.close();
}

Constant loadData(string name){
	ifstream newFile;
	newFile.open(name, ios::in);

	int rank;
	vector<int> dimentions;
	vector<double> values;
	int temp1;
	double temp2;

	newFile >> rank;

	for(int i = 0; i < rank; i++){
		newFile >> temp1;
		dimentions.push_back(temp1);
	}

	while(newFile >> temp2){
		values.push_back(temp2);
	}

	newFile.close();

	return Constant(values, dimentions);
}


Gate::Gate(Node* a){
	closed = false;
	inputs.push_back(a);
}

void Gate::getValue(){
	inputs[0]->getValue();
	derivativeMemo = inputs[0]->derivativeMemo;
}

void Gate::getValueDimentions(){
	inputs[0]->getValueDimentions();

	outRank = inputs[0]->outRank;
	outSize = inputs[0]->outSize;
	outDimentions = inputs[0]->outDimentions;
}

void Gate::derive(vector<double>& seed){
	if(closed == false){
		inputs[0]->derive(seed);
	}
}

void Gate::deriveDimentions(vector<int>& seedDimentionsVal){
	seedDimentions = seedDimentionsVal;
	seedRank = seedDimentions.size();
	seedSize = 1;
	for(int i = 0; i < seedRank; i++){
		seedSize *= seedDimentions[i];
	}

	inputs[0]->deriveDimentions(seedDimentions);
}

string Gate::describe(){
	return inputs[0]->describe();
}

