#include "HelperFunctions.hpp"
#include <time.h>
#include <cmath>
#include <iostream>
#include <fstream>
#include <iomanip>
#include <random>

void derive(Node* expression, vector<Variable*>& variables){
	resetDerivative(variables);
	expression->getValue();

	NumObject seed = NumObject(1.0);
	expression->derive(seed);
}

void resetDerivative(vector<Variable*>& variables){
	for(int i = 0; i < variables.size(); i++){
		variables[i]->derivative = NumObject(variables[i]->derivative.dimentions, 0.0);
	}
}

NumObject randomNums(vector<int> dimentions, double mean, double stdDev){
	srand(time(NULL));

	random_device rd;
    mt19937 gen(rd());
    normal_distribution<double> d(mean, stdDev);

	if (dimentions.size() == 0){
		return d(gen);
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

NumObject equal(NumObject& a, NumObject& b){
	NumObject ans = NumObject(a.dimentions, 0);
	for(int i = 0; i < a.values.size(); i++){
		if (a.values[i] == b.values[i]){
			ans.values[i] = 1;
		}
	}
	return ans;
}

void numDerive(Node* expression, vector<Variable*>& variables, int n){
	double delta = 0.00001;
	int maxIdx = n;

	NumObject a = expression->getValue();

	for(int i = 0; i < variables.size(); i++){
		NumObject ans = NumObject(variables[i]->value.dimentions);

		if(n == -1){
			maxIdx = variables[i]->value.values.size();
		}

		for(int x = 0; x < maxIdx; x++){
			variables[i]->value.values[x] += delta;
			NumObject b = expression->getValue();
			ans.values.push_back((b.values[0] - a.values[0]) / delta);
			variables[i]->derivative = ans;
			variables[i]->value.values[x] -= delta;
		}

		if(n != -1){
			for(int x = 0; x < (variables[i]->value.values.size() - maxIdx); x++){
				ans.values.push_back(0.0);
			}
		}

		variables[i]->derivative = ans;
	}
}

vector<NumObject> compareDerivatives(Node* expression, vector<Variable*>& variables, int n){
	derive(expression, variables);

	vector<NumObject> a;
	for(int i = 0; i < variables.size(); i++){
		a.push_back(variables[i]->derivative);
	}

	numDerive(expression, variables, n);

	vector<NumObject> b;
	for(int i = 0; i < variables.size(); i++){
		b.push_back(variables[i]->derivative);
	}

	vector<NumObject> ans;
	for(int i = 0; i < b.size(); i++){
		NumObject temp = NumObject(b[i].dimentions);
		for(int x = 0; x < b[i].values.size(); x++){
			temp.values.push_back(abs(a[i].values[x] - b[i].values[x]));
		}
		ans.push_back(temp);
	}
	return ans;
}

NumObject oneHot(NumObject items, int low, int high){
	NumObject ans = NumObject(vector<int>{items.dimentions[0], (high - low) + 1});
	for(int i = 0; i < items.values.size(); i++){
		for(int x = 0; x < (high - low) + 1; x++){
			if(items.values[i] == x + low){
				ans.values.push_back(1.0);
			}
			else{
				ans.values.push_back(0.0);
			}
		}
	}
	return ans;
}

void saveData(NumObject data, string name){
	ofstream newFile;
	newFile.open(name, ios::out | ios::trunc);

	newFile << data.rank << endl;

	for(int i = 0; i < data.rank; i++){
		newFile << data.dimentions[i] << endl;
	}

	for(int i = 0; i < data.values.size(); i++){
		newFile << fixed << setprecision(6) << data.values[i] << endl;
	}

	newFile.close();
}

NumObject loadData(string name){
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

	return NumObject(values, dimentions);
}

