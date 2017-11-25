#ifndef HELPERFUNCTIONS_H
#define HELPERFUNCTIONS_H
#include "Node.hpp"

NumObject derive(Node* expression);
NumObject deriveTime(Node* expression, vector<vector<NumObject>>& timeVals, vector<Constant*>& sub);

NumObject getValueTime(Node* expression, vector<vector<NumObject>>& timeVals, vector<Constant*>& sub);

NumObject gaussianRandomNums(vector<int> dimentions, double mean, double stdDev);
NumObject trunGaussianRandomNums(vector<int> dimentions, double mean, double stdDev);
NumObject uniformRandomNums(vector<int> dimentions, double low, double high);

NumObject equal(NumObject& a, NumObject& b);

void numDerive(Node* expression, vector<Variable*>& variables, int n = -1);
void numDeriveTime(Node* expression, vector<vector<NumObject>>& timeVals, vector<Constant*>& sub, vector<Variable*>& variables, int n = -1);
vector<NumObject> compareDerivatives(Node* expression, vector<Variable*>& variables, int n = -1);
vector<NumObject> compareDerivativesTime(Node* expression, vector<vector<NumObject>>& timeVals, vector<Constant*>& sub, vector<Variable*>& variables, int n = -1);

NumObject oneHot(NumObject items, int low, int high);

void saveData(NumObject data, string name);
NumObject loadData(string name);


class Set: public Node{
public:
	Set(Node* source, Variable* goal);
	NumObject getValue(int t = 0, int tf = 0);
	void derive(NumObject& seed, int t = 0, int tf = 0);
	string describe();
};


// class Gate: public Node{
// public:
// 	bool closed;

// 	Gate(Node* a);
// 	NumObject getValue(int t = 0, int tf = 0);
// 	void derive(NumObject& seed, int t = 0, int tf = 0);
// 	string describe();
// 	void closeGate();
// 	void closeGatePropagate(Node* a);
// 	void openGate();
// 	void openGatePropagate(Node* a);
// };

#endif