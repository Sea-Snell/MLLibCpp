#ifndef HELPERFUNCTIONS_H
#define HELPERFUNCTIONS_H
#include "Node.hpp"

Constant getValue(Node* expression);

void derive(Node* expression, vector<Variable*>& variables);
void resetDerivative(vector<Variable*>& variables);

void initalize(Node* expression);

Constant gaussianRandomNums(vector<int> dimentions, double mean, double stdDev);
Constant trunGaussianRandomNums(vector<int> dimentions, double mean, double stdDev);
Constant uniformRandomNums(vector<int> dimentions, double low, double high);

Constant equal(Constant& a, Constant& b);

void numDerive(Node* expression, vector<Variable*>& variables, int n = -1);
vector<Constant> compareDerivatives(Node* expression, vector<Variable*>& variables, int n = -1);

Constant oneHot(Constant items, int low, int high);

void saveData(Constant data, string name);
Constant loadData(string name);

class Gate: public Node{
public:
	bool closed;

	Gate(Node* a);
	void getValue();
	void getValueDimentions();
	void derive(vector<double>& seed);
	void deriveDimentions(vector<int>& seedDimentionsVal);
	string describe();
};

#endif