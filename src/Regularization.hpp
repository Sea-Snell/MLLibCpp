#ifndef REGULARIZATION_H
#define REGULARIZATION_H
#include "Node.hpp"

class L2: public Node{
public:
	double parameter;
	int size;

	int inputSize;

	vector<vector<double>> ans;

	L2(Node* cost, vector<Node*> weights, int dataSize, double parameterVal);
	void getValue();
	void getValueDimentions();
	void derive(vector<double>& seed);
	void deriveDimentions(vector<int>& seedDimentionsVal);
	string describe();
};

class L1: public L2{
public:
	L1(Node* cost, vector<Node*> weights, int dataSize, double parameterVal): L2(cost, weights, dataSize, parameterVal){name = "L1";}
	void getValue();
	void derive(vector<double>& seed);
};

void maxNorm(Constant& weight, int dimention, double c);

class Dropout: public Node{
public:
	int dimention;
	double probability;
	vector<double> dropped;
	bool training;

	int preSum;
	int postSum;

	vector<double> ans;

	int preSum1;
	int postSum1;

	Dropout(Node* a, int dimentionVal, double probabilityVal);
	void getValue();
	void getValueDimentions();
	void updateDrop(int size);
	void derive(vector<double>& seed);
	void deriveDimentions(vector<int>& seedDimentionsVal);
	string describe();
};

#endif