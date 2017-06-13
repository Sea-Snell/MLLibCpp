#ifndef REGULARIZATION_H
#define REGULARIZATION_H
#include "Node.hpp"

class L2: public Node{
public:
	double parameter;
	int size;

	L2(Node* cost, vector<Node*> weights, int dataSize, double parameterVal);
	NumObject getValue();
	void derive(NumObject& seed);
	double operation(vector<double>& a);
	string describe();
};

class L1: public L2{
public:
	L1(Node* cost, vector<Node*> weights, int dataSize, double parameterVal): L2(cost, weights, dataSize, parameterVal){name = "L1";}
	NumObject getValue();
	void derive(NumObject& seed);
	double operation(vector<double>& a);
};

void maxNorm(NumObject& weight, int dimention, double c);

class Dropout: public Node{
public:
	int dimention;
	double probability;
	vector<double> dropped;
	bool training;

	Dropout(Node* a, int dimentionVal, double probabilityVal);
	NumObject getValue();
	void updateDrop(int size);
	void derive(NumObject& seed);
	string describe();
};

#endif