#ifndef ACTIVATIONS_H
#define ACTIVATIONS_H
#include "Node.hpp"

class Sigmoid: public BasicFunction{
public:
	Sigmoid(Node* a): BasicFunction(a){name = "Sigmoid";}

	double operation(vector<double>& a);
	void derive(NumObject& seed);
	double deriveOperation1(vector<double>& a);
};

class ReLU: public BasicFunction{
public:
	ReLU(Node* a): BasicFunction(a){name = "ReLU";}

	double operation(vector<double>& a);
	void derive(NumObject& seed);
	double deriveOperation1(vector<double>& a);
};

class LeakyReLU: public BasicFunction{
public:
	LeakyReLU(Node* a): BasicFunction(a){name = "LeakyReLU";}

	double operation(vector<double>& a);
	void derive(NumObject& seed);
	double deriveOperation1(vector<double>& a);
};

class Gaussian: public BasicFunction{
public:
	Gaussian(Node* a): BasicFunction(a){name = "Gaussian";}

	double operation(vector<double>& a);
	void derive(NumObject& seed);
	double deriveOperation1(vector<double>& a);
};

class Softmax: public Node{
public:
	int dimention;

	Softmax(Node* a, int dimentionVal = -1);
	
	NumObject getValue();
	double operation1(vector<double>& a);
	void derive(NumObject& seed);
	double deriveOperation1(vector<double>& a);
};

class TanH: public BasicFunction{
public:
	TanH(Node* a): BasicFunction(a){name = "TanH";}

	double operation(vector<double>& a);
	void derive(NumObject& seed);
	double deriveOperation1(vector<double>& a);
};

#endif