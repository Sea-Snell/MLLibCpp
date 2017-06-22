#ifndef ACTIVATIONS_H
#define ACTIVATIONS_H
#include "Node.hpp"

class Sigmoid: public BasicFunction{
public:
	vector<double> ans;

	Sigmoid(Node* a): BasicFunction(a){name = "Sigmoid";}

	double operation(double a);
	void derive(vector<double>& seed);
	void deriveDimentions(vector<int>& seedDimentionsVal);
};

class ReLU: public Sigmoid{
public:

	ReLU(Node* a): Sigmoid(a){name = "ReLU";}

	double operation(double a);
	void derive(vector<double>& seed);
};

class LeakyReLU: public Sigmoid{
public:

	LeakyReLU(Node* a): Sigmoid(a){name = "LeakyReLU";}

	double operation(double a);
	void derive(vector<double>& seed);
};

class Gaussian: public Sigmoid{
public:
	Gaussian(Node* a): Sigmoid(a){name = "Gaussian";}

	double operation(double a);
	void derive(vector<double>& seed);
};

class Softmax: public Node{
public:
	int dimention;

	vector<int> newDimentions;
	vector<double> maxVals;
	int newSize;

	int preSum;
	int postSum;

	vector<double> temp;
	vector<double> temp2;

	vector<double> temp3;
	vector<double> sums;

	vector<double> ans;

	Softmax(Node* a, int dimentionVal = -1);
	
	void getValue();
	void getValueDimentions();
	void derive(vector<double>& seed);
	void deriveDimentions(vector<int>& seedDimentionsVal);
};

class TanH: public Sigmoid{
public:
	TanH(Node* a): Sigmoid(a){name = "TanH";}

	double operation(double a);
	void derive(vector<double>& seed);
};

#endif