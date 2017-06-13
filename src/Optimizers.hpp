#ifndef OPTIMIZERS_H
#define OPTIMIZERS_H
#include "Node.hpp"

class GradientDescent{
public:
	NumObject learningRate;

	GradientDescent(double LR);
	void minimize(Node* expression, vector<Variable*>& variables);
	double operation(vector<double>& a);
};

class MomentumGradientDescent{
public:
	NumObject learningRate;
	vector<NumObject> velocity;
	NumObject momentumRate;

	MomentumGradientDescent(double LR, double momentum);
	void minimize(Node* expression, vector<Variable*>& variables);
	double operation1(vector<double>& a);
	double operation2(vector<double>& a);
};

#endif