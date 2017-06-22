#ifndef OPTIMIZERS_H
#define OPTIMIZERS_H
#include "Node.hpp"

class GradientDescent{
public:
	double learningRate;

	GradientDescent(double LR);
	void minimize(Node* expression, vector<Variable*>& variables, vector<Variable*>& noClearVariables);
};

class MomentumGradientDescent{
public:
	double learningRate;
	vector<vector<double>> velocity;
	double momentumRate;

	MomentumGradientDescent(double LR, double momentum);
	void preprocessVelocity();
	void minimize(Node* expression, vector<Variable*>& variables, vector<Variable*>& noClearVariables);
};

#endif