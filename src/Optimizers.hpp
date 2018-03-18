#ifndef OPTIMIZERS_H
#define OPTIMIZERS_H
#include "Node.hpp"

void gradientDescent(vector<Variable*>& variables, double learningRate);

// class MomentumGradientDescent{
// public:
// 	NumObject learningRate;
// 	vector<NumObject> velocity;
// 	NumObject momentumRate;

// 	MomentumGradientDescent(double LR, double momentum);
// 	void minimize(vector<Variable*>& variables);
// 	double operation1(vector<double>& a);
// 	double operation2(vector<double>& a);
// };

#endif