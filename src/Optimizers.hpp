#ifndef OPTIMIZERS_H
#define OPTIMIZERS_H
#include "Node.hpp"

void gradientDescent(vector<Variable*>& variables, float learningRate);

class RMSProp{
public:
	vector<cl::Buffer> residuals;
	float decayRate;
	float learningRate;
	float offset;
	vector<Variable*> variables;

	RMSProp(float LR, float decay, float off, vector<Variable*>& vars);
	void minimize();
};

void clipGradients(vector<Variable*>& variables, float clipMin, float clipMax);

class ClipGradientsNorm{
public:
	vector<cl::Buffer> residuals;
	vector<cl::Buffer> results;
	float clipMagnitude;
	vector<Variable*> variables;
	int GROUP_SIZE;
	vector<int> globalSize;
	vector<int> numGroups;

	ClipGradientsNorm(float clipMag, vector<Variable*>& vars);
	void run();
};

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