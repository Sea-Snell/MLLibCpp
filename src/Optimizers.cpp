#include "Optimizers.hpp"

RMSProp::RMSProp(float LR, float decay, float off, vector<Variable*>& vars){
	learningRate = LR;
	decayRate = decay;
	offset = off;
	variables = vars;
	residuals = {};

	for (int i = 0; i < vars.size(); i++){
		residuals.push_back(cl::Buffer(context, CL_MEM_READ_WRITE, sizeof(float) * variables[i]->seedDims.size));
		NumObject emptyVals = NumObject(variables[i]->seedDims.dimentions, 0.0);
		queue.enqueueWriteBuffer(residuals[i], CL_TRUE, 0, sizeof(float) * variables[i]->seedDims.size, &emptyVals.values[0]);
	}
}

void RMSProp::minimize(){
	for (int i = 0; i < variables.size(); i++){
		RMSResid(cl::EnqueueArgs(queue, cl::NullRange, cl::NDRange(variables[i]->seedDims.size), cl::NullRange), variables[i]->seed, residuals[i], decayRate);
		RMSStep(cl::EnqueueArgs(queue, cl::NullRange, cl::NDRange(variables[i]->resultDims.size), cl::NullRange), variables[i]->seedDims.dimBuf, variables[i]->seed, residuals[i], variables[i]->result[0], learningRate, offset);
	}
}


void gradientDescent(vector<Variable*>& variables, float learningRate){
	for(int i = 0; i < variables.size(); i++){
		gradientDescentStep(cl::EnqueueArgs(queue, cl::NullRange, cl::NDRange(variables[i]->resultDims.size), cl::NullRange), variables[i]->seedDims.dimBuf, variables[i]->seed, variables[i]->result[0], learningRate);
	}
}

void clipGradients(vector<Variable*>& variables, float clipMin, float clipMax){
	for(int i = 0; i < variables.size(); i++){
		clipGrads(cl::EnqueueArgs(queue, cl::NullRange, cl::NDRange(variables[i]->seedDims.size), cl::NullRange), variables[i]->seed, clipMin, clipMax);
	}
}

ClipGradientsNorm::ClipGradientsNorm(float clipMag, vector<Variable*>& vars){
	clipMagnitude = clipMag;
	variables = vars;
	residuals = {};
	results = {};
	GROUP_SIZE = 128;
	numGroups = {};
	globalSize = {};

	for (int i = 0; i < vars.size(); i++){
		numGroups.push_back(vars[i]->seedDims.size / GROUP_SIZE);
		if (numGroups[i] * GROUP_SIZE != vars[i]->seedDims.size){
			numGroups[i] += 1;
		}
		globalSize.push_back(vars[i]->seedDims.size + (GROUP_SIZE - vars[i]->seedDims.size % GROUP_SIZE));
		if (vars[i]->seedDims.size % GROUP_SIZE == 0){
			globalSize[i] -= GROUP_SIZE;
		}
	}

	for (int i = 0; i < vars.size(); i++){
		results.push_back(cl::Buffer(context, CL_MEM_READ_WRITE, sizeof(float) * 1));
		residuals.push_back(cl::Buffer(context, CL_MEM_READ_WRITE, sizeof(float) * numGroups[i]));
	}
}

void ClipGradientsNorm::run(){
	for (int i = 0; i < variables.size(); i++){
		clipGradsNormPt1(cl::EnqueueArgs(queue, cl::NullRange, cl::NDRange(globalSize[i]), cl::NDRange(GROUP_SIZE)), variables[i]->seedDims.dimBuf, variables[i]->seed, residuals[i]);
		clipGradsNormPt2(cl::EnqueueArgs(queue, cl::NullRange, cl::NDRange(1), cl::NullRange), residuals[i], results[i], numGroups[i]);
		clipGradsNormPt3(cl::EnqueueArgs(queue, cl::NullRange, cl::NDRange(variables[i]->seedDims.size), cl::NullRange), results[i], variables[i]->seed, clipMagnitude);
	}
}

// MomentumGradientDescent::MomentumGradientDescent(double LR, double momentum){
// 	learningRate = NumObject(LR);
// 	momentumRate = NumObject(momentum);
// }

// void  MomentumGradientDescent::minimize(vector<Variable*>& variables){
// 	if(velocity.size() != variables.size()){
// 		velocity.clear();
// 		for(int i = 0; i < variables.size(); i++){
// 			vector<NumObject> items = {variables[i]->derivative, learningRate};
// 			velocity.push_back(mapVals(this, &MomentumGradientDescent::operation2, items));
// 			vector<NumObject> items2 = {variables[i]->value, velocity[i]};
// 			variables[i]->value = mapVals(this, &MomentumGradientDescent::operation1, items2);
// 		}
// 	}
// 	else{
// 		for(int i = 0; i < variables.size(); i++){
// 			for(int x = 0; x < variables[i]->derivative.values.size(); x++){
// 				velocity[i].values[x] = momentumRate.values[0] * velocity[i].values[x] + learningRate.values[0] * variables[i]->derivative.values[x];
// 			}
// 			vector<NumObject> items2 = {variables[i]->value, velocity[i]};
// 			variables[i]->value = mapVals(this, &MomentumGradientDescent::operation1, items2);
// 		}
// 	}
// }

// double MomentumGradientDescent::operation1(vector<double>& a){
// 	return a[0] - a[1];
// }

// double MomentumGradientDescent::operation2(vector<double>& a){
// 	return a[0] * a[1];
// }


