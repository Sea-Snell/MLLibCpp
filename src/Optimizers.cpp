#include "Optimizers.hpp"
#include "HelperFunctions.hpp"
#include "MapVals.hpp"

GradientDescent::GradientDescent(double LR){
	learningRate = NumObject(LR);
}

void GradientDescent::minimize(vector<Variable*>& variables){
	for(int i = 0; i < variables.size(); i++){
		vector<NumObject> items = {variables[i]->value, variables[i]->derivative, learningRate};
		variables[i]->value = mapVals(this, &GradientDescent::operation, items);
	}
}

double GradientDescent::operation(vector<double>& a){
	return a[0] - a[1] * a[2];
}

MomentumGradientDescent::MomentumGradientDescent(double LR, double momentum){
	learningRate = NumObject(LR);
	momentumRate = NumObject(momentum);
}

void  MomentumGradientDescent::minimize(vector<Variable*>& variables){
	if(velocity.size() != variables.size()){
		velocity.clear();
		for(int i = 0; i < variables.size(); i++){
			vector<NumObject> items = {variables[i]->derivative, learningRate};
			velocity.push_back(mapVals(this, &MomentumGradientDescent::operation2, items));
			vector<NumObject> items2 = {variables[i]->value, velocity[i]};
			variables[i]->value = mapVals(this, &MomentumGradientDescent::operation1, items2);
		}
	}
	else{
		for(int i = 0; i < variables.size(); i++){
			for(int x = 0; x < variables[i]->derivative.values.size(); x++){
				velocity[i].values[x] = momentumRate.values[0] * velocity[i].values[x] + learningRate.values[0] * variables[i]->derivative.values[x];
			}
			vector<NumObject> items2 = {variables[i]->value, velocity[i]};
			variables[i]->value = mapVals(this, &MomentumGradientDescent::operation1, items2);
		}
	}
}

double MomentumGradientDescent::operation1(vector<double>& a){
	return a[0] - a[1];
}

double MomentumGradientDescent::operation2(vector<double>& a){
	return a[0] * a[1];
}


