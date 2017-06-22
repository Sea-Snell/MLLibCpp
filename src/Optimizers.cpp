#include "Optimizers.hpp"
#include "HelperFunctions.hpp"

GradientDescent::GradientDescent(double LR){
	learningRate = LR;
}

void GradientDescent::minimize(Node* expression, vector<Variable*>& variables, vector<Variable*>& noClearVariables){
	derive(expression, variables);

	for(int i = 0; i < variables.size(); i++){
		for(int x = 0; x < variables[i]->outSize; x++){
			variables[i]->derivativeMemo[x] -= learningRate * variables[i]->derivative[x];
		}
	}

	for(int i = 0; i < noClearVariables.size(); i++){
		for(int x = 0; x < noClearVariables[i]->outSize; x++){
			noClearVariables[i]->derivativeMemo[x] -= learningRate * noClearVariables[i]->derivative[x];
		}
	}
}

MomentumGradientDescent::MomentumGradientDescent(double LR, double momentum){
	learningRate = LR;
	momentumRate = momentum;
}

void  MomentumGradientDescent::minimize(Node* expression, vector<Variable*>& variables, vector<Variable*>& noClearVariables){
	derive(expression, variables);

	if(velocity.size() != variables.size() + noClearVariables.size()){
		velocity.clear();
		for(int i = 0; i < variables.size(); i++){
			vector<double> temp;
			for(int x = 0; x < variables[i]->outSize; x++){
				temp.push_back(learningRate * variables[i]->derivative[x]);
			}
			velocity.push_back(temp);

			for(int x = 0; x < variables[i]->outSize; x++){
				variables[i]->derivativeMemo[x] -= velocity[i][x];
			}
		}

		for(int i = 0; i < noClearVariables.size(); i++){
			vector<double> temp;
			for(int x = 0; x < noClearVariables[i]->outSize; x++){
				temp.push_back(learningRate * noClearVariables[i]->derivative[x]);
			}
			velocity.push_back(temp);

			for(int x = 0; x < noClearVariables[i]->outSize; x++){
				noClearVariables[i]->derivativeMemo[x] -= velocity[i + variables.size()][x];
			}
		}
	}
	else{
		for(int i = 0; i < variables.size(); i++){
			for(int x = 0; x < variables[i]->outSize; x++){
				velocity[i][x] = momentumRate * velocity[i][x] + learningRate * variables[i]->derivative[x];
			}
			for(int x = 0; x < variables[i]->outSize; x++){
				variables[i]->derivativeMemo[x] -= velocity[i][x];
			}
		}

		for(int i = 0; i < noClearVariables.size(); i++){
			for(int x = 0; x < noClearVariables[i]->outSize; x++){
				velocity[i + variables.size()][x] = momentumRate * velocity[i + variables.size()][x] + learningRate * noClearVariables[i]->derivative[x];
			}
			for(int x = 0; x < noClearVariables[i]->outSize; x++){
				noClearVariables[i]->derivativeMemo[x] -= velocity[i + variables.size()][x];
			}
		}
	}
}


