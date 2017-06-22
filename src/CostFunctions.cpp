#include "CostFunctions.hpp"

MeanSquared::MeanSquared(Node* hypothesis, Node* y, int dimentionVal){
	inputs.push_back(hypothesis);
	inputs.push_back(y);
	name = "MeanSquared";
	dimention = dimentionVal;
}

void MeanSquared::getValue(){
	inputs[0]->getValue();
	inputs[1]->getValue();

	double total = 0.0;
	for(int i = 0; i < inputs[0]->outSize; i++){
		differenceMemo[i] = inputs[0]->derivativeMemo[i] - inputs[1]->derivativeMemo[i];
		total += differenceMemo[i] * differenceMemo[i];
	}

	derivativeMemo[0] = total / inputs[0]->outDimentions[dimention];
}

void MeanSquared::getValueDimentions(){
	inputs[0]->getValueDimentions();
	inputs[1]->getValueDimentions();

	outDimentions = {};
	outRank = 0;
	outSize = 1;
	derivativeMemo.clear();
	derivativeMemo.resize(1, 0.0);

	differenceMemo.clear();
	differenceMemo.resize(inputs[0]->outSize, 0.0);
}

void MeanSquared::derive(vector<double>& seed){
	if (typeid(*inputs[0]) != typeid(Constant)){
		for(int i = 0; i < ansSize; i++){
			ans[i] = seed[0] * average * differenceMemo[i];
		}
		inputs[0]->derive(ans);
	}
}

void MeanSquared::deriveDimentions(vector<int>& seedDimentionsVal){
	seedDimentions = seedDimentionsVal;
	seedRank = seedDimentions.size();
	seedSize = 1;
	for(int i = 0; i < seedRank; i++){
		seedSize *= seedDimentions[i];
	}

	ansDimentions = inputs[0]->outDimentions;
	ansRank = inputs[0]->outRank;
	ansSize = inputs[0]->outSize;
	ans.clear();
	ans.resize(ansSize, 0.0);

	average = 2.0 / inputs[0]->outDimentions[dimention];

	inputs[0]->deriveDimentions(ansDimentions);
}

string MeanSquared::describe(){
	return name + "(" + inputs[0]->describe() + ", " + inputs[1]->describe() + ", " + to_string(dimention) + ")";
}

CrossEntropy::CrossEntropy(Node* hypothesis, Node* y, int dimentionVal){
	inputs.push_back(hypothesis);
	inputs.push_back(y);
	name = "CrossEntropy";
	dimention = dimentionVal;
}

void CrossEntropy::getValue(){
	inputs[0]->getValue();
	inputs[1]->getValue();

	double total = 0.0;
	for(int i = 0; i < inputs[0]->outSize; i++){
		total -= inputs[1]->derivativeMemo[i] * log(inputs[0]->derivativeMemo[i]) + (1.0 - inputs[1]->derivativeMemo[i]) * log(1.0 - inputs[0]->derivativeMemo[i]);
	}
	derivativeMemo[0] = total / inputs[0]->outDimentions[dimention];
}

void CrossEntropy::getValueDimentions(){
	inputs[0]->getValueDimentions();
	inputs[1]->getValueDimentions();

	outDimentions = {};
	outRank = 0;
	outSize = 1;
	derivativeMemo.clear();
	derivativeMemo.resize(1, 0.0);
}

void CrossEntropy::derive(vector<double>& seed){
	for(int i = 0; i < ansSize; i++){
		ans[i] = (seed[0] * average * (inputs[1]->derivativeMemo[i] - inputs[0]->derivativeMemo[i])) / (inputs[0]->derivativeMemo[i] * (1.0 - inputs[0]->derivativeMemo[i]));
	}
	inputs[0]->derive(ans);
}

void CrossEntropy::deriveDimentions(vector<int>& seedDimentionsVal){
	seedDimentions = seedDimentionsVal;
	seedRank = seedDimentions.size();
	seedSize = 1;
	for(int i = 0; i < seedRank; i++){
		seedSize *= seedDimentions[i];
	}

	ansDimentions = inputs[0]->outDimentions;
	ansRank = inputs[0]->outRank;
	ansSize = inputs[0]->outSize;
	ans.clear();
	ans.resize(ansSize, 0.0);

	average = -1.0 / inputs[0]->outDimentions[dimention];

	inputs[0]->deriveDimentions(ansDimentions);
}

string CrossEntropy::describe(){
	return name + "(" + inputs[0]->describe() + ", " + inputs[1]->describe() + ", " + to_string(dimention) + ")";
}

CrossEntropySoftmax::CrossEntropySoftmax(Node* hypothesis, Node* y, int dimentionVal, int meanDimentionVal){
	inputs.push_back(hypothesis);
	inputs.push_back(y);
	name = "CrossEntropySoftmax";
	dimention = dimentionVal;
	meanDimention = meanDimentionVal;
}

void CrossEntropySoftmax::getValue(){
	inputs[0]->getValue();
	inputs[1]->getValue();

	double lowVal = -numeric_limits<double>::infinity();
	for(int i = 0; i < newSize; i++){
		maxVals[i] = lowVal;
	}

	for(int i = 0; i < preSum; i++){
		for(int x = 0; x < inputs[0]->outDimentions[dimention]; x++){
			for(int z = 0; z < postSum; z++){
				maxVals[i * postSum + z] = max(maxVals[i * postSum + z], inputs[0]->derivativeMemo[i * postSum * inputs[0]->outDimentions[dimention] + postSum * x + z]);
			}
		}
	}

	for(int i = 0; i < preSum; i++){
		for(int z = 0; z < postSum; z++){
			double sum = 0.0;
			for(int x = 0; x < inputs[0]->outDimentions[dimention]; x++){
				sum += exp(inputs[0]->derivativeMemo[i * postSum * inputs[0]->outDimentions[dimention] + x * postSum + z] - maxVals[i * postSum + z]);
			}
			temp[i * postSum + z] = sum;
		}
	}

	for(int i = 0; i < newSize; i++){
		temp2[i] = maxVals[i] + log(temp[i]);
	}

	for(int i = 0; i < preSum; i++){
		for(int x = 0; x < inputs[0]->outDimentions[dimention]; x++){
			for(int z = 0; z < postSum; z++){
				double temp3 = inputs[0]->derivativeMemo[i * postSum * inputs[0]->outDimentions[dimention] + x * postSum + z] - temp2[i * postSum + z];
				softmaxMemo[i * postSum * inputs[0]->outDimentions[dimention] + x * postSum + z] = exp(temp3);
				ans[i * postSum * inputs[0]->outDimentions[dimention] + x * postSum + z] = inputs[1]->derivativeMemo[i * postSum * inputs[0]->outDimentions[dimention] + x * postSum + z] * temp3;
			}
		}
	}

	double total = 0.0;
	for(int i = 0; i < inputs[0]->outSize; i++){
		total += ans[i];
	}

	derivativeMemo[0] = total / -inputs[0]->outDimentions[meanDimention];
}

void CrossEntropySoftmax::getValueDimentions(){
	inputs[0]->getValueDimentions();
	inputs[1]->getValueDimentions();

	if(dimention == -1){
		dimention = inputs[0]->outRank - 1;
	}

	newSize = 1;
	newDimentions.clear();
	for(int i = 0; i < inputs[0]->outRank; i++){
		if (i != dimention){
			newSize *= inputs[0]->outDimentions[i];
			newDimentions.push_back(inputs[0]->outDimentions[i]);
		}
	}
	
	double lowVal = -numeric_limits<double>::infinity();
	maxVals.clear();
	maxVals.resize(newSize, lowVal);


	preSum = 1;
	for(int i = 0; i < dimention; i++){
		preSum *= inputs[0]->outDimentions[i];
	}
	postSum = inputs[0]->outSize / (preSum * inputs[0]->outDimentions[dimention]);

	temp.clear();
	temp.resize(newSize, 0.0);

	temp2.clear();
	temp2.resize(newSize, 0.0);

	ans.clear();
	ans.resize(inputs[0]->outSize);

	softmaxMemo.clear();
	softmaxMemo.resize(inputs[0]->outSize);

	outRank = 0;
	outDimentions = {};
	outSize = 1;
	derivativeMemo.clear();
	derivativeMemo.resize(1, 0.0);
}

void CrossEntropySoftmax::derive(vector<double>& seed){
	if (typeid(*inputs[0]) != typeid(Constant)){
		for(int i = 0; i < ansSize2; i++){
			ans2[i] = seed[0] * average * (softmaxMemo[i] - inputs[1]->derivativeMemo[i]);
		}
		inputs[0]->derive(ans2);
	}
}

void CrossEntropySoftmax::deriveDimentions(vector<int>& seedDimentionsVal){
	seedDimentions = seedDimentionsVal;
	seedRank = seedDimentions.size();
	seedSize = 1;
	for(int i = 0; i < seedRank; i++){
		seedSize *= seedDimentions[i];
	}

	ansDimentions2 = inputs[0]->outDimentions;
	ansRank2 = inputs[0]->outRank;
	ansSize2 = inputs[0]->outSize;
	ans2.clear();
	ans2.resize(ansSize2, 0.0);

	average = 1.0 / inputs[0]->outDimentions[meanDimention];

	inputs[0]->deriveDimentions(ansDimentions2);
}

string CrossEntropySoftmax::describe(){
	return name + "(" + inputs[0]->describe() + ", " + inputs[1]->describe() + ", " + to_string(dimention) + ", " + to_string(meanDimention) + ")";
}

void KL::getValue(){
	inputs[0]->getValue();
	inputs[1]->getValue();

	double total = 0.0;
	for(int i = 0; i < inputs[0]->outSize; i++){
		total += inputs[1]->derivativeMemo[i] * log(inputs[1]->derivativeMemo[i] / inputs[0]->derivativeMemo[i]);
	}
	derivativeMemo[0] = total / inputs[0]->outDimentions[dimention];
}

void KL::derive(vector<double>& seed){
	for(int i = 0; i < ansSize; i++){
		ans[i] = seed[0] * inputs[0]->derivativeMemo[i];
	}
	inputs[0]->derive(ans);
}

