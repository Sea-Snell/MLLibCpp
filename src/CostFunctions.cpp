#include "CostFunctions.hpp"
#include "MapVals.hpp"

MeanSquared::MeanSquared(Node* hypothesis, Node* y, int dimentionVal){
	inputs.push_back(hypothesis);
	inputs.push_back(y);
	name = "MeanSquared";
	dimention = dimentionVal;
}

NumObject MeanSquared::getValue(){
	NumObject hypothesis = inputs[0]->getValue();
	NumObject y = inputs[1]->getValue();

	vector<NumObject> items1 = {hypothesis, y};
	differenceMemo = mapVals(this, &MeanSquared::differenceOperation, items1);

	vector<NumObject> items2 = {differenceMemo};
	NumObject temp = mapVals(this, &MeanSquared::powerOperation, items2);
	NumObject total = reduceSumByDimention(temp, hypothesis.rank);
	total.values[0] /= differenceMemo.dimentions[dimention];
	return memoize(total);
}

double MeanSquared::differenceOperation(vector<double>& a){
	return a[0] - a[1];
}

double MeanSquared::powerOperation(vector<double>& a){
	return a[0] * a[0];
}

void MeanSquared::derive(NumObject& seed){
	if (typeid(*inputs[0]) != typeid(Constant)){
		vector<NumObject> items1 = {seed, NumObject(2.0 / differenceMemo.dimentions[dimention]), differenceMemo};
		NumObject eval1 = mapVals(this, &MeanSquared::deriveOperation1, items1);
		inputs[0]->derive(eval1);
	}
}

double MeanSquared::deriveOperation1(vector<double>& a){
	return a[0] * a[1] * a[2];
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

NumObject CrossEntropy::getValue(){
	NumObject hypothesis = inputs[0]->getValue();
	NumObject y = inputs[1]->getValue();

	vector<NumObject> items1 = {hypothesis, y};
	NumObject temp = mapVals(this, &CrossEntropy::operation, items1);
	NumObject ans = reduceSumByDimention(temp, temp.rank);
	ans.values[0] /= temp.dimentions[dimention];
	return memoize(ans);
}

double CrossEntropy::operation(vector<double>& a){
	return -(a[1] * log(a[0]) + (1.0 - a[1]) * log(1.0 - a[0]));
}

void CrossEntropy::derive(NumObject& seed){
	if (typeid(*inputs[0]) != typeid(Constant)){
		vector<NumObject> items1 = {seed, NumObject(-1.0 / inputs[0]->derivativeMemo.dimentions[dimention]), inputs[1]->derivativeMemo, inputs[0]->derivativeMemo};
		NumObject eval1 = mapVals(this, &CrossEntropy::deriveOperation1, items1);
		inputs[0]->derive(eval1);
	}
}

double CrossEntropy::deriveOperation1(vector<double>& a){
	return (a[0] * a[1] * (a[2] - a[3])) / (a[3] * (1.0 - a[3]));
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

NumObject CrossEntropySoftmax::getValue(){
	NumObject hypothesis = inputs[0]->getValue();
	NumObject y = inputs[1]->getValue();

	if(dimention == -1){
		dimention = hypothesis.rank - 1;
	}

	vector<int> newDimentions;
	for(int i = 0; i < hypothesis.rank; i++){
		if (i != dimention){
			newDimentions.push_back(hypothesis.dimentions[i]);
		}
	}
	double lowVal = -numeric_limits<double>::infinity();
	NumObject maxVals = NumObject(newDimentions, lowVal);

	int preSum = 1;
	for(int i = 0; i < dimention; i++){
		preSum *= hypothesis.dimentions[i];
	}
	int postSum = hypothesis.values.size() / (preSum * hypothesis.dimentions[dimention]);

	for(int i = 0; i < preSum; i++){
		for(int x = 0; x < hypothesis.dimentions[dimention]; x++){
			for(int z = 0; z < postSum; z++){
					maxVals.values[i * postSum + z] = max(maxVals.values[i * postSum + z], hypothesis.values[i * postSum * hypothesis.dimentions[dimention] + postSum * x + z]);
			}
		}
	}

	NumObject temp = NumObject(maxVals.dimentions, 0.0);
	for(int i = 0; i < preSum; i++){
		for(int x = 0; x < hypothesis.dimentions[dimention]; x++){
			for(int z = 0; z < postSum; z++){
				temp.values[i * postSum + z] += exp(hypothesis.values[i * postSum * hypothesis.dimentions[dimention] + x * postSum + z] - maxVals.values[i * postSum + z]);
			}
		}
	}

	vector<NumObject> items1 = {maxVals, temp};
	NumObject temp2 = mapVals(this, &CrossEntropySoftmax::operation1, items1);

	NumObject ans = NumObject(hypothesis.dimentions);
	softmaxMemo = NumObject(hypothesis.dimentions);
	double temp3;
	for(int i = 0; i < preSum; i++){
		for(int x = 0; x < hypothesis.dimentions[dimention]; x++){
			for(int z = 0; z < postSum; z++){
				temp3 = hypothesis.values[i * postSum * hypothesis.dimentions[dimention] + x * postSum + z] - temp2.values[i * postSum + z];
				softmaxMemo.values.push_back(exp(temp3));
				ans.values.push_back(y.values[i * postSum * hypothesis.dimentions[dimention] + x * postSum + z] * temp3);
			}
		}
	}

	NumObject ans2 = reduceSumByDimention(ans, hypothesis.rank);
	ans2.values[0] /= -hypothesis.dimentions[meanDimention];

	return memoize(ans2);
}

double CrossEntropySoftmax::operation1(vector<double>& a){
	return a[0] + log(a[1]);
}

void CrossEntropySoftmax::derive(NumObject& seed){
	if (typeid(*inputs[0]) != typeid(Constant)){
		vector<NumObject> items1 = {seed, NumObject(1.0 / inputs[0]->derivativeMemo.dimentions[meanDimention]), softmaxMemo, inputs[1]->derivativeMemo};
		NumObject eval1 = mapVals(this, &CrossEntropySoftmax::deriveOperation1, items1);
		inputs[0]->derive(eval1);
	}
}

double CrossEntropySoftmax::deriveOperation1(vector<double>& a){
	return a[0] * a[1] * (a[2] - a[3]);
}

string CrossEntropySoftmax::describe(){
	return name + "(" + inputs[0]->describe() + ", " + inputs[1]->describe() + ", " + to_string(dimention) + ", " + to_string(meanDimention) + ")";
}

