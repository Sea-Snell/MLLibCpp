#include "CostFunctions.hpp"
#include "MapVals.hpp"

MeanSquared::MeanSquared(Node* hypothesis, Node* y, int dimentionVal){
	outCount = 0;
	inputs.push_back(hypothesis);
	inputs.push_back(y);
	hypothesis->outCount += 1;
	y->outCount += 1;
	name = "MeanSquared";
	dimention = dimentionVal;
	dCallCount = 0;
	gCallCount = 0;
}

NumObject MeanSquared::getValue(int t, int tf){
	gCallCount += 1;
	if(gCallCount > 1){
		if(gCallCount >= outCount){
			gCallCount = 0;
		}
		return derivativeMemo[t];
	}
	if(gCallCount >= outCount){
		gCallCount = 0;
	}

	NumObject hypothesis = inputs[0]->getValue(t, tf);
	NumObject y = inputs[1]->getValue(t, tf);

	vector<NumObject> items1 = {hypothesis, y};
	if(t == 0){
		differenceMemo.clear();
		differenceMemo.resize(tf + 1);
	}
	differenceMemo[t] = mapVals(this, &MeanSquared::differenceOperation, items1);

	vector<NumObject> items2 = {differenceMemo[t]};
	NumObject temp = mapVals(this, &MeanSquared::powerOperation, items2);
	NumObject total = reduceSumByDimention(temp, hypothesis.rank);
	total.values[0] /= differenceMemo[t].dimentions[dimention];
	return memoize(total, t, tf);
}

double MeanSquared::differenceOperation(vector<double>& a){
	return a[0] - a[1];
}

double MeanSquared::powerOperation(vector<double>& a){
	return a[0] * a[0];
}

void MeanSquared::derive(NumObject& seed, int t, int tf){
	if(sumSeed(seed)){
		if (typeid(*inputs[0]) != typeid(Constant)){
			vector<NumObject> items1 = {tempSeed, NumObject(2.0 / differenceMemo[t].dimentions[dimention]), differenceMemo[t]};
			NumObject eval1 = mapVals(this, &MeanSquared::deriveOperation1, items1);
			inputs[0]->derive(eval1, t, tf);
		}
	}
}

double MeanSquared::deriveOperation1(vector<double>& a){
	return a[0] * a[1] * a[2];
}

string MeanSquared::describe(){
	return name + "(" + inputs[0]->describe() + ", " + inputs[1]->describe() + ", " + to_string(dimention) + ")";
}

CrossEntropy::CrossEntropy(Node* hypothesis, Node* y, int dimentionVal){
	outCount = 0;
	inputs.push_back(hypothesis);
	inputs.push_back(y);
	hypothesis->outCount += 1;
	y->outCount += 1;
	name = "CrossEntropy";
	dimention = dimentionVal;
	dCallCount = 0;
	gCallCount = 0;
}

NumObject CrossEntropy::getValue(int t, int tf){
	gCallCount += 1;
	if(gCallCount > 1){
		if(gCallCount >= outCount){
			gCallCount = 0;
		}
		return derivativeMemo[t];
	}
	if(gCallCount >= outCount){
		gCallCount = 0;
	}

	NumObject hypothesis = inputs[0]->getValue(t, tf);
	NumObject y = inputs[1]->getValue(t, tf);

	vector<NumObject> items1 = {hypothesis, y};
	NumObject temp = mapVals(this, &CrossEntropy::operation, items1);
	NumObject ans = reduceSumByDimention(temp, temp.rank);
	ans.values[0] /= temp.dimentions[dimention];
	return memoize(ans, t, tf);
}

double CrossEntropy::operation(vector<double>& a){
	return -(a[1] * log(a[0]) + (1.0 - a[1]) * log(1.0 - a[0]));
}

void CrossEntropy::derive(NumObject& seed, int t, int tf){
	if(sumSeed(seed)){
		if (typeid(*inputs[0]) != typeid(Constant)){
			vector<NumObject> items1 = {tempSeed, NumObject(-1.0 / inputs[0]->derivativeMemo[t].dimentions[dimention]), inputs[1]->derivativeMemo[t], inputs[0]->derivativeMemo[t]};
			NumObject eval1 = mapVals(this, &CrossEntropy::deriveOperation1, items1);
			inputs[0]->derive(eval1, t, tf);
		}
	}
}

double CrossEntropy::deriveOperation1(vector<double>& a){
	return (a[0] * a[1] * (a[2] - a[3])) / (a[3] * (1.0 - a[3]));
}

string CrossEntropy::describe(){
	return name + "(" + inputs[0]->describe() + ", " + inputs[1]->describe() + ", " + to_string(dimention) + ")";
}

CrossEntropySoftmax::CrossEntropySoftmax(Node* hypothesis, Node* y, int dimentionVal, int meanDimentionVal){
	outCount = 0;
	inputs.push_back(hypothesis);
	inputs.push_back(y);
	hypothesis->outCount += 1;
	y->outCount += 1;
	name = "CrossEntropySoftmax";
	dimention = dimentionVal;
	meanDimention = meanDimentionVal;
	dCallCount = 0;
	gCallCount = 0;
}

NumObject CrossEntropySoftmax::getValue(int t, int tf){
	gCallCount += 1;
	if(gCallCount > 1){
		if(gCallCount >= outCount){
			gCallCount = 0;
		}
		return derivativeMemo[t];
	}
	if(gCallCount >= outCount){
		gCallCount = 0;
	}

	NumObject hypothesis = inputs[0]->getValue(t, tf);
	NumObject y = inputs[1]->getValue(t, tf);

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
	if(t == 0){
		softmaxMemo.clear();
		softmaxMemo.resize(tf + 1);
	}
	softmaxMemo[t] = NumObject(hypothesis.dimentions);
	double temp3;
	for(int i = 0; i < preSum; i++){
		for(int x = 0; x < hypothesis.dimentions[dimention]; x++){
			for(int z = 0; z < postSum; z++){
				temp3 = hypothesis.values[i * postSum * hypothesis.dimentions[dimention] + x * postSum + z] - temp2.values[i * postSum + z];
				softmaxMemo[t].values.push_back(exp(temp3));
				ans.values.push_back(y.values[i * postSum * hypothesis.dimentions[dimention] + x * postSum + z] * temp3);
			}
		}
	}

	NumObject ans2 = reduceSumByDimention(ans, hypothesis.rank);
	ans2.values[0] /= -hypothesis.dimentions[meanDimention];

	return memoize(ans2, t, tf);
}

double CrossEntropySoftmax::operation1(vector<double>& a){
	return a[0] + log(a[1]);
}

void CrossEntropySoftmax::derive(NumObject& seed, int t, int tf){
	if(sumSeed(seed)){
		if (typeid(*inputs[0]) != typeid(Constant)){
			vector<NumObject> items1 = {tempSeed, NumObject(1.0 / inputs[0]->derivativeMemo[t].dimentions[meanDimention]), softmaxMemo[t], inputs[1]->derivativeMemo[t]};
			NumObject eval1 = mapVals(this, &CrossEntropySoftmax::deriveOperation1, items1);
			inputs[0]->derive(eval1, t, tf);
		}
	}
}

double CrossEntropySoftmax::deriveOperation1(vector<double>& a){
	return a[0] * a[1] * (a[2] - a[3]);
}

string CrossEntropySoftmax::describe(){
	return name + "(" + inputs[0]->describe() + ", " + inputs[1]->describe() + ", " + to_string(dimention) + ", " + to_string(meanDimention) + ")";
}

