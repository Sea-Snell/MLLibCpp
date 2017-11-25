#include "Node.hpp"
#include "Math.hpp"
#include "MapVals.hpp"

NumObject::NumObject(){}

NumObject::NumObject(double val){
	rank = 0;
	values.push_back(val);
}

NumObject::NumObject(vector<double> val, vector<int> dimentionsList){
	rank = dimentionsList.size();
	dimentions = dimentionsList;
	values = val;
}

NumObject::NumObject(vector<int> dimentionsList, double fill){
	rank = dimentionsList.size();
	dimentions = dimentionsList;

	int tempSize = 1.0;
	for (int i = 0; i < rank; i++){
		tempSize *= dimentions[i];
	}

	values.resize(tempSize, fill);
}

NumObject::NumObject(vector<int> dimentionsList){
	rank = dimentionsList.size();
	dimentions = dimentionsList;

	int tempSize = 1.0;
	for (int i = 0; i < rank; i++){
		tempSize *= dimentions[i];
	}

	values.reserve(tempSize);
}

string NumObject::describe(){
	if (rank == 0){
		return to_string(values[0]);
	}
	string returnVal = "";
	int tempMod = 0;
	int totalElements = values.size();

	for(int i = 0; i < totalElements; i++){
		tempMod = totalElements;
		for (int x = 0; x < rank; x++){
			if (i % tempMod == 0){
				returnVal += "[";
			}
			tempMod /= dimentions[x];
		}

		returnVal += to_string(values[i]);

		tempMod = totalElements;
		for (int x = 0; x < rank; x++){
			if ((i + 1) % tempMod == 0){
				returnVal += "]";
			}
			tempMod /= dimentions[x];
		}
		
		if (i < totalElements - 1){
			returnVal += ", ";
		}
	}
	return returnVal;
}

NumObject NumObject::getIndex(vector<int> idx){
	int startIdx = 0;
	int endIdx = 0;
	int temp = 0;
	int currentProduct = 1;
	vector<int> newDimentions;

	for(int i = dimentions.size() - 1; i >= 0; i--){
		if(i < idx.size()){
			if (i == idx.size() - 1){
				temp = currentProduct;
			}
			startIdx += idx[i] * currentProduct;
		}
		else{
			newDimentions.push_back(dimentions[dimentions.size() + idx.size() - 1 - i]);
		}
		currentProduct *= dimentions[i];
	}

	endIdx = startIdx + temp - 1;


	vector<double> newVals (begin(values) + startIdx, begin(values) + endIdx + 1);

	return NumObject(newVals, newDimentions);
}

void NumObject::setIndex(vector<int> idx, vector<double> val){
	int startIdx = 0;
	int currentProduct = 1;

	for(int i = dimentions.size() - 1; i >= 0; i--){
		if(i < idx.size()){
			startIdx += idx[i] * currentProduct;
		}
		currentProduct *= dimentions[i];
	}

	for(int i = 0; i < val.size(); i++){
		values[startIdx + i] = val[i];
	}
}

void NumObject::setIndex(vector<int> idx, double val){
	int startIdx = 0;
	int currentProduct = 1;

	for(int i = dimentions.size() - 1; i >= 0; i--){
		startIdx += idx[i] * currentProduct;
		currentProduct *= dimentions[i];
	}
	values[startIdx] = val;
}

string Node::describe(){
	string ans = name + "(";
	for(int i = 0; i < inputs.size(); i++){
		ans += inputs[i]->describe();
		if(i != inputs.size() - 1){
			ans += ", ";
		}
	}
	ans += ")";
	return ans;
}

NumObject& Node::memoize(NumObject& val, int t, int tf){
	if(t == 0){
		derivativeMemo.clear();
		derivativeMemo.resize(tf + 1);
	}
	derivativeMemo[t] = val;
	return val;
}

bool Node::sumSeed(NumObject& seed){
	dCallCount += 1;
	if(dCallCount == 1){
		tempSeed = seed;
	}
	else{
		vector<NumObject> tempItems = {seed, tempSeed};
		tempSeed = mapVals(this, &Node::addSeedOperation, tempItems);
	}
	if(dCallCount >= outCount){
		dCallCount = 0;
		return true;
	}
	return false;
}

double Node::addSeedOperation(vector<double>& a){
	return a[0] + a[1];
}

Constant::Constant(NumObject val, string placeHolder){
	value = val;
	name = placeHolder;
	outCount = 0;
	dCallCount = 0;
	gCallCount = 0;
}

NumObject Constant::getValue(int t, int tf){
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

	return memoize(value, t, tf);
}

string Constant::describe(){
	if (name == ""){
		return value.describe();
	}
	return name;
}

void Constant::derive(NumObject& seed, int t, int tf){
	return;
}

Variable::Variable(NumObject val, string placeHolder): Constant(val, placeHolder){
	derivative = NumObject(value.dimentions, 0.0);
}

void Variable::derive(NumObject& seed, int t, int tf){
	dCallCount += 1;
	if(dCallCount == 1 && t == tf){
		for(int i = 0; i < derivative.values.size(); i++){
			derivative.values[i] = seed.values[i % seed.values.size()];
		}
	}
	else{
		for(int i = 0; i < derivative.values.size(); i++){
			derivative.values[i] += seed.values[i % seed.values.size()];
		}
	}

	if(dCallCount >= outCount){
		dCallCount = 0;
	}
}

NumObject reduceSumByDimention(NumObject& nums, int byDimention){
	if(byDimention <= 0){
		return nums;
	}

	int resultSize = nums.values.size();
	vector<int> resultDimentions;
	for(int i = 0; i < nums.rank; i++){
		if (i < byDimention){
			resultSize /= nums.dimentions[i];
		}
		else{
			resultDimentions.push_back(nums.dimentions[i]);
		}
	}

	NumObject answer = NumObject(resultDimentions);
	for(int i = 0; i < nums.values.size(); i++){
		if (i < resultSize){
			answer.values.push_back(nums.values[i]);
		}
		else{
			answer.values[i % resultSize] += nums.values[i];
		}
	}

	return answer;
}

BasicOperator::BasicOperator(Node* a, Node* b){
	outCount = 0;
	inputs.push_back(a);
	inputs.push_back(b);
	a->outCount += 1;
	b->outCount += 1;
	dCallCount = 0;
	gCallCount = 0;
}

NumObject BasicOperator::getValue(int t, int tf){
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
	vector<NumObject> items = {inputs[0]->getValue(t, tf), inputs[1]->getValue(t, tf)};
	NumObject ans = mapVals(this, &BasicOperator::operation, items);
	return memoize(ans, t, tf);
}

string BasicOperator::describe(){
	return "(" + inputs[0]->describe() + " " + name + " " + inputs[1]->describe() + ")";
}

BasicFunction::BasicFunction(Node* a){
	outCount = 0;
	inputs.push_back(a);
	a->outCount += 1;
	dCallCount = 0;
	gCallCount = 0;
}

NumObject BasicFunction::getValue(int t, int tf){
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
	vector<NumObject> items = {inputs[0]->getValue(t, tf)};
	NumObject ans = mapVals(this, &BasicFunction::operation, items);
	return memoize(ans, t, tf);
}


