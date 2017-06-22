#include "Node.hpp"
#include "Math.hpp"

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

Constant::Constant(vector<double> val, vector<int> dimentionsVal, string placeHolder){
	derivativeMemo = val;
	name = placeHolder;
	outDimentions = dimentionsVal;
	outRank = dimentionsVal.size();

	outSize = 1;
	for(int i = 0; i < outRank; i++){
		outSize *= outDimentions[i];
	}
}

Constant::Constant(double val, string placeHolder){
	derivativeMemo.push_back(val);
	name = placeHolder;
	outRank = 0;
	outSize = 1;
}

Constant::Constant(const Constant &a){
	derivativeMemo = a.derivativeMemo;
	name = a.name;
	outDimentions = a.outDimentions;
	outRank = a.outRank;
	outSize = a.outSize;
}

void Constant::getValue(){
	return;
}

string Constant::describe(){
	if (name == ""){
		if (outRank == 0){
			return to_string(derivativeMemo[0]);
		}
		string returnVal = "";
		int tempMod = 0;
		int totalElements = outSize;

		for(int i = 0; i < totalElements; i++){
			tempMod = totalElements;
			for (int x = 0; x < outRank; x++){
				if (i % tempMod == 0){
					returnVal += "[";
				}
				tempMod /= outDimentions[x];
			}

			returnVal += to_string(derivativeMemo[i]);

			tempMod = totalElements;
			for (int x = 0; x < outRank; x++){
				if ((i + 1) % tempMod == 0){
					returnVal += "]";
				}
				tempMod /= outDimentions[x];
			}
		
			if (i < totalElements - 1){
				returnVal += ", ";
			}
		}
		return returnVal;
	}
	return name;
}

void Constant::derive(vector<double>& seed){
	return;
}

void Constant::getValueDimentions(){
	return;
}

void Constant::deriveDimentions(vector<int>& seedDimentionsVal){
	return;
}

Variable::Variable(vector<double> val, vector<int> dimentions, string placeHolder): Constant(val, dimentions, placeHolder){
	derivative.resize(outSize, 0.0);
}

Variable::Variable(double val, string placeHolder): Constant(val, placeHolder){
	derivative.resize(outSize, 0.0);
}

Variable::Variable(const Constant &a): Constant(a){
	derivative.resize(outSize, 0.0);
}

void Variable::getValueDimentions(){
	location = 0;
}

void Variable::getValue(){
	location = 0;
}

void Variable::derive(vector<double>& seed){
	for(int i = 0; i < outSize; i++){
		derivative[i] += seed[i % seedSizes[location]];
	}
	location += 1;
}

void Variable::deriveDimentions(vector<int>& seedDimentionsVal){
	int temp = 1;
	for(int i = 0; i < seedDimentionsVal.size(); i++){
		temp *= seedDimentionsVal[i];
	}
	seedSizes.push_back(temp);
	location += 1;
}

BasicOperator::BasicOperator(Node* a, Node* b){
	inputs.push_back(a);
	inputs.push_back(b);
}

void BasicOperator::getValue(){
	inputs[0]->getValue();
	inputs[1]->getValue();

	for(int i = 0; i < outSize; i++){
		derivativeMemo[i] = operation(inputs[0]->derivativeMemo[i % inputs[0]->outSize], inputs[1]->derivativeMemo[i % inputs[1]->outSize]);
	}
}

string BasicOperator::describe(){
	return "(" + inputs[0]->describe() + " " + name + " " + inputs[1]->describe() + ")";
}

void BasicOperator::getValueDimentions(){
	inputs[0]->getValueDimentions();
	inputs[1]->getValueDimentions();

	if(inputs[0]->outRank > inputs[1]->outRank){
		outRank = inputs[0]->outRank;
		outDimentions = inputs[0]->outDimentions;
	}
	else{
		outRank = inputs[1]->outRank;
		outDimentions = inputs[1]->outDimentions;
	}

	outSize = 1;
	for(int i = 0; i < outRank; i++){
		outSize *= outDimentions[i];
	}

	derivativeMemo.clear();
	derivativeMemo.resize(outSize, 0.0);
}

void BasicOperator::helpDeriveDimentions(vector<vector<int>> dependentDimentions1, vector<vector<int>> dependentDimentions2){
	tempDimentions1 = helpMapDimentions(dependentDimentions1);
	tempRank1 = tempDimentions1.size();
	tempSize1 = getSize(tempDimentions1);
	temp1.clear();
	temp1.resize(tempSize1, 0.0);

	ansDimentions1 = helpReduceDimentions(tempDimentions1, inputs[0]->outDimentions);
	ansRank1 = ansDimentions1.size();
	ansSize1 = getSize(ansDimentions1);
	ans1.clear();
	ans1.resize(ansSize1, 0.0);

	tempDimentions2 = helpMapDimentions(dependentDimentions2);
	tempRank2 = tempDimentions2.size();
	tempSize2 = getSize(tempDimentions2);
	temp2.clear();
	temp2.resize(tempSize2, 0.0);

	ansDimentions2 = helpReduceDimentions(tempDimentions2, inputs[1]->outDimentions);
	ansRank2 = ansDimentions2.size();
	ansSize2 = getSize(ansDimentions2);
	ans2.clear();
	ans2.resize(ansSize2, 0.0);
}

BasicFunction::BasicFunction(Node* a){
	inputs.push_back(a);
}

void BasicFunction::getValue(){
	inputs[0]->getValue();

	for(int i = 0; i < outSize; i++){
		derivativeMemo[i] = operation(inputs[0]->derivativeMemo[i % inputs[0]->outSize]);
	}
}

void BasicFunction::getValueDimentions(){
	inputs[0]->getValueDimentions();

	outRank = inputs[0]->outRank;
	outDimentions = inputs[0]->outDimentions;
	outSize = inputs[0]->outSize;

	derivativeMemo.clear();
	derivativeMemo.resize(outSize, 0.0);
}

void BasicFunction::helpDeriveDimentions(vector<vector<int>> dependentDimentions1){
	tempDimentions1 = helpMapDimentions(dependentDimentions1);
	tempRank1 = tempDimentions1.size();
	tempSize1 = getSize(tempDimentions1);
	temp1.clear();
	temp1.resize(tempSize1, 0.0);

	ansDimentions1 = helpReduceDimentions(tempDimentions1, inputs[0]->outDimentions);
	ansRank1 = ansDimentions1.size();
	ansSize1 = getSize(ansDimentions1);
	ans1.clear();
	ans1.resize(ansSize1, 0.0);
}

vector<int> helpMapDimentions(vector<vector<int>> dimentions){
	int maxRank = 0;
	int maxIdx = 0;
	for(int i = 0; i < dimentions.size(); i++){
		if(dimentions[i].size() > maxRank){
			maxRank = dimentions[i].size();
			maxIdx = i;
		}
	}

	return dimentions[maxIdx];
}

vector<int> helpReduceDimentions(vector<int> a, vector<int> b){
	if((int)(a.size()) - (int)(b.size()) <= 0){
		return a;
	}
	return b;
}

int getSize(vector<int> dimentions){
	int size = 1;
	for(int i = 0; i < dimentions.size(); i++){
		size *= dimentions[i];
	}
	return size;
}
