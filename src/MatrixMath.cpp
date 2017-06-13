#include "MatrixMath.hpp"
#include "Node.hpp"
#include "MapVals.hpp"
#include <limits>

MatMul::MatMul(Node* a, Node* b, int dimentionAVal, int dimentionBVal){
	inputs.push_back(a);
	inputs.push_back(b);
	name = "MatMul";
	dimentionA = dimentionAVal;
	dimentionB = dimentionBVal;
}

NumObject MatMul::getValue(){
	NumObject a = inputs[0]->getValue();
	NumObject b = inputs[1]->getValue();

	vector<int> newDimentions;
	vector<int> realDimentions;

	int bNormalizeProdct = 1;

	int lowIdx;
	int highIdx;
	int adjustedDimentionA = dimentionA;

	if(a.rank < b.rank){
		adjustedDimentionA = dimentionA + (b.rank - a.rank);
		lowIdx = min(adjustedDimentionA, dimentionB);
		highIdx = max(adjustedDimentionA, dimentionB);	
	}
	else{
		lowIdx = min(dimentionA, dimentionB);
		highIdx = max(dimentionA, dimentionB);
	}

	vector<vector<int>> dimentions = getDimentions(a.dimentions, b.dimentions, adjustedDimentionA);
	newDimentions = dimentions[0];
	realDimentions = dimentions[1];


	int firstProduct = 1;
	for(int i = 0; i < lowIdx; i++){
		firstProduct *= newDimentions[i];
	}
	int secondProduct = 1;
	for(int i = lowIdx + 1; i < highIdx; i++){
		secondProduct *= newDimentions[i];
	}
	int thirdProduct = 1;
	for(int i = highIdx + 1; i < newDimentions.size(); i++){
		thirdProduct *= newDimentions[i];
	}

	if(b.rank < a.rank){
		for(int i = newDimentions.size() - 1; i >= newDimentions.size() - (a.rank - b.rank); i--){
			bNormalizeProdct *= newDimentions[i];
		}
	}

	int prodA = thirdProduct;
	int prodB;
	int prodC;
	int prodD;

	int prodE;
	int prodF;
	int prodG;

	if(lowIdx == adjustedDimentionA){
		prodB = prodA * newDimentions[highIdx];
		prodC = prodB * secondProduct;
		prodD = prodC * a.dimentions[dimentionA];

		prodE = prodA * b.dimentions[dimentionB];
		prodF = prodE * secondProduct;
		prodG = prodF * newDimentions[lowIdx];
	}
	else{
		prodB = prodA * a.dimentions[dimentionA];
		prodC = prodB * secondProduct;
		prodD = prodC * newDimentions[lowIdx];

		prodE = prodA * newDimentions[highIdx];
		prodF = prodE * secondProduct;
		prodG = prodF * b.dimentions[dimentionB];
	}

	NumObject ans = NumObject(realDimentions);
	for(int c = 0; c < firstProduct; c++){
		for(int d = 0; d < newDimentions[lowIdx]; d++){
			for(int e = 0; e < secondProduct; e++){
				for(int f = 0; f < newDimentions[highIdx]; f++){
					for(int g = 0; g < thirdProduct; g++){
						double sum = 0.0;
						if(lowIdx == adjustedDimentionA){
							for(int h = 0; h < a.dimentions[dimentionA]; h++){
								sum += a.values[(c * prodD + h * prodC + e * prodB + f * prodA + g) % a.values.size()] * b.values[(c * prodG + d * prodF + e * prodE + h * prodA + g) / bNormalizeProdct];
							}
						}
						else{
							for(int h = 0; h < a.dimentions[dimentionA]; h++){
								sum += a.values[(c * prodD + d * prodC + e * prodB + h * prodA + g) % a.values.size()] * b.values[(c * prodG + h * prodF + e * prodE + f * prodA + g) / bNormalizeProdct];
							}
						}
						ans.values.push_back(sum);
					}
				}
			}
		}
	}
	newDimentionsMemo = newDimentions;
	adjustedDimentionAMemo = adjustedDimentionA;
	return memoize(ans);
}


vector<vector<int>> MatMul::getDimentions(vector<int> a, vector<int> b, int adjustedDimentionA){
	vector<int> newDimentions;
	vector<int> realDimentions;

	if(adjustedDimentionA == dimentionB){
		if(a.size() >= b.size()){
			for(int i = 0; i < a.size(); i++){
				if(i != dimentionA){
					newDimentions.push_back(a[i]);
					realDimentions.push_back(a[i]);
				}
				else{
					newDimentions.push_back(1);
				}
			}
		}
		else{
			for(int i = 0; i < b.size(); i++){
				if(i != dimentionB){
					newDimentions.push_back(b[i]);
					realDimentions.push_back(b[i]);
				}
				else{
					newDimentions.push_back(1);
				}
			}
		}
	}
	else{
		if(a.size() >= b.size()){
			for(int i = 0; i < a.size(); i++){
				if(i == adjustedDimentionA){
					if(i < b.size()){
						newDimentions.push_back(b[i]);
						realDimentions.push_back(b[i]);
					}
					else{
						newDimentions.push_back(1);
					}
				}
				else{
					newDimentions.push_back(a[i]);
					realDimentions.push_back(a[i]);
				}
			}
		}
		else{
			for(int i = 0; i < b.size(); i++){
				if(i == dimentionB){
					if(i >= (b.size() - a.size())){
						newDimentions.push_back(a[i - (b.size() - a.size())]);
						realDimentions.push_back(a[i - (b.size() - a.size())]);
					}
					else{
						newDimentions.push_back(1);
					}
				}
				else{
					newDimentions.push_back(b[i]);
					realDimentions.push_back(b[i]);
				}
			}
		}
	}

	vector<vector<int>> returnVal = {newDimentions, realDimentions};
	return returnVal;
}


void MatMul::derive(NumObject& seed){
	NumObject a = inputs[0]->derivativeMemo;
	NumObject b = inputs[1]->derivativeMemo;

	if (typeid(*inputs[0]) != typeid(Constant)){
		vector<int> newDimentions;
		vector<int> tempBDimentions;

		int bNormalizeProdct = 1;
		int realDimentionsProduct = 1;

		int lowIdx;
		int highIdx;


		if(adjustedDimentionAMemo == dimentionB){
			int firstProduct = 1;
			int secondProduct = 1;

			if(a.rank >= b.rank){
				for(int i = 0; i < dimentionA; i++){
					firstProduct *= a.dimentions[i];
				}
				for(int i = dimentionA + 1; i < a.rank; i++){
					secondProduct *= a.dimentions[i];
				}
			}
			else{
				for(int i = 0; i < dimentionB; i++){
					firstProduct *= b.dimentions[i];
				}
				for(int i = dimentionB + 1; i < b.rank; i++){
					secondProduct *= b.dimentions[i];
				}
			}

			for(int i = 0; i < a.rank; i++){
				realDimentionsProduct *= a.dimentions[i];
			}

			if(b.rank < a.rank){
				for(int i = a.rank - 1; i >= b.rank; i--){
					bNormalizeProdct *= a.dimentions[i];
				}
			}

			NumObject ans = NumObject(a.dimentions);
			for(int c = 0; c < firstProduct; c++){
				for(int d = 0; d < a.dimentions[dimentionA]; d++){
					for(int e = 0; e < secondProduct; e++){
						if(ans.values.size() >= realDimentionsProduct || (c * secondProduct * a.dimentions[dimentionA] + d * secondProduct + e) % realDimentionsProduct != ans.values.size()){
							ans.values[(c * secondProduct * a.dimentions[dimentionA] + d * secondProduct + e) % realDimentionsProduct] += seed.values[(c * secondProduct + e) % seed.values.size()] * b.values[((c * secondProduct * a.dimentions[dimentionA] + d * secondProduct + e) / bNormalizeProdct) % b.values.size()];
						}
						else{
							ans.values.push_back(seed.values[(c * secondProduct + e) % seed.values.size()] * b.values[((c * secondProduct * a.dimentions[dimentionA] + d * secondProduct + e) / bNormalizeProdct) % b.values.size()]);
						}
					}
				}
			}
			inputs[0]->derive(ans);
		}
		else{
			if(b.rank >= a.rank){
				tempBDimentions = b.dimentions;
				int temp = tempBDimentions[dimentionB];
				tempBDimentions[dimentionB] = tempBDimentions[dimentionA + (b.rank - a.rank)];
				tempBDimentions[dimentionA + (b.rank - a.rank)] = temp;
			}
			else{
				tempBDimentions = a.dimentions;
				if(dimentionA < b.rank){
					tempBDimentions[dimentionB] = b.dimentions[dimentionA];
				}
				else{
					tempBDimentions[dimentionB] = 1;
				}
			}

			highIdx = max(adjustedDimentionAMemo, dimentionB);
			lowIdx = min(adjustedDimentionAMemo, dimentionB);

			vector<vector<int>> dimentions = getDimentions(newDimentionsMemo, tempBDimentions, adjustedDimentionAMemo);
			newDimentions = dimentions[0];

			int firstProduct = 1;
			for(int i = 0; i < lowIdx; i++){
				firstProduct *= newDimentions[i];
			}
			int secondProduct = 1;
			for(int i = lowIdx + 1; i < highIdx; i++){
				secondProduct *= newDimentions[i];
			}
			int thirdProduct = 1;
			for(int i = highIdx + 1; i < newDimentions.size(); i++){
				thirdProduct *= newDimentions[i];
			}

			if(b.rank < newDimentionsMemo.size()){
				for(int i = newDimentions.size() - 1; i >= newDimentions.size() - (newDimentionsMemo.size() - tempBDimentions.size()); i--){
					bNormalizeProdct *= newDimentions[i];
				}
			}

			for(int i = 0; i < a.rank; i++){
				realDimentionsProduct *= a.dimentions[i];
			}

			int prodA = thirdProduct;
			int prodB;
			int prodC;
			int prodD;

			int prodE;
			int prodF;
			int prodG;

			int prodH = prodA * newDimentions[highIdx];
			int prodI = prodH * secondProduct;
			int prodJ = prodI * newDimentions[lowIdx];

			if(lowIdx == adjustedDimentionAMemo){
				prodB = prodA * newDimentions[highIdx];
				prodC = prodB * secondProduct;
				prodD = prodC * newDimentionsMemo[adjustedDimentionAMemo];

				prodE = prodA * b.dimentions[dimentionB];
				prodF = prodE * secondProduct;
				prodG = prodF * newDimentionsMemo[lowIdx];
			}
			else{
				prodB = prodA * newDimentionsMemo[adjustedDimentionAMemo];
				prodC = prodB * secondProduct;
				prodD = prodC * newDimentions[lowIdx];

				prodE = prodA * newDimentionsMemo[highIdx];
				prodF = prodE * secondProduct;
				prodG = prodF * b.dimentions[dimentionB];
			}

			NumObject ans = NumObject(a.dimentions);
			for(int c = 0; c < firstProduct; c++){
				for(int d = 0; d < newDimentions[lowIdx]; d++){
					for(int e = 0; e < secondProduct; e++){
						for(int f = 0; f < newDimentions[highIdx]; f++){
							for(int g = 0; g < thirdProduct; g++){
								double sum = 0.0;
								if(lowIdx == adjustedDimentionAMemo){
									for(int h = 0; h < newDimentionsMemo[lowIdx]; h++){
										sum += seed.values[(c * prodD + h * prodC + e * prodB + f * prodA + g) % seed.values.size()] * b.values[(c * prodG + h * prodF + e * prodE + d * prodA + g) / bNormalizeProdct];
									}
								}
								else{
									for(int h = 0; h < newDimentionsMemo[highIdx]; h++){
										sum += seed.values[(c * prodD + d * prodC + e * prodB + h * prodA + g) % seed.values.size()] * b.values[(c * prodG + f * prodF + e * prodE + h * prodA + g) / bNormalizeProdct];
									}
								}
								if(ans.values.size() >= realDimentionsProduct || (c * prodJ + d * prodI + e * prodH + f * prodA + g) % realDimentionsProduct != ans.values.size()){
									ans.values[(c * prodJ + d * prodI + e * prodH + f * prodA + g) % realDimentionsProduct] += sum;
								}
								else{
									ans.values.push_back(sum);
								}
							}
						}
					}
				}
			}
			inputs[0]->derive(ans);
		}
	}


	if (typeid(*inputs[1]) != typeid(Constant)){
		vector<int> newDimentions;
		vector<int> tempADimentions;

		int bNormalizeProdct = 1;
		int realDimentionsProduct = 1;

		int lowIdx;
		int highIdx;



		if(adjustedDimentionAMemo == dimentionB){
			int firstProduct = 1;
			int secondProduct = 1;

			if(b.rank >= a.rank){
				for(int i = 0; i < dimentionB; i++){
					firstProduct *= b.dimentions[i];
				}
				for(int i = dimentionB + 1; i < b.rank; i++){
					secondProduct *= b.dimentions[i];
				}
			}
			else{
				for(int i = 0; i < dimentionA; i++){
					firstProduct *= a.dimentions[i];
				}
				for(int i = dimentionA + 1; i < a.rank; i++){
					secondProduct *= a.dimentions[i];
				}
			}

			for(int i = 0; i < b.rank; i++){
				realDimentionsProduct *= b.dimentions[i];
			}

			if(b.rank < a.rank){
				for(int i = a.rank - 1; i >= b.rank; i--){
					bNormalizeProdct *= a.dimentions[i];
				}
			}

			cout << firstProduct << ", " << secondProduct << ", " << realDimentionsProduct << ", " << bNormalizeProdct << endl;

			NumObject ans = NumObject(b.dimentions);
			for(int c = 0; c < firstProduct; c++){
				for(int d = 0; d < b.dimentions[dimentionB]; d++){
					for(int e = 0; e < secondProduct; e++){
						if(ans.values.size() >= realDimentionsProduct || ((c * secondProduct * b.dimentions[dimentionB] + d * secondProduct + e) / bNormalizeProdct) % realDimentionsProduct != ans.values.size()){
							ans.values[((c * secondProduct * b.dimentions[dimentionB] + d * secondProduct + e) / bNormalizeProdct) % realDimentionsProduct] += seed.values[(c * secondProduct + e) % seed.values.size()] * a.values[(c * secondProduct * b.dimentions[dimentionB] + d * secondProduct + e) % a.values.size()];
						}
						else{
							ans.values.push_back(seed.values[(c * secondProduct + e) % seed.values.size()] * a.values[(c * secondProduct * b.dimentions[dimentionB] + d * secondProduct + e) % a.values.size()]);
						}
					}
				}
			}
			inputs[1]->derive(ans);
		}
		else{
			if(a.rank >= b.rank){
				tempADimentions = a.dimentions;
				int temp = tempADimentions[dimentionA];
				tempADimentions[dimentionA] = tempADimentions[dimentionB];
				tempADimentions[dimentionB] = temp;
			}
			else{
				tempADimentions = b.dimentions;
				if(dimentionB < a.rank && dimentionB >= (b.rank - a.rank)){
					tempADimentions[adjustedDimentionAMemo] = a.dimentions[dimentionB];
				}
				else{
					tempADimentions[adjustedDimentionAMemo] = 1;
				}
			}

			lowIdx = min(adjustedDimentionAMemo, dimentionB);
			highIdx = max(adjustedDimentionAMemo, dimentionB);

			vector<vector<int>> dimentions = getDimentions(tempADimentions, newDimentionsMemo, adjustedDimentionAMemo);
			newDimentions = dimentions[0];

			int firstProduct = 1;
			for(int i = 0; i < lowIdx; i++){
				firstProduct *= newDimentions[i];
			}
			int secondProduct = 1;
			for(int i = lowIdx + 1; i < highIdx; i++){
				secondProduct *= newDimentions[i];
			}
			int thirdProduct = 1;
			for(int i = highIdx + 1; i < newDimentions.size(); i++){
				thirdProduct *= newDimentions[i];
			}

			if(newDimentionsMemo.size() < a.rank){
				for(int i = newDimentions.size() - 1; i >= newDimentions.size() - (tempADimentions.size() - newDimentionsMemo.size()); i--){
					bNormalizeProdct *= newDimentions[i];
				}
			}

			for(int i = 0; i < b.rank; i++){
				realDimentionsProduct *= b.dimentions[i];
			}

			int prodA = thirdProduct;
			int prodB;
			int prodC;
			int prodD;

			int prodE;
			int prodF;
			int prodG;

			int prodH = prodA * newDimentions[highIdx];
			int prodI = prodH * secondProduct;
			int prodJ = prodI * newDimentions[lowIdx];

			if(lowIdx == adjustedDimentionAMemo){
				prodB = prodA * newDimentionsMemo[dimentionB];
				prodC = prodB * secondProduct;
				prodD = prodC * a.dimentions[dimentionA];

				prodE = prodA * newDimentionsMemo[dimentionB];
				prodF = prodE * secondProduct;
				prodG = prodF * newDimentions[lowIdx];
			}
			else{
				prodB = prodA * a.dimentions[dimentionA];
				prodC = prodB * secondProduct;
				prodD = prodC * newDimentionsMemo[dimentionB];

				prodE = prodA * newDimentions[highIdx];
				prodF = prodE * secondProduct;
				prodG = prodF * newDimentionsMemo[dimentionB];
			}

			NumObject ans = NumObject(b.dimentions);
			for(int c = 0; c < firstProduct; c++){
				for(int d = 0; d < newDimentions[lowIdx]; d++){
					for(int e = 0; e < secondProduct; e++){
						for(int f = 0; f < newDimentions[highIdx]; f++){
							for(int g = 0; g < thirdProduct; g++){
								double sum = 0.0;
								if(lowIdx == adjustedDimentionAMemo){
									for(int h = 0; h < newDimentionsMemo[highIdx]; h++){
										sum += a.values[(c * prodD + f * prodC + e * prodB + h * prodA + g) % a.values.size()] * seed.values[((c * prodG + d * prodF + e * prodE + h * prodA + g) / bNormalizeProdct) % seed.values.size()];
									}
								}
								else{
									for(int h = 0; h < newDimentionsMemo[lowIdx]; h++){
										sum += a.values[(c * prodD + h * prodC + e * prodB + d * prodA + g) % a.values.size()] * seed.values[((c * prodG + h * prodF + e * prodE + f * prodA + g) / bNormalizeProdct) % seed.values.size()];
									}
								}
								if(ans.values.size() >= realDimentionsProduct || (c * prodJ + d * prodI + e * prodH + f * prodA + g) % realDimentionsProduct != ans.values.size()){
									ans.values[(c * prodJ + d * prodI + e * prodH + f * prodA + g) % realDimentionsProduct] += sum;
								}
								else{
									ans.values.push_back(sum);
								}
							}
						}
					}
				}
			}

			inputs[1]->derive(ans);
		}
	}
}

string MatMul::describe(){
	return name + "(" + inputs[0]->describe() + ", " + inputs[1]->describe() + ", " + to_string(dimentionA) + ", " + to_string(dimentionB) + ")";
}



Sum::Sum(Node* a, int dimentionVal){
	inputs.push_back(a);
	dimention = dimentionVal;
	name = "Sum";
}

NumObject Sum::getValue(){
	NumObject a = inputs[0]->getValue();

	vector<int> newDimentions;
	for(int i = 0; i < a.rank; i++){
		if (i != dimention){
			newDimentions.push_back(a.dimentions[i]);
		}
	}
	NumObject answer = NumObject(newDimentions, 0.0);

	int preSum = 1;
	for(int i = 0; i < dimention; i++){
		preSum *= a.dimentions[i];
	}
	int postSum = a.values.size() / (preSum * a.dimentions[dimention]);

	for(int i = 0; i < preSum; i++){
		for(int x = 0; x < a.dimentions[dimention]; x++){
			for(int z = 0; z < postSum; z++){
				answer.values[i * postSum + z] += a.values[i * postSum * a.dimentions[dimention] + postSum * x + z];
			}
		}
	}

	return memoize(answer);
}


void Sum::derive(NumObject& seed){
	NumObject a = inputs[0]->derivativeMemo;
	if(seed.rank < (a.rank - dimention)){
		inputs[0]->derive(seed);
	}
	else{
		NumObject temp1 = expandDerivative(seed, dimention - (a.rank - seed.rank) + 1, a.dimentions[dimention]);
		inputs[0]->derive(temp1);
	}
}

string Sum::describe(){
	return name + "(" + inputs[0]->describe() + ", " + to_string(dimention) + ")";
}

Mean::Mean(Node* a, int dimentionVal){
	inputs.push_back(a);
	dimention = dimentionVal;
	name = "Mean";
}

NumObject Mean::getValue(){
	NumObject a = inputs[0]->getValue();

	vector<int> newDimentions;
	for(int i = 0; i < a.rank; i++){
		if (i != dimention){
			newDimentions.push_back(a.dimentions[i]);
		}
	}
	NumObject answer = NumObject(newDimentions, 0.0);

	int preSum = 1;
	for(int i = 0; i < dimention; i++){
		preSum *= a.dimentions[i];
	}
	int postSum = a.values.size() / (preSum * a.dimentions[dimention]);

	for(int i = 0; i < preSum; i++){
		for(int x = 0; x < a.dimentions[dimention]; x++){
			for(int z = 0; z < postSum; z++){
				answer.values[i * postSum + z] += a.values[i * postSum * a.dimentions[dimention] + postSum * x + z];
			}
		}
	}

	vector<NumObject> items = {answer, NumObject(a.dimentions[dimention])};

	NumObject ans2 = mapVals(this, &Mean::operation, items);
	return memoize(ans2);
}

double Mean::operation(vector<double>& a){
	return a[0] / a[1];
}

void Mean::derive(NumObject& seed){
	NumObject a = inputs[0]->derivativeMemo;

	vector<NumObject> items = {seed, NumObject(a.dimentions[dimention])};
	NumObject eval1 = mapVals(this, &Mean::operation, items);

	if(eval1.rank < (a.rank - dimention)){
		inputs[0]->derive(eval1);
	}
	else{
		NumObject temp1 = expandDerivative(eval1, dimention - (a.rank - eval1.rank) + 1, a.dimentions[dimention]);
		inputs[0]->derive(temp1);
	}
}

string Mean::describe(){
	return name + "(" + inputs[0]->describe() + ", " + to_string(dimention) + ")";
}

NumObject expandDerivative(NumObject& a, int dimention, int size){
	vector<int> newDimentions;
	for(int i = 0; i < a.rank; i++){
		if(i == dimention){
			newDimentions.push_back(size);
		}
		newDimentions.push_back(a.dimentions[i]);
	}
	if(dimention == a.rank){
		newDimentions.push_back(size);
	}

	NumObject answer = NumObject(newDimentions);

	int preSum = 1;
	for(int i = 0; i < dimention; i++){
		preSum *= a.dimentions[i];
	}
	int postSum = a.values.size() / preSum;

	for(int i = 0; i < preSum; i++){
		for(int x = 0; x < size; x++){
			for(int z = 0; z < postSum; z++){
				answer.values.push_back(a.values[i * postSum + z]);
			}
		}
	}

	return answer;
}


Trans::Trans(Node* a, vector<int> permutations){
	inputs.push_back(a);
	perm = permutations;
	name = "Trans";
}

NumObject Trans::getValue(){
	NumObject a = inputs[0]->getValue();

	vector<int> newDimentions;
	newDimentions.resize(a.rank, 0);
	for(int i = 0; i < a.rank; i++){
		newDimentions[i] = a.dimentions[perm[i]];
	}

	NumObject ans = NumObject(newDimentions, 0.0);
	for(int i = 0; i < a.values.size(); i++){
		ans.values[flipIdx(i, a, ans)] = a.values[i];
	}
	return memoize(ans);
}

int Trans::flipIdx(int num, NumObject& a, NumObject& b){
	int totalSize = a.values.size();
	vector<int> idx;
	for(int i = 0; i < a.rank; i++){
		idx.push_back((num % totalSize) / (totalSize / a.dimentions[i]));
		totalSize /= a.dimentions[i];
	}

	totalSize = b.values.size();
	int newNum = 0;
	for(int i = 0; i < b.rank; i++){
		totalSize /= b.dimentions[i];
		newNum += totalSize * idx[perm[i]];
	}

	return newNum;
}

int Trans::flipIdxDerive(int num, NumObject& a, NumObject& b){
	int totalSize = a.values.size();
	vector<int> idx;
	for(int i = 0; i < a.rank; i++){
		idx.push_back((num % totalSize) / (totalSize / a.dimentions[i]));
		totalSize /= a.dimentions[i];
	}

	vector<int> newIdx;
	newIdx.resize(idx.size(), 0);
	for(int i = 0; i < idx.size(); i++){
		newIdx[perm[i]] = idx[i];
	}

	totalSize = b.values.size();
	int newNum = 0;
	for(int i = 0; i < b.rank; i++){
		totalSize /= b.dimentions[i];
		newNum += totalSize * newIdx[i];
	}

	return newNum;
}

void Trans::derive(NumObject& seed){
	NumObject a = inputs[0]->derivativeMemo;

	bool isInternalized = true;

	for(int i = 0; i < seed.rank; i++){
		bool found = false;
		for(int x = 0; x < seed.rank; x++){
			if(i == perm[a.rank - seed.rank + x]){
				found = true;
				break;
			}
		}
		if (found == false){
			isInternalized = false;
			break;
		}
	}

	if (isInternalized == true){
		vector<int> newDimentions;
		for(int i = 0; i < seed.rank; i++){
			newDimentions.push_back(a.dimentions[a.rank - seed.rank + i]);
		}

		NumObject ans = NumObject(newDimentions, 0.0);
		for(int i = 0; i < seed.values.size(); i++){
			ans.values[flipIdxDerive(i, seed, ans)] = seed.values[i];
		}
		inputs[0]->derive(ans);
	}
	else{
		NumObject ans = NumObject(a.dimentions, 0.0);
		for(int i = 0; i < derivativeMemo.values.size(); i++){
			ans.values[flipIdxDerive(i, derivativeMemo, ans)] = seed.values[i % seed.values.size()];
		}
		inputs[0]->derive(ans);
	}
}

string Trans::describe(){
	string ans = name + "(" + inputs[0]->describe() + ", [";
	for(int i = 0; i < perm.size(); i++){
		ans += to_string(perm[i]);
		if(i != perm.size() - 1){
			ans += ", ";
		}
	}
	ans += "])";
	return ans;
}


Max::Max(Node* a, int dimentionVal){
	inputs.push_back(a);
	dimention = dimentionVal;
	name = "Max";
}

NumObject Max::getValue(){
	NumObject a = inputs[0]->getValue();

	vector<int> newDimentions;
	for(int i = 0; i < a.rank; i++){
		if (i != dimention){
			newDimentions.push_back(a.dimentions[i]);
		}
	}
	double lowVal = -numeric_limits<double>::infinity();
	NumObject answer = NumObject(newDimentions, lowVal);
	idx = NumObject(newDimentions, 0.0);

	int preSum = 1;
	for(int i = 0; i < dimention; i++){
		preSum *= a.dimentions[i];
	}
	int postSum = a.values.size() / (preSum * a.dimentions[dimention]);

	for(int i = 0; i < preSum; i++){
		for(int x = 0; x < a.dimentions[dimention]; x++){
			for(int z = 0; z < postSum; z++){
				if(a.values[i * postSum * a.dimentions[dimention] + postSum * x + z] > answer.values[i * postSum + z]){
					answer.values[i * postSum + z] = a.values[i * postSum * a.dimentions[dimention] + postSum * x + z];
					idx.values[i * postSum + z] = x;
				}
			}
		}
	}
	return memoize(answer);
}

void Max::derive(NumObject& seed){
	NumObject a = inputs[0]->derivativeMemo;
	NumObject ans = NumObject(a.dimentions);

	int preSum = 1;
	for(int i = 0; i < dimention; i++){
		preSum *= a.dimentions[i];
	}
	int postSum = a.values.size() / (preSum * a.dimentions[dimention]);


	if(seed.rank < (a.rank - dimention)){
		for(int i = 0; i < preSum; i++){
			for(int x = 0; x < a.dimentions[dimention]; x++){
				for(int z = 0; z < postSum; z++){
					if(idx.values[i * postSum + z] == x){
						ans.values.push_back(seed.values[(i * postSum + z) % seed.values.size()]);
					}
					else{
						ans.values.push_back(0.0);
					}
				}
			}
		}
	}
	else{
		NumObject temp1 = expandDerivative(seed, dimention - (a.rank - seed.rank) + 1, a.dimentions[dimention]);
		for(int i = 0; i < preSum; i++){
			for(int x = 0; x < a.dimentions[dimention]; x++){
				for(int z = 0; z < postSum; z++){
					if(idx.values[i * postSum + z] == x){
						ans.values.push_back(temp1.values[(i * postSum * a.dimentions[dimention] + postSum * x + z) % temp1.values.size()]);
					}
					else{
						ans.values.push_back(0.0);
					}
				}
			}
		}
	}

	inputs[0]->derive(ans);
}

string Max::describe(){
	return name + "(" + inputs[0]->describe() + ", " + to_string(dimention) + ")";
}

NumObject Min::getValue(){
	NumObject a = inputs[0]->getValue();

	vector<int> newDimentions;
	for(int i = 0; i < a.rank; i++){
		if (i != dimention){
			newDimentions.push_back(a.dimentions[i]);
		}
	}
	double lowVal = numeric_limits<double>::infinity();
	NumObject answer = NumObject(newDimentions, lowVal);
	idx = NumObject(newDimentions, 0.0);

	int preSum = 1;
	for(int i = 0; i < dimention; i++){
		preSum *= a.dimentions[i];
	}
	int postSum = a.values.size() / (preSum * a.dimentions[dimention]);

	for(int i = 0; i < preSum; i++){
		for(int x = 0; x < a.dimentions[dimention]; x++){
			for(int z = 0; z < postSum; z++){
				if(a.values[i * postSum * a.dimentions[dimention] + postSum * x + z] < answer.values[i * postSum + z]){
					answer.values[i * postSum + z] = a.values[i * postSum * a.dimentions[dimention] + postSum * x + z];
					idx.values[i * postSum + z] = x;
				}
			}
		}
	}
	return memoize(answer);
}


