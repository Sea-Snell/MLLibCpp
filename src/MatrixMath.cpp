#include "MatrixMath.hpp"
#include "Node.hpp"
#include <limits>

MatMul::MatMul(Node* a, Node* b, int dimentionAVal, int dimentionBVal){
	inputs.push_back(a);
	inputs.push_back(b);
	name = "MatMul";
	dimentionA = dimentionAVal;
	dimentionB = dimentionBVal;
}

void MatMul::getValue(){
	inputs[0]->getValue();
	inputs[1]->getValue();

	if(lowIdx == adjustedDimentionA){
		for(int c = 0; c < firstProduct; c++){
			for(int e = 0; e < secondProduct; e++){
				for(int g = 0; g < thirdProduct; g++){
					for(int d = 0; d < newDimentions[lowIdx]; d++){
						for(int f = 0; f < newDimentions[highIdx]; f++){
							double sum = 0.0;
							for(int h = 0; h < inputs[0]->outDimentions[dimentionA]; h++){
								sum += inputs[0]->derivativeMemo[(c * prodD + h * prodC + e * prodB + f * prodA + g) % inputs[0]->outSize] * inputs[1]->derivativeMemo[(c * prodG + d * prodF + e * prodE + h * prodA + g) / bNormalizeProdct];
							}
							derivativeMemo[c * prodJ + d * prodI + e * prodH + f * prodA + g] = sum;
						}
					}
				}
			}
		}
	}
	else{
		for(int c = 0; c < firstProduct; c++){
			for(int e = 0; e < secondProduct; e++){
				for(int g = 0; g < thirdProduct; g++){
					for(int d = 0; d < newDimentions[lowIdx]; d++){
						for(int f = 0; f < newDimentions[highIdx]; f++){
							double sum = 0.0;
							for(int h = 0; h < inputs[0]->outDimentions[dimentionA]; h++){
								sum += inputs[0]->derivativeMemo[(c * prodD + d * prodC + e * prodB + h * prodA + g) % inputs[0]->outSize] * inputs[1]->derivativeMemo[(c * prodG + h * prodF + e * prodE + f * prodA + g) / bNormalizeProdct];
							}
							derivativeMemo[c * prodJ + d * prodI + e * prodH + f * prodA + g] = sum;
						}
					}
				}
			}
		}
	}
}

void MatMul::getValueDimentions(){
	inputs[0]->getValueDimentions();
	inputs[1]->getValueDimentions();

	bNormalizeProdct = 1;

	adjustedDimentionA = dimentionA;

	if(inputs[0]->outRank < inputs[1]->outRank){
		adjustedDimentionA = dimentionA + (inputs[1]->outRank - inputs[0]->outRank);
		lowIdx = min(adjustedDimentionA, dimentionB);
		highIdx = max(adjustedDimentionA, dimentionB);	
	}
	else{
		lowIdx = min(dimentionA, dimentionB);
		highIdx = max(dimentionA, dimentionB);
	}

	vector<vector<int>> dimentions = getDimentions(inputs[0]->outDimentions, inputs[1]->outDimentions, adjustedDimentionA);
	newDimentions = dimentions[0];

	outDimentions = dimentions[1];
	outRank = outDimentions.size();
	outSize = 1;
	for(int i = 0; i < outRank; i++){
		outSize *= outDimentions[i];
	}

	derivativeMemo.clear();
	derivativeMemo.resize(outSize, 0.0);


	firstProduct = 1;
	for(int i = 0; i < lowIdx; i++){
		firstProduct *= newDimentions[i];
	}
	secondProduct = 1;
	for(int i = lowIdx + 1; i < highIdx; i++){
		secondProduct *= newDimentions[i];
	}
	thirdProduct = 1;
	for(int i = highIdx + 1; i < newDimentions.size(); i++){
		thirdProduct *= newDimentions[i];
	}

	if(inputs[1]->outRank < inputs[0]->outRank){
		for(int i = newDimentions.size() - 1; i >= newDimentions.size() - (inputs[0]->outRank - inputs[1]->outRank); i--){
			bNormalizeProdct *= newDimentions[i];
		}
	}

	prodA = thirdProduct;

	prodH = prodA * newDimentions[highIdx];
	prodI = prodH * secondProduct;
	prodJ = prodI * newDimentions[lowIdx];

	if(lowIdx == adjustedDimentionA){
		prodB = prodA * newDimentions[highIdx];
		prodC = prodB * secondProduct;
		prodD = prodC * inputs[0]->outDimentions[dimentionA];

		prodE = prodA * inputs[1]->outDimentions[dimentionB];
		prodF = prodE * secondProduct;
		prodG = prodF * newDimentions[lowIdx];
	}
	else{
		prodB = prodA * inputs[0]->outDimentions[dimentionA];
		prodC = prodB * secondProduct;
		prodD = prodC * newDimentions[lowIdx];

		prodE = prodA * newDimentions[highIdx];
		prodF = prodE * secondProduct;
		prodG = prodF * inputs[1]->outDimentions[dimentionB];
	}
}


vector<vector<int>> MatMul::getDimentions(vector<int> a, vector<int> b, int adjustedDimentionAVal){
	vector<int> newDimentionsVal;
	vector<int> realDimentions;

	if(adjustedDimentionAVal == dimentionB){
		if(a.size() >= b.size()){
			for(int i = 0; i < a.size(); i++){
				if(i != dimentionA){
					newDimentionsVal.push_back(a[i]);
					realDimentions.push_back(a[i]);
				}
				else{
					newDimentionsVal.push_back(1);
				}
			}
		}
		else{
			for(int i = 0; i < b.size(); i++){
				if(i != dimentionB){
					newDimentionsVal.push_back(b[i]);
					realDimentions.push_back(b[i]);
				}
				else{
					newDimentionsVal.push_back(1);
				}
			}
		}
	}
	else{
		if(a.size() >= b.size()){
			for(int i = 0; i < a.size(); i++){
				if(i == adjustedDimentionAVal){
					if(i < b.size()){
						newDimentionsVal.push_back(b[i]);
						realDimentions.push_back(b[i]);
					}
					else{
						newDimentionsVal.push_back(1);
					}
				}
				else{
					newDimentionsVal.push_back(a[i]);
					realDimentions.push_back(a[i]);
				}
			}
		}
		else{
			for(int i = 0; i < b.size(); i++){
				if(i == dimentionB){
					if(i >= (b.size() - a.size())){
						newDimentionsVal.push_back(a[i - (b.size() - a.size())]);
						realDimentions.push_back(a[i - (b.size() - a.size())]);
					}
					else{
						newDimentionsVal.push_back(1);
					}
				}
				else{
					newDimentionsVal.push_back(b[i]);
					realDimentions.push_back(b[i]);
				}
			}
		}
	}

	vector<vector<int>> returnVal = {newDimentionsVal, realDimentions};
	return returnVal;
}


void MatMul::derive(vector<double>& seed){
	if (typeid(*inputs[0]) != typeid(Constant)){
		for(int i = 0; i < inputs[0]->outSize; i++){
			ans1[i] = 0.0;
		}

		if(adjustedDimentionA == dimentionB){
			for(int c = 0; c < firstProduct1; c++){
				for(int d = 0; d < inputs[0]->outDimentions[dimentionA]; d++){
					for(int e = 0; e < secondProduct1; e++){
						ans1[(c * secondProduct1 * inputs[0]->outDimentions[dimentionA] + d * secondProduct1 + e) % inputs[0]->outSize] += seed[(c * secondProduct1 + e) % seedSize] * inputs[1]->derivativeMemo[((c * secondProduct1 * inputs[0]->outDimentions[dimentionA] + d * secondProduct1 + e) / bNormalizeProdct1) % inputs[1]->outSize];
					}
				}
			}
		}
		else{
			if(lowIdx == adjustedDimentionA){
				for(int c = 0; c < firstProduct1; c++){
					for(int e = 0; e < secondProduct1; e++){
						for(int g = 0; g < thirdProduct1; g++){
							for(int d = 0; d < newDimentions1[lowIdx]; d++){
								for(int f = 0; f < newDimentions1[highIdx]; f++){
									double sum = 0.0;
									for(int h = 0; h < newDimentions[lowIdx]; h++){
										sum += seed[(c * prodD1 + h * prodC1 + e * prodB1 + f * prodA1 + g) % seedSize] * inputs[1]->derivativeMemo[(c * prodG1 + h * prodF1 + e * prodE1 + d * prodA1 + g) / bNormalizeProdct1];
									}
									ans1[(c * prodJ1 + d * prodI1 + e * prodH1 + f * prodA1 + g) % inputs[0]->outSize] += sum;
								}
							}
						}
					}
				}
			}
			else{
				for(int c = 0; c < firstProduct1; c++){
					for(int e = 0; e < secondProduct1; e++){
						for(int g = 0; g < thirdProduct1; g++){
							for(int d = 0; d < newDimentions1[lowIdx]; d++){
								for(int f = 0; f < newDimentions1[highIdx]; f++){
									double sum = 0.0;
									for(int h = 0; h < newDimentions[highIdx]; h++){
										sum += seed[(c * prodD1 + d * prodC1 + e * prodB1 + h * prodA1 + g) % seedSize] * inputs[1]->derivativeMemo[(c * prodG1 + f * prodF1 + e * prodE1 + h * prodA1 + g) / bNormalizeProdct1];
									}
									ans1[(c * prodJ1 + d * prodI1 + e * prodH1 + f * prodA1 + g) % inputs[0]->outSize] += sum;
								}
							}
						}
					}
				}
			}
		}
		inputs[0]->derive(ans1);
	}


	if (typeid(*inputs[1]) != typeid(Constant)){
		for(int i = 0; i < inputs[1]->outSize; i++){
			ans2[i] = 0.0;
		}

		if(adjustedDimentionA == dimentionB){
			for(int c = 0; c < firstProduct2; c++){
				for(int d = 0; d < inputs[1]->outDimentions[dimentionB]; d++){
					for(int e = 0; e < secondProduct2; e++){
						ans2[((c * secondProduct2 * inputs[1]->outDimentions[dimentionB] + d * secondProduct2 + e) / bNormalizeProdct2) % inputs[1]->outSize] += seed[(c * secondProduct2 + e) % seedSize] * inputs[0]->derivativeMemo[(c * secondProduct2 * inputs[1]->outDimentions[dimentionB] + d * secondProduct2 + e) % inputs[0]->outSize];
					}
				}
			}
		}
		else{
			if(lowIdx == adjustedDimentionA){
				for(int c = 0; c < firstProduct2; c++){
					for(int e = 0; e < secondProduct2; e++){
						for(int g = 0; g < thirdProduct2; g++){
							for(int d = 0; d < newDimentions2[lowIdx]; d++){
								for(int f = 0; f < newDimentions2[highIdx]; f++){
									double sum = 0.0;
									for(int h = 0; h < newDimentions[highIdx]; h++){
										sum += inputs[0]->derivativeMemo[(c * prodD2 + f * prodC2 + e * prodB2 + h * prodA2 + g) % inputs[0]->outSize] * seed[((c * prodG2 + d * prodF2 + e * prodE2 + h * prodA2 + g) / bNormalizeProdct2) % seedSize];
									}
									ans2[(c * prodJ2 + d * prodI2 + e * prodH2 + f * prodA2 + g) % inputs[1]->outSize] += sum;
								}
							}
						}
					}
				}
			}
			else{
				for(int c = 0; c < firstProduct2; c++){
					for(int e = 0; e < secondProduct2; e++){
						for(int g = 0; g < thirdProduct2; g++){
							for(int d = 0; d < newDimentions2[lowIdx]; d++){
								for(int f = 0; f < newDimentions2[highIdx]; f++){
									double sum = 0.0;
									for(int h = 0; h < newDimentions[lowIdx]; h++){
										sum += inputs[0]->derivativeMemo[(c * prodD2 + h * prodC2 + e * prodB2 + d * prodA2 + g) % inputs[0]->outSize] * seed[((c * prodG2 + h * prodF2 + e * prodE2 + f * prodA2 + g) / bNormalizeProdct2) % seedSize];
									}
									ans2[(c * prodJ2 + d * prodI2 + e * prodH2 + f * prodA2 + g) % inputs[1]->outSize] += sum;
								}
							}
						}
					}
				}
			}
		}
		inputs[1]->derive(ans2);
	}
}

void MatMul::deriveDimentions(vector<int>& seedDimentionsVal){
	seedDimentions = seedDimentionsVal;
	seedRank = seedDimentions.size();
	seedSize = 1;
	for(int i = 0; i < seedRank; i++){
		seedSize *= seedDimentions[i];
	}


	bNormalizeProdct1 = 1;

	ans1.clear();
	ans1.resize(inputs[0]->outSize, 0.0);

	if(adjustedDimentionA == dimentionB){
		firstProduct1 = 1;
		secondProduct1 = 1;

		if(inputs[0]->outRank >= inputs[1]->outRank){
			for(int i = 0; i < dimentionA; i++){
				firstProduct1 *= inputs[0]->outDimentions[i];
			}
			for(int i = dimentionA + 1; i < inputs[0]->outRank; i++){
				secondProduct1 *= inputs[0]->outDimentions[i];
			}
		}
		else{
			for(int i = 0; i < dimentionB; i++){
				firstProduct1 *= inputs[1]->outDimentions[i];
			}
			for(int i = dimentionB + 1; i < inputs[1]->outRank; i++){
				secondProduct1 *= inputs[1]->outDimentions[i];
			}
		}

		if(inputs[1]->outRank < inputs[0]->outRank){
			for(int i = inputs[0]->outRank - 1; i >= inputs[1]->outRank; i--){
				bNormalizeProdct1 *= inputs[0]->outDimentions[i];
			}
		}

		inputs[0]->deriveDimentions(inputs[0]->outDimentions);
	}
	else{
		if(inputs[1]->outRank >= inputs[0]->outRank){
			tempBDimentions1 = inputs[1]->outDimentions;
			int temp = tempBDimentions1[dimentionB];
			tempBDimentions1[dimentionB] = tempBDimentions1[dimentionA + (inputs[1]->outRank - inputs[0]->outRank)];
			tempBDimentions1[dimentionA + (inputs[1]->outRank - inputs[0]->outRank)] = temp;
		}
		else{
			tempBDimentions1 = inputs[0]->outDimentions;
			if(dimentionA < inputs[1]->outRank){
				tempBDimentions1[dimentionB] = inputs[1]->outDimentions[dimentionA];
			}
			else{
				tempBDimentions1[dimentionB] = 1;
			}
		}

		vector<vector<int>> dimentions = getDimentions(newDimentions, tempBDimentions1, adjustedDimentionA);
		newDimentions1 = dimentions[0];

		firstProduct1 = 1;
		for(int i = 0; i < lowIdx; i++){
			firstProduct1 *= newDimentions1[i];
		}
		secondProduct1 = 1;
		for(int i = lowIdx + 1; i < highIdx; i++){
			secondProduct1 *= newDimentions1[i];
		}
		thirdProduct1 = 1;
		for(int i = highIdx + 1; i < newDimentions1.size(); i++){
			thirdProduct1 *= newDimentions1[i];
		}

		if(inputs[1]->outRank < newDimentions.size()){
			for(int i = newDimentions1.size() - 1; i >= newDimentions1.size() - (newDimentions.size() - tempBDimentions1.size()); i--){
				bNormalizeProdct1 *= newDimentions1[i];
			}
		}

		prodA1 = thirdProduct1;

		prodH1 = prodA1 * newDimentions1[highIdx];
		prodI1 = prodH1 * secondProduct1;
		prodJ1 = prodI1 * newDimentions1[lowIdx];

		if(lowIdx == adjustedDimentionA){
			prodB1 = prodA1 * newDimentions1[highIdx];
			prodC1 = prodB1 * secondProduct1;
			prodD1 = prodC1 * newDimentions[adjustedDimentionA];

			prodE1 = prodA1 * inputs[1]->outDimentions[dimentionB];
			prodF1 = prodE1 * secondProduct1;
			prodG1 = prodF1 * newDimentions[lowIdx];
		}
		else{
			prodB1 = prodA1 * newDimentions[adjustedDimentionA];
			prodC1 = prodB1 * secondProduct1;
			prodD1 = prodC1 * newDimentions1[lowIdx];

			prodE1 = prodA1 * newDimentions[highIdx];
			prodF1 = prodE1 * secondProduct1;
			prodG1 = prodF1 * inputs[1]->outDimentions[dimentionB];
		}

		inputs[0]->deriveDimentions(inputs[0]->outDimentions);
	}

	bNormalizeProdct2 = 1;

	ans2.clear();
	ans2.resize(inputs[1]->outSize, 0.0);

	if(adjustedDimentionA == dimentionB){
		firstProduct2 = 1;
		secondProduct2 = 1;

		if(inputs[1]->outRank >= inputs[0]->outRank){
			for(int i = 0; i < dimentionB; i++){
				firstProduct2 *= inputs[1]->outDimentions[i];
			}
			for(int i = dimentionB + 1; i < inputs[1]->outRank; i++){
				secondProduct2 *= inputs[1]->outDimentions[i];
			}
		}
		else{
			for(int i = 0; i < dimentionA; i++){
				firstProduct2 *= inputs[0]->outDimentions[i];
			}
			for(int i = dimentionA + 1; i < inputs[0]->outRank; i++){
				secondProduct2 *= inputs[0]->outDimentions[i];
			}
		}

		if(inputs[1]->outRank < inputs[0]->outRank){
			for(int i = inputs[0]->outRank - 1; i >= inputs[1]->outRank; i--){
				bNormalizeProdct2 *= inputs[0]->outDimentions[i];
			}
		}

		inputs[1]->deriveDimentions(inputs[1]->outDimentions);
	}
	else{
		if(inputs[0]->outRank >= inputs[1]->outRank){
			tempADimentions2 = inputs[0]->outDimentions;
			int temp = tempADimentions2[dimentionA];
			tempADimentions2[dimentionA] = tempADimentions2[dimentionB];
			tempADimentions2[dimentionB] = temp;
		}
		else{
			tempADimentions2 = inputs[1]->outDimentions;
			if(dimentionB < inputs[0]->outRank && dimentionB >= (inputs[1]->outRank - inputs[0]->outRank)){
				tempADimentions2[adjustedDimentionA] = inputs[0]->outDimentions[dimentionB];
			}
			else{
				tempADimentions2[adjustedDimentionA] = 1;
			}
		}

		vector<vector<int>> dimentions = getDimentions(tempADimentions2, newDimentions, adjustedDimentionA);
		newDimentions2 = dimentions[0];

		firstProduct2 = 1;
		for(int i = 0; i < lowIdx; i++){
			firstProduct2 *= newDimentions2[i];
		}
		secondProduct2 = 1;
		for(int i = lowIdx + 1; i < highIdx; i++){
			secondProduct2 *= newDimentions2[i];
		}
		thirdProduct2 = 1;
		for(int i = highIdx + 1; i < newDimentions2.size(); i++){
			thirdProduct2 *= newDimentions2[i];
		}

		if(newDimentions.size() < inputs[0]->outRank){
			for(int i = newDimentions2.size() - 1; i >= newDimentions2.size() - (tempADimentions2.size() - newDimentions.size()); i--){
				bNormalizeProdct2 *= newDimentions2[i];
			}
		}

		prodA2 = thirdProduct2;

		prodH2 = prodA2 * newDimentions2[highIdx];
		prodI2 = prodH2 * secondProduct2;
		prodJ2 = prodI2 * newDimentions2[lowIdx];

		if(lowIdx == adjustedDimentionA){
			prodB2 = prodA2 * newDimentions[dimentionB];
			prodC2 = prodB2 * secondProduct2;
			prodD2 = prodC2 * inputs[0]->outDimentions[dimentionA];

			prodE2 = prodA2 * newDimentions[dimentionB];
			prodF2 = prodE2 * secondProduct2;
			prodG2 = prodF2 * newDimentions2[lowIdx];
		}
		else{
			prodB2 = prodA2 * inputs[0]->outDimentions[dimentionA];
			prodC2 = prodB2 * secondProduct2;
			prodD2 = prodC2 * newDimentions[dimentionB];

			prodE2 = prodA2 * newDimentions2[highIdx];
			prodF2 = prodE2 * secondProduct2;
			prodG2 = prodF2 * newDimentions[dimentionB];
		}

		inputs[1]->deriveDimentions(inputs[1]->outDimentions);
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

void Sum::getValue(){
	inputs[0]->getValue();

	for(int i = 0; i < preSum; i++){
		for(int z = 0; z < postSum; z++){
			double sum = 0.0;
			for(int x = 0; x < inputs[0]->outDimentions[dimention]; x++){
				sum += inputs[0]->derivativeMemo[i * postSum * inputs[0]->outDimentions[dimention] + postSum * x + z];
			}
			derivativeMemo[i * postSum + z] = sum;
		}
	}
}


void Sum::getValueDimentions(){
	inputs[0]->getValueDimentions();

	outDimentions.clear();
	for(int i = 0; i < inputs[0]->outRank; i++){
		if (i != dimention){
			outDimentions.push_back(inputs[0]->outDimentions[i]);
		}
	}
	outRank = inputs[0]->outRank - 1;
	outSize = inputs[0]->outSize / inputs[0]->outDimentions[dimention];

	preSum = 1;
	for(int i = 0; i < dimention; i++){
		preSum *= outDimentions[i];
	}
	postSum = outSize / preSum;

	derivativeMemo.clear();
	derivativeMemo.resize(outSize, 0.0);
}

void Sum::derive(vector<double>& seed){
	if(seedRank < (inputs[0]->outRank - dimention)){
		inputs[0]->derive(seed);
	}
	else{
		for(int i = 0; i < preSum; i++){
			for(int x = 0; x < inputs[0]->outDimentions[dimention]; x++){
				for(int z = 0; z < postSum; z++){
					ans[i * postSum * inputs[0]->outDimentions[dimention] + x * postSum + z] = seed[(i * postSum + z) % seedSize];
				}
			}
		}
		inputs[0]->derive(ans);
	}
}

void Sum::deriveDimentions(vector<int>& seedDimentionsVal){
	seedDimentions = seedDimentionsVal;
	seedRank = seedDimentions.size();
	seedSize = 1;
	for(int i = 0; i < seedRank; i++){
		seedSize *= seedDimentions[i];
	}

	if(seedRank < (inputs[0]->outRank - dimention)){
		inputs[0]->deriveDimentions(seedDimentions);
	}
	else{
		adjustedDimention = dimention - (inputs[0]->outRank - seedRank) + 1;

		ansDimentions.clear();
		for(int i = 0; i < seedRank; i++){
			if(i == adjustedDimention){
				ansDimentions.push_back(inputs[0]->outDimentions[dimention]);
			}
			ansDimentions.push_back(seedDimentions[i]);
		}
		if(adjustedDimention == seedRank){
			ansDimentions.push_back(inputs[0]->outDimentions[dimention]);
		}

		ansRank = ansDimentions.size();
		ansSize = 1;
		for(int i = 0; i < ansRank; i++){
			ansSize *= ansDimentions[i];
		}

		ans.clear();
		ans.resize(ansSize, 0.0);

		preSum1 = 1;
		for(int i = 0; i < adjustedDimention; i++){
			preSum1 *= seedDimentions[i];
		}
		postSum1 = seedSize / preSum1;

		inputs[0]->deriveDimentions(ansDimentions);
	}
}

string Sum::describe(){
	return name + "(" + inputs[0]->describe() + ", " + to_string(dimention) + ")";
}

void Mean::getValue(){
	inputs[0]->getValue();

	for(int i = 0; i < preSum; i++){
		for(int z = 0; z < postSum; z++){
			double sum = 0.0;
			for(int x = 0; x < inputs[0]->outDimentions[dimention]; x++){
				sum += inputs[0]->derivativeMemo[i * postSum * inputs[0]->outDimentions[dimention] + postSum * x + z];
			}
			derivativeMemo[i * postSum + z] = sum / inputs[0]->outDimentions[dimention];
		}
	}
}

void Mean::derive(vector<double>& seed){
	for(int i = 0; i < seedSize; i++){
		temp[i] = seed[i] / inputs[0]->outDimentions[dimention];
	}

	if(seedRank < (inputs[0]->outRank - dimention)){
		inputs[0]->derive(temp);
	}
	else{
		for(int i = 0; i < preSum; i++){
			for(int x = 0; x < inputs[0]->outDimentions[dimention]; x++){
				for(int z = 0; z < postSum; z++){
					ans[i * postSum * inputs[0]->outDimentions[dimention] + x * postSum + z] = temp[(i * postSum + z) % seedSize];
				}
			}
		}
		inputs[0]->derive(ans);
	}
}

void Mean::deriveDimentions(vector<int>& seedDimentionsVal){
	seedDimentions = seedDimentionsVal;
	seedRank = seedDimentions.size();
	seedSize = 1;
	for(int i = 0; i < seedRank; i++){
		seedSize *= seedDimentions[i];
	}

	temp.clear();
	temp.resize(seedSize, 0.0);

	if(seedRank < (inputs[0]->outRank - dimention)){
		inputs[0]->deriveDimentions(seedDimentions);
	}
	else{
		adjustedDimention = dimention - (inputs[0]->outRank - seedRank) + 1;

		ansDimentions.clear();
		for(int i = 0; i < seedRank; i++){
			if(i == adjustedDimention){
				ansDimentions.push_back(inputs[0]->outDimentions[dimention]);
			}
			ansDimentions.push_back(seedDimentions[i]);
		}
		if(adjustedDimention == seedRank){
			ansDimentions.push_back(inputs[0]->outDimentions[dimention]);
		}

		ansRank = ansDimentions.size();
		ansSize = 1;
		for(int i = 0; i < ansRank; i++){
			ansSize *= ansDimentions[i];
		}

		ans.clear();
		ans.resize(ansSize, 0.0);

		preSum1 = 1;
		for(int i = 0; i < adjustedDimention; i++){
			preSum1 *= seedDimentions[i];
		}
		postSum1 = seedSize / preSum1;

		inputs[0]->deriveDimentions(ansDimentions);
	}
}

string Mean::describe(){
	return name + "(" + inputs[0]->describe() + ", " + to_string(dimention) + ")";
}


Trans::Trans(Node* a, vector<int> permutations){
	inputs.push_back(a);
	perm = permutations;
	name = "Trans";
}

void Trans::getValue(){
	inputs[0]->getValue();

	for(int i = 0; i < inputs[0]->outSize; i++){
		int totalSize = inputs[0]->outSize;
		int newNum = 0;

		for(int x = 0; x < inputs[0]->outRank; x++){
			idx[x] = (i % totalSize) / (totalSize / inputs[0]->outDimentions[x]);
			totalSize /= inputs[0]->outDimentions[x];
		}

		totalSize = outSize;
		for(int x = 0; x < outRank; x++){
			totalSize /= outDimentions[x];
			newNum += totalSize * idx[perm[x]];
		}

		derivativeMemo[newNum] = inputs[0]->derivativeMemo[i];
	}
}

void Trans::getValueDimentions(){
	inputs[0]->getValueDimentions();

	outDimentions.clear();
	outDimentions.resize(inputs[0]->outRank, 0);
	for(int i = 0; i < inputs[0]->outRank; i++){
		outDimentions[i] = inputs[0]->outDimentions[perm[i]];
	}
	outRank = inputs[0]->outRank;
	outSize = inputs[0]->outSize;

	idx.clear();
	idx.resize(inputs[0]->outRank, 0);
	derivativeMemo.clear();
	derivativeMemo.resize(outSize, 0.0);
}

void Trans::derive(vector<double>& seed){
	if (isInternalized == true){
		for(int i = 0; i < seedSize; i++){

			int totalSize = seedSize;
			for(int x = 0; x < seedRank; x++){
				idx1[x] = (i % totalSize) / (totalSize / seedDimentions[x]);
				totalSize /= seedDimentions[x];
			}

			for(int x = 0; x < seedRank; x++){
				newIdx1[perm[x]] = idx1[x];
			}

			totalSize = ansSize;
			int newNum = 0;
			for(int x = 0; x < ansRank; x++){
				totalSize /= ansDimentions[x];
				newNum += totalSize * newIdx1[x];
			}

			ans[newNum] = seed[i];
		}
	}
	else{
		for(int i = 0; i < outSize; i++){
			int totalSize = outSize;
			for(int x = 0; x < outRank; x++){
				idx1[x] = (i % totalSize) / (totalSize / outDimentions[x]);
				totalSize /= outDimentions[x];
			}

			for(int i = 0; i < outRank; i++){
				newIdx1[perm[i]] = idx1[i];
			}

			totalSize = ansSize;
			int newNum = 0;
			for(int i = 0; i < outRank; i++){
				totalSize /= outDimentions[i];
				newNum += totalSize * newIdx1[i];
			}

			ans[newNum] = seed[i % seedSize];
		}
	}
	inputs[0]->derive(ans);
}

void Trans::deriveDimentions(vector<int>& seedDimentionsVal){
	seedDimentions = seedDimentionsVal;
	seedRank = seedDimentions.size();
	seedSize = 1;
	for(int i = 0; i < seedRank; i++){
		seedSize *= seedDimentions[i];
	}

	isInternalized = true;

	for(int i = 0; i < seedRank; i++){
		bool found = false;
		for(int x = 0; x < seedRank; x++){
			if(i == perm[inputs[0]->outRank - seedRank + x]){
				found = true;
				break;
			}
		}
		if (found == false){
			isInternalized = false;
			break;
		}
	}

	idx1.clear();
	idx1.resize(seedRank, 0);
	newIdx1.clear();
	newIdx1.resize(seedRank, 0);

	if (isInternalized == true){
		ansSize = 1;
		ansDimentions.clear();
		for(int i = 0; i < seedRank; i++){
			ansDimentions.push_back(inputs[0]->outDimentions[inputs[0]->outRank - seedRank + i]);
			ansSize *= ansDimentions[i];
		}
		ansRank = seedRank;
		ans.clear();
		ans.resize(ansSize, 0.0);

		inputs[0]->deriveDimentions(ansDimentions);
	}
	else{
		idx1.clear();
		idx1.resize(outRank, 0);
		newIdx1.clear();
		newIdx1.resize(outRank, 0);

		inputs[0]->deriveDimentions(inputs[0]->outDimentions);
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

void Max::getValue(){
	inputs[0]->getValue();

	double lowVal = -numeric_limits<double>::infinity();
	for(int i = 0; i < outSize; i++){
		derivativeMemo[i] = lowVal;
	}

	for(int i = 0; i < preSum; i++){
		for(int x = 0; x < inputs[0]->outDimentions[dimention]; x++){
			for(int z = 0; z < postSum; z++){
				if(inputs[0]->derivativeMemo[i * postSum * inputs[0]->outDimentions[dimention] + postSum * x + z] > derivativeMemo[i * postSum + z]){
					derivativeMemo[i * postSum + z] = inputs[0]->derivativeMemo[i * postSum * inputs[0]->outDimentions[dimention] + postSum * x + z];
					idx[i * postSum + z] = x;
				}
			}
		}
	}
}

void Max::getValueDimentions(){
	inputs[0]->getValueDimentions();

	outDimentions.clear();
	for(int i = 0; i < inputs[0]->outRank; i++){
		if(i != dimention){
			outDimentions.push_back(inputs[0]->outDimentions[i]);
		}
	}
	outRank = inputs[0]->outRank - 1;
	outSize = inputs[0]->outSize / inputs[0]->outDimentions[dimention];

	double lowVal = -numeric_limits<double>::infinity();
	derivativeMemo.clear();
	derivativeMemo.resize(outSize, lowVal);
	idx.clear();
	idx.resize(outSize, 0);

	preSum = 1;
	for(int i = 0; i < dimention; i++){
		preSum *= inputs[0]->outDimentions[i];
	}
	postSum = outSize / preSum;
}

void Max::derive(vector<double>& seed){
	if(seedRank < (inputs[0]->outRank - dimention)){
		for(int i = 0; i < preSum; i++){
			for(int x = 0; x < inputs[0]->outDimentions[dimention]; x++){
				for(int z = 0; z < postSum; z++){
					ans[i * postSum * inputs[0]->outDimentions[dimention] + x * postSum + z] = seed[(i * postSum + z) % seedSize] * (int)(idx[i * postSum + z] == x);
				}
			}
		}
	}
	else{
		for(int i = 0; i < preSum1; i++){
			for(int x = 0; x < inputs[0]->outDimentions[dimention]; x++){
				for(int z = 0; z < postSum1; z++){
					temp1[i * postSum * inputs[0]->outDimentions[dimention] + x * postSum + z] = seed[i * postSum + z];
				}
			}
		}

		for(int i = 0; i < preSum; i++){
			for(int x = 0; x < inputs[0]->outDimentions[dimention]; x++){
				for(int z = 0; z < postSum; z++){
					ans[i * postSum * inputs[0]->outDimentions[dimention] + x * postSum + z] = temp1[(i * postSum * inputs[0]->outDimentions[dimention] + x * postSum + z) % tempSize1] * (int)(idx[i * postSum + z] == x);
				}
			}
		}
	}

	inputs[0]->derive(ans);
}

string Max::describe(){
	return name + "(" + inputs[0]->describe() + ", " + to_string(dimention) + ")";
}

void Max::deriveDimentions(vector<int>& seedDimentionsVal){
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

	adjustedDimention = dimention - (inputs[0]->outRank - seedRank) + 1;

	tempDimentions1.clear();
	for(int i = 0; i < seedRank; i++){
		if(i == adjustedDimention){
			tempDimentions1.push_back(inputs[0]->outDimentions[dimention]);
		}
		tempDimentions1.push_back(seedDimentions[i]);
	}
	if(adjustedDimention == seedRank){
		tempDimentions1.push_back(inputs[0]->outDimentions[dimention]);
	}

	tempRank1 = tempDimentions1.size();
	tempSize1 = 1;
	for(int i = 0; i < tempRank1; i++){
		tempSize1 *= tempDimentions1[i];
	}

	temp1.clear();
	temp1.resize(tempSize1, 0.0);

	preSum1 = 1;
	for(int i = 0; i < adjustedDimention; i++){
		preSum1 *= seedDimentions[i];
	}
	postSum1 = seedSize / preSum1;

	inputs[0]->deriveDimentions(ansDimentions);
}

void Min::getValue(){
	inputs[0]->getValue();

	double highVal = numeric_limits<double>::infinity();
	for(int i = 0; i < outSize; i++){
		derivativeMemo[i] = highVal;
	}

	for(int i = 0; i < preSum; i++){
		for(int x = 0; x < inputs[0]->outDimentions[dimention]; x++){
			for(int z = 0; z < postSum; z++){
				if(inputs[0]->derivativeMemo[i * postSum * inputs[0]->outDimentions[dimention] + postSum * x + z] < derivativeMemo[i * postSum + z]){
					derivativeMemo[i * postSum + z] = inputs[0]->derivativeMemo[i * postSum * inputs[0]->outDimentions[dimention] + postSum * x + z];
					idx[i * postSum + z] = x;
				}
			}
		}
	}
}

void Min::getValueDimentions(){
	inputs[0]->getValueDimentions();

	outDimentions.clear();
	for(int i = 0; i < inputs[0]->outRank; i++){
		if(i != dimention){
			outDimentions.push_back(inputs[0]->outDimentions[i]);
		}
	}
	outRank = inputs[0]->outRank - 1;
	outSize = inputs[0]->outSize / inputs[0]->outDimentions[dimention];

	double highVal = numeric_limits<double>::infinity();
	derivativeMemo.clear();
	derivativeMemo.resize(outSize, highVal);
	idx.clear();
	idx.resize(outSize, 0);

	preSum = 1;
	for(int i = 0; i < dimention; i++){
		preSum *= inputs[0]->outDimentions[i];
	}
	postSum = outSize / preSum;
}


