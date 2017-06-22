#include "Math.hpp"
#include "Node.hpp"

double Add::operation(double a, double b){
	return a + b;
}

void Add::derive(vector<double>& seed){
	if (typeid(*inputs[0]) != typeid(Constant)){
		if(seedRank - inputs[0]->outRank > 0){
			for(int i = 0; i < ansSize1; i++){
				double sum = 0.0;
				for(int x = 0; x < seedSize / ansSize1; x++){
					sum += seed[x * ansSize1 + i];
				}
				ans1[i] = sum;
			}

			inputs[0]->derive(ans1);
		}
		else{
			inputs[0]->derive(seed);
		}
	}

	if (typeid(*inputs[1]) != typeid(Constant)){
		if(seedRank - inputs[1]->outRank > 0){
			for(int i = 0; i < ansSize2; i++){
				double sum = 0.0;
				for(int x = 0; x < seedSize / ansSize2; x++){
					sum += seed[x * ansSize2 + i];
				}
				ans2[i] = sum;
			}

			inputs[1]->derive(ans2);
		}
		else{
			inputs[1]->derive(seed);
		}
	}
}

void Add::deriveDimentions(vector<int>& seedDimentionsVal){
	seedDimentions = seedDimentionsVal;
	seedRank = seedDimentionsVal.size();
	seedSize = getSize(seedDimentions);

	ansDimentions1 = helpReduceDimentions(seedDimentions, inputs[0]->outDimentions);
	ansRank1 = ansDimentions1.size();
	ansSize1 = getSize(ansDimentions1);
	ans1.clear();
	ans1.resize(ansSize1, 0.0);

	inputs[0]->deriveDimentions(ansDimentions1);


	ansDimentions2 = helpReduceDimentions(seedDimentions, inputs[1]->outDimentions);
	ansRank2 = ansDimentions2.size();
	ansSize2 = getSize(ansDimentions2);
	ans2.clear();
	ans2.resize(ansSize2, 0.0);

	inputs[1]->deriveDimentions(ansDimentions2);
}

double Subtract::operation(double a, double b){
	return a - b;
}

void Subtract::derive(vector<double>& seed){
	if (typeid(*inputs[0]) != typeid(Constant)){
		if(seedRank - inputs[0]->outRank > 0){
			for(int i = 0; i < ansSize1; i++){
				double sum = 0.0;
				for(int x = 0; x < seedSize / ansSize1; x++){
					sum += seed[x * ansSize1 + i];
				}
				ans1[i] = sum;
			}
			inputs[0]->derive(ans1);
		}
		else{
			inputs[0]->derive(seed);
		}
	}

	if (typeid(*inputs[1]) != typeid(Constant)){
		if(seedRank - inputs[1]->outRank > 0){
			for(int i = 0; i < ansSize2; i++){
				double sum = 0.0;
				for(int x = 0; x < seedSize / ansSize2; x++){
					sum += seed[x * ansSize2 + i];
				}
				ans2[i] = -sum;
			}
		}
		else{
			for(int i = 0; i < ansSize2; i++){
				ans2[i] = -seed[i];
			}
		}
		inputs[1]->derive(ans2);
	}
}

double Multiply::operation(double a, double b){
	return a * b;
}

void Multiply::derive(vector<double>& seed){
	if (typeid(*inputs[0]) != typeid(Constant)){
		for(int i = 0; i < tempSize1; i++){
			temp1[i] = seed[i % seedSize] * inputs[1]->derivativeMemo[i % inputs[1]->outSize];
		}

		if(tempRank1 - inputs[0]->outRank > 0){
			for(int i = 0; i < ansSize1; i++){
				double sum = 0.0;
				for(int x = 0; x < tempSize1 / ansSize1; x++){
					sum += temp1[x * ansSize1 + i];
				}
				ans1[i] = sum;
			}
			inputs[0]->derive(ans1);
		}
		else{
			inputs[0]->derive(temp1);
		}
	}

	if (typeid(*inputs[1]) != typeid(Constant)){
		for(int i = 0; i < tempSize2; i++){
			temp2[i] = seed[i % seedSize] * inputs[0]->derivativeMemo[i % inputs[0]->outSize];
		}

		if(tempRank2 - inputs[1]->outRank > 0){
			for(int i = 0; i < ansSize2; i++){
				double sum = 0.0;
				for(int x = 0; x < tempSize2 / ansSize2; x++){
					sum += temp2[x * ansSize2 + i];
				}
				ans2[i] = sum;
			}
			inputs[1]->derive(ans2);
		}
		else{
			inputs[1]->derive(temp2);
		}
	}
}

void Multiply::deriveDimentions(vector<int>& seedDimentionsVal){
	seedDimentions = seedDimentionsVal;
	seedRank = seedDimentionsVal.size();
	seedSize = getSize(seedDimentions);

	helpDeriveDimentions(vector<vector<int>>{seedDimentions, inputs[1]->outDimentions}, vector<vector<int>>{seedDimentions, inputs[0]->outDimentions});
	inputs[0]->deriveDimentions(ansDimentions1);
	inputs[1]->deriveDimentions(ansDimentions2);
}

double Divide::operation(double a, double b){
	return a / b;
}

void Divide::derive(vector<double>& seed){
	if (typeid(*inputs[0]) != typeid(Constant)){
		for(int i = 0; i < tempSize1; i++){
			temp1[i] = seed[i % seedSize] / inputs[1]->derivativeMemo[i % inputs[1]->outSize];
		}

		if(tempRank1 - inputs[0]->outRank > 0){
			for(int i = 0; i < ansSize1; i++){
				double sum = 0.0;
				for(int x = 0; x < tempSize1 / ansSize1; x++){
					sum += temp1[x * ansSize1 + i];
				}
				ans1[i] = sum;
			}
			inputs[0]->derive(ans1);
		}
		else{
			inputs[0]->derive(temp1);
		}
	}

	if (typeid(*inputs[1]) != typeid(Constant)){
		for(int i = 0; i < tempSize2; i++){
			double tempVal = inputs[1]->derivativeMemo[i % inputs[1]->outSize];
			temp2[i] = (-seed[i % seedSize] * inputs[0]->derivativeMemo[i % inputs[0]->outSize]) / (tempVal * tempVal);
		}

		if(tempRank2 - inputs[1]->outRank > 0){
			for(int i = 0; i < ansSize2; i++){
				double sum = 0.0;
				for(int x = 0; x < tempSize2 / ansSize2; x++){
					sum += temp2[x * ansSize2 + i];
				}
				ans2[i] = sum;
			}
			inputs[1]->derive(ans2);
		}
		else{
			inputs[1]->derive(temp2);
		}
	}
}

void Divide::deriveDimentions(vector<int>& seedDimentionsVal){
	seedDimentions = seedDimentionsVal;
	seedRank = seedDimentionsVal.size();
	seedSize = getSize(seedDimentions);

	helpDeriveDimentions(vector<vector<int>>{seedDimentions, inputs[1]->outDimentions}, vector<vector<int>>{seedDimentions, inputs[0]->outDimentions, inputs[1]->outDimentions});
	inputs[0]->deriveDimentions(ansDimentions1);
	inputs[1]->deriveDimentions(ansDimentions2);
}

double Pow::operation(double a, double b){
	return pow(a, b);
}


void Pow::derive(vector<double>& seed){
	if (typeid(*inputs[0]) != typeid(Constant)){
		for(int i = 0; i < tempSize1; i++){
			temp1[i] = seed[i % seedSize] * inputs[1]->derivativeMemo[i % inputs[1]->outSize] * pow(inputs[0]->derivativeMemo[i % inputs[0]->outSize], (inputs[1]->derivativeMemo[i % inputs[1]->outSize] - 1.0));
		}

		if(tempRank1 - inputs[0]->outRank > 0){
			for(int i = 0; i < ansSize1; i++){
				double sum = 0.0;
				for(int x = 0; x < tempSize1 / ansSize1; x++){
					sum += temp1[x * ansSize1 + i];
				}
				ans1[i] = sum;
			}
			inputs[0]->derive(ans1);
		}
		else{
			inputs[0]->derive(temp1);
		}
	}

	if (typeid(*inputs[1]) != typeid(Constant)){
		for(int i = 0; i < tempSize2; i++){
			temp2[i] = seed[i % seedSize] * derivativeMemo[i % outSize] * log(inputs[0]->derivativeMemo[i % inputs[0]->outSize]);
		}

		if(tempRank2 - inputs[1]->outRank > 0){
			for(int i = 0; i < ansSize2; i++){
				double sum = 0.0;
				for(int x = 0; x < tempSize2 / ansSize2; x++){
					sum += temp2[x * ansSize2 + i];
				}
				ans2[i] = sum;
			}
			inputs[1]->derive(ans2);
		}
		else{
			inputs[1]->derive(temp2);
		}
	}
}

void Pow::deriveDimentions(vector<int>& seedDimentionsVal){
	seedDimentions = seedDimentionsVal;
	seedRank = seedDimentionsVal.size();
	seedSize = getSize(seedDimentions);

	helpDeriveDimentions(vector<vector<int>>{seedDimentions, inputs[0]->outDimentions, inputs[1]->outDimentions}, vector<vector<int>>{seedDimentions, inputs[0]->outDimentions, outDimentions});
	inputs[0]->deriveDimentions(ansDimentions1);
	inputs[1]->deriveDimentions(ansDimentions2);
}

double Ln::operation(double a){
	return log(a);
}

void Ln::derive(vector<double>& seed){
	if (typeid(*inputs[0]) != typeid(Constant)){
		for(int i = 0; i < tempSize1; i++){
			temp1[i] = seed[i % seedSize] / inputs[0]->derivativeMemo[i % inputs[0]->outSize];
		}

		if(tempRank1 - inputs[0]->outRank > 0){
			for(int i = 0; i < ansSize1; i++){
				double sum = 0.0;
				for(int x = 0; x < tempSize1 / ansSize1; x++){
					sum += temp1[x * ansSize1 + i];
				}
				ans1[i] = sum;
			}
			inputs[0]->derive(ans1);
		}
		else{
			inputs[0]->derive(temp1);
		}
	}
}

void Ln::deriveDimentions(vector<int>& seedDimentionsVal){
	seedDimentions = seedDimentionsVal;
	seedRank = seedDimentionsVal.size();
	seedSize = getSize(seedDimentions);

	helpDeriveDimentions(vector<vector<int>>{seedDimentions, inputs[0]->outDimentions});
	inputs[0]->deriveDimentions(ansDimentions1);
}

double Exp::operation(double a){
	return exp(a);
}

void Exp::derive(vector<double>& seed){
	if (typeid(*inputs[0]) != typeid(Constant)){
		for(int i = 0; i < tempSize1; i++){
			temp1[i] = seed[i % seedSize] * derivativeMemo[i % outSize];
		}

		if(tempRank1 - inputs[0]->outRank > 0){
			for(int i = 0; i < ansSize1; i++){
				double sum = 0.0;
				for(int x = 0; x < tempSize1 / ansSize1; x++){
					sum += temp1[x * ansSize1 + i];
				}
				ans1[i] = sum;
			}
			inputs[0]->derive(ans1);
		}
		else{
			inputs[0]->derive(temp1);
		}
	}
}

void Exp::deriveDimentions(vector<int>& seedDimentionsVal){
	seedDimentions = seedDimentionsVal;
	seedRank = seedDimentionsVal.size();
	seedSize = getSize(seedDimentions);

	helpDeriveDimentions(vector<vector<int>>{seedDimentions, outDimentions});
	inputs[0]->deriveDimentions(ansDimentions1);
}

Log::Log(Node* a, double baseVal): BasicFunction(a){
	name = "Log";
	base = baseVal;
}

double Log::operation(double a){
	return log(a) / log(base);
}

void Log::derive(vector<double>& seed){
	if (typeid(*inputs[0]) != typeid(Constant)){
		for(int i = 0; i < tempSize1; i++){
			temp1[i] = seed[i % seedSize] / (inputs[0]->derivativeMemo[i % inputs[0]->outSize] * log(base));
		}

		if(tempRank1 - inputs[0]->outRank > 0){
			for(int i = 0; i < ansSize1; i++){
				double sum = 0.0;
				for(int x = 0; x < tempSize1 / ansSize1; x++){
					sum += temp1[x * ansSize1 + i];
				}
				ans1[i] = sum;
			}
			inputs[0]->derive(ans1);
		}
		else{
			inputs[0]->derive(temp1);
		}
	}
}

string Log::describe(){
	return name + "(" + inputs[0]->describe() + ", " + to_string(base) + ")";
}

void Log::deriveDimentions(vector<int>& seedDimentionsVal){
	seedDimentions = seedDimentionsVal;
	seedRank = seedDimentionsVal.size();
	seedSize = getSize(seedDimentions);

	helpDeriveDimentions(vector<vector<int>>{seedDimentions, inputs[0]->outDimentions});
	inputs[0]->deriveDimentions(ansDimentions1);
}

double Sin::operation(double a){
	return sin(a);
}

void Sin::derive(vector<double>& seed){
	if (typeid(*inputs[0]) != typeid(Constant)){
		for(int i = 0; i < tempSize1; i++){
			temp1[i] = seed[i % seedSize] * cos(inputs[0]->derivativeMemo[i % inputs[0]->outSize]);
		}

		if(tempRank1 - inputs[0]->outRank > 0){
			for(int i = 0; i < ansSize1; i++){
				double sum = 0.0;
				for(int x = 0; x < tempSize1 / ansSize1; x++){
					sum += temp1[x * ansSize1 + i];
				}
				ans1[i] = sum;
			}
			inputs[0]->derive(ans1);
		}
		else{
			inputs[0]->derive(temp1);
		}
	}
}

void Sin::deriveDimentions(vector<int>& seedDimentionsVal){
	seedDimentions = seedDimentionsVal;
	seedRank = seedDimentionsVal.size();
	seedSize = getSize(seedDimentions);

	helpDeriveDimentions(vector<vector<int>>{seedDimentions, inputs[0]->outDimentions});
	inputs[0]->deriveDimentions(ansDimentions1);
}

double Cos::operation(double a){
	return cos(a);
}

void Cos::derive(vector<double>& seed){
	if (typeid(*inputs[0]) != typeid(Constant)){
		for(int i = 0; i < tempSize1; i++){
			temp1[i] = -seed[i % seedSize] * sin(inputs[0]->derivativeMemo[i % inputs[0]->outSize]);
		}

		if(tempRank1 - inputs[0]->outRank > 0){
			for(int i = 0; i < ansSize1; i++){
				double sum = 0.0;
				for(int x = 0; x < tempSize1 / ansSize1; x++){
					sum += temp1[x * ansSize1 + i];
				}
				ans1[i] = sum;
			}
			inputs[0]->derive(ans1);
		}
		else{
			inputs[0]->derive(temp1);
		}
	}
}

double Tan::operation(double a){
	return tan(a);
}

void Tan::derive(vector<double>& seed){
	if (typeid(*inputs[0]) != typeid(Constant)){
		for(int i = 0; i < tempSize1; i++){
			double tempVal = (1.0 / cos(inputs[0]->derivativeMemo[i % inputs[0]->outSize]));
			temp1[i] = seed[i % seedSize] * tempVal * tempVal;
		}

		if(tempRank1 - inputs[0]->outRank > 0){
			for(int i = 0; i < ansSize1; i++){
				double sum = 0.0;
				for(int x = 0; x < tempSize1 / ansSize1; x++){
					sum += temp1[x * ansSize1 + i];
				}
				ans1[i] = sum;
			}
			inputs[0]->derive(ans1);
		}
		else{
			inputs[0]->derive(temp1);
		}
	}
}

double ArcSin::operation(double a){
	return asin(a);
}

void ArcSin::derive(vector<double>& seed){
	if (typeid(*inputs[0]) != typeid(Constant)){
		for(int i = 0; i < tempSize1; i++){
			double tempVal = inputs[0]->derivativeMemo[i % inputs[0]->outSize];
			temp1[i] = seed[i % seedSize] / pow(1.0 - tempVal * tempVal, 0.5);
		}

		if(tempRank1 - inputs[0]->outRank > 0){
			for(int i = 0; i < ansSize1; i++){
				double sum = 0.0;
				for(int x = 0; x < tempSize1 / ansSize1; x++){
					sum += temp1[x * ansSize1 + i];
				}
				ans1[i] = sum;
			}
			inputs[0]->derive(ans1);
		}
		else{
			inputs[0]->derive(temp1);
		}
	}
}

double ArcCos::operation(double a){
	return acos(a);
}

void ArcCos::derive(vector<double>& seed){
	if (typeid(*inputs[0]) != typeid(Constant)){
		for(int i = 0; i < tempSize1; i++){
			double tempVal = inputs[0]->derivativeMemo[i % inputs[0]->outSize];
			temp1[i] = -seed[i % seedSize] / pow(1.0 - tempVal * tempVal, 0.5);
		}

		if(tempRank1 - inputs[0]->outRank > 0){
			for(int i = 0; i < ansSize1; i++){
				double sum = 0.0;
				for(int x = 0; x < tempSize1 / ansSize1; x++){
					sum += temp1[x * ansSize1 + i];
				}
				ans1[i] = sum;
			}
			inputs[0]->derive(ans1);
		}
		else{
			inputs[0]->derive(temp1);
		}
	}
}

double ArcTan::operation(double a){
	return atan(a);
}

void ArcTan::derive(vector<double>& seed){
	if (typeid(*inputs[0]) != typeid(Constant)){
		for(int i = 0; i < tempSize1; i++){
			double tempVal = inputs[0]->derivativeMemo[i % inputs[0]->outSize];
			temp1[i] = seed[i % seedSize] / (1.0 + tempVal * tempVal);
		}

		if(tempRank1 - inputs[0]->outRank > 0){
			for(int i = 0; i < ansSize1; i++){
				double sum = 0.0;
				for(int x = 0; x < tempSize1 / ansSize1; x++){
					sum += temp1[x * ansSize1 + i];
				}
				ans1[i] = sum;
			}
			inputs[0]->derive(ans1);
		}
		else{
			inputs[0]->derive(temp1);
		}
	}
}

