#include "MatrixMath.hpp"
#include "Node.hpp"
#include "MapVals.hpp"
#include <limits>

MatMul::MatMul(Node* a, Node* b){
	outCount = 0;
	inputs.push_back(a);
	inputs.push_back(b);
	a->outCount += 1;
	b->outCount += 1;
	name = "MatMul";
	dCallCount = 0;
	gCallCount = 0;
}

NumObject MatMul::getValue(int t, int tf){
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

	NumObject a = inputs[0]->getValue(t, tf);
	NumObject b = inputs[1]->getValue(t, tf);

	if (a.rank == 1 && b.rank == 1){
		NumObject total = NumObject(0.0);
		for(int i = 0; i < a.dimentions[0]; i++){
			total.values[0] += a.values[i] * b.values[i];
		}
		return memoize(total, t, tf);
	}

	if (a.rank == 1 && b.rank == 2){
		NumObject answer = NumObject(vector<int>{b.dimentions[1]}, 0.0);
		for(int i = 0; i < b.dimentions[1]; i++){
			for(int x = 0; x < a.dimentions[0]; x++){
				answer.values[i] += a.values[x] * b.values[x * b.dimentions[1] + i];
			}
		}
		return memoize(answer, t, tf);
	}

	if (a.rank == 2 && b.rank == 1){
		NumObject answer = NumObject(vector<int>{a.dimentions[0]}, 0.0);
		for(int i = 0; i < a.dimentions[0]; i++){
			for(int x = 0; x < b.dimentions[0]; x++){
				answer.values[i] += a.values[b.dimentions[0] * i + x] * b.values[x];
			}
		}
		return memoize(answer, t, tf);
	}

	NumObject answer = NumObject(vector<int>{a.dimentions[0], b.dimentions[1]}, 0.0);
	for(int i = 0; i < b.dimentions[1]; i++){
		for(int x = 0; x < a.dimentions[0]; x++){
			for(int z = 0; z < b.dimentions[0]; z++){
				answer.values[x * b.dimentions[1] + i] += a.values[x * a.dimentions[1] + z] * b.values[z * b.dimentions[1] + i];
			}
		}
	}

	return memoize(answer, t, tf);
}

void MatMul::derive(NumObject& seed, int t, int tf){
	if(sumSeed(seed)){
		NumObject a = inputs[0]->derivativeMemo[t];
		NumObject b = inputs[1]->derivativeMemo[t];

		if(a.rank == 2 && b.rank == 2){
			if (typeid(*inputs[0]) != typeid(Constant)){
				NumObject answer = NumObject(a.dimentions, 0.0);
				for(int i = 0; i < a.dimentions[0]; i++){
					for(int x = 0; x < a.dimentions[1]; x++){
						for(int z = 0; z < b.dimentions[1]; z++){
							answer.values[i * a.dimentions[1] + x] += b.values[x * b.dimentions[1] + z] * tempSeed.values[(i * b.dimentions[1] + z) % tempSeed.values.size()];
						}
					}
				}
				inputs[0]->derive(answer, t, tf);
			}

			if (typeid(*inputs[1]) != typeid(Constant)){
				NumObject answer = NumObject(b.dimentions, 0.0);
				for(int i = 0; i < b.dimentions[0]; i++){
					for(int x = 0; x < b.dimentions[1]; x++){
						for(int z = 0; z < a.dimentions[0]; z++){
							answer.values[i * b.dimentions[1] + x] += a.values[z * a.dimentions[1] + i] * tempSeed.values[(z * b.dimentions[1] + x) % tempSeed.values.size()];
						}
					}
				}
				inputs[1]->derive(answer, t, tf);
			}
		}

		else if(a.rank == 2 && b.rank == 1){
			if (typeid(*inputs[0]) != typeid(Constant)){
				NumObject answer = NumObject(a.dimentions, 0.0);
				for(int i = 0; i < a.dimentions[0]; i++){
					for(int x = 0; x < a.dimentions[1]; x++){
						answer.values[i * a.dimentions[1] + x] = b.values[x] * tempSeed.values[i % tempSeed.values.size()];
					}
				}
				inputs[0]->derive(answer, t, tf);
			}
			if (typeid(*inputs[1]) != typeid(Constant)){
				NumObject answer = NumObject(b.dimentions, 0.0);
				for(int i = 0; i < b.dimentions[0]; i++){
					for(int x = 0; x < a.dimentions[0]; x++){
						answer.values[i] += a.values[x * a.dimentions[1] + i] * tempSeed.values[x % tempSeed.values.size()];
					}
				}
				inputs[1]->derive(answer, t, tf);
			}
		}

		else if(a.rank == 1 && b.rank == 2){
			if (typeid(*inputs[0]) != typeid(Constant)){
				NumObject answer = NumObject(a.dimentions, 0.0);
				for(int i = 0; i < a.dimentions[0]; i++){
					for(int x = 0; x < b.dimentions[1]; x++){
						answer.values[i] += b.values[i * b.dimentions[1] + x] * tempSeed.values[x % tempSeed.values.size()];
					}
				}
				inputs[0]->derive(answer, t, tf);
			}
			if (typeid(*inputs[1]) != typeid(Constant)){
				NumObject answer = NumObject(b.dimentions, 0.0);
				for(int i = 0; i < b.dimentions[0]; i++){
					for(int x = 0; x < b.dimentions[1]; x++){
						answer.values[i * b.dimentions[1] + x] = a.values[i] * tempSeed.values[x % tempSeed.values.size()];
					}
				}
				inputs[1]->derive(answer, t, tf);
			}
		}

		else{
			if (typeid(*inputs[0]) != typeid(Constant)){
				NumObject answer = NumObject(a.dimentions, 0.0);
				for(int i = 0; i < a.dimentions[0]; i++){
					answer.values[i] = b.values[i] * tempSeed.values[0];
				}
				inputs[0]->derive(answer, t, tf);
			}
			if (typeid(*inputs[1]) != typeid(Constant)){
				NumObject answer = NumObject(b.dimentions, 0.0);
				for(int i = 0; i < b.dimentions[0]; i++){
					answer.values[i] = a.values[i] * tempSeed.values[0];
				}
				inputs[1]->derive(answer, t, tf);
			}
		}
	}
}



Sum::Sum(Node* a, int dimentionVal){
	outCount = 0;
	inputs.push_back(a);
	a->outCount += 1;
	dimention = dimentionVal;
	name = "Sum";
	dCallCount = 0;
	gCallCount = 0;
}

NumObject Sum::getValue(int t, int tf){
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

	NumObject a = inputs[0]->getValue(t, tf);

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

	return memoize(answer, t, tf);
}


void Sum::derive(NumObject& seed, int t, int tf){
	if(sumSeed(seed)){
		NumObject a = inputs[0]->derivativeMemo[t];
		if(tempSeed.rank < (a.rank - dimention)){
			inputs[0]->derive(tempSeed, t, tf);
		}
		else{
			NumObject temp1 = expandDerivative(tempSeed, dimention - (a.rank - tempSeed.rank) + 1, a.dimentions[dimention]);
			inputs[0]->derive(temp1, t, tf);
		}
	}
}

string Sum::describe(){
	return name + "(" + inputs[0]->describe() + ", " + to_string(dimention) + ")";
}

Mean::Mean(Node* a, int dimentionVal){
	outCount = 0;
	inputs.push_back(a);
	a->outCount += 1;
	dimention = dimentionVal;
	name = "Mean";
	dCallCount = 0;
	gCallCount = 0;
}

NumObject Mean::getValue(int t, int tf){
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

	NumObject a = inputs[0]->getValue(t, tf);

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
	return memoize(ans2, t, tf);
}

double Mean::operation(vector<double>& a){
	return a[0] / a[1];
}

void Mean::derive(NumObject& seed, int t, int tf){
	if(sumSeed(seed)){
		NumObject a = inputs[0]->derivativeMemo[t];

		vector<NumObject> items = {tempSeed, NumObject(a.dimentions[dimention])};
		NumObject eval1 = mapVals(this, &Mean::operation, items);

		if(eval1.rank < (a.rank - dimention)){
			inputs[0]->derive(eval1, t, tf);
		}
		else{
			NumObject temp1 = expandDerivative(eval1, dimention - (a.rank - eval1.rank) + 1, a.dimentions[dimention]);
			inputs[0]->derive(temp1, t, tf);
		}
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
	outCount = 0;
	inputs.push_back(a);
	a->outCount += 1;
	perm = permutations;
	name = "Trans";
	dCallCount = 0;
	gCallCount = 0;
}

NumObject Trans::getValue(int t, int tf){
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

	NumObject a = inputs[0]->getValue(t, tf);

	vector<int> newDimentions;
	newDimentions.resize(a.rank, 0);
	for(int i = 0; i < a.rank; i++){
		newDimentions[i] = a.dimentions[perm[i]];
	}

	NumObject ans = NumObject(newDimentions, 0.0);
	for(int i = 0; i < a.values.size(); i++){
		ans.values[flipIdx(i, a, ans)] = a.values[i];
	}
	return memoize(ans, t, tf);
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

void Trans::derive(NumObject& seed, int t, int tf){
	if(sumSeed(seed)){
		NumObject a = inputs[0]->derivativeMemo[t];

		bool isInternalized = true;

		for(int i = 0; i < tempSeed.rank; i++){
			bool found = false;
			for(int x = 0; x < tempSeed.rank; x++){
				if(i == perm[a.rank - tempSeed.rank + x]){
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
			for(int i = 0; i < tempSeed.rank; i++){
				newDimentions.push_back(a.dimentions[a.rank - tempSeed.rank + i]);
			}

			NumObject ans = NumObject(newDimentions, 0.0);
			for(int i = 0; i < tempSeed.values.size(); i++){
				ans.values[flipIdxDerive(i, tempSeed, ans)] = tempSeed.values[i];
			}
			inputs[0]->derive(ans, t, tf);
		}
		else{
			NumObject ans = NumObject(a.dimentions, 0.0);
			for(int i = 0; i < derivativeMemo[t].values.size(); i++){
				ans.values[flipIdxDerive(i, derivativeMemo[t], ans)] = tempSeed.values[i % tempSeed.values.size()];
			}
			inputs[0]->derive(ans, t, tf);
		}
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
	outCount = 0;
	inputs.push_back(a);
	a->outCount += 1;
	dimention = dimentionVal;
	name = "Max";
	dCallCount = 0;
	gCallCount = 0;
}

NumObject Max::getValue(int t, int tf){
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

	NumObject a = inputs[0]->getValue(t, tf);

	vector<int> newDimentions;
	for(int i = 0; i < a.rank; i++){
		if (i != dimention){
			newDimentions.push_back(a.dimentions[i]);
		}
	}
	double lowVal = -numeric_limits<double>::infinity();
	NumObject answer = NumObject(newDimentions, lowVal);
	if(t == 0){
		idx.clear();
		idx.resize(tf + 1);
	}
	idx[t] = NumObject(newDimentions, 0.0);

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
					idx[t].values[i * postSum + z] = x;
				}
			}
		}
	}
	return memoize(answer, t, tf);
}

void Max::derive(NumObject& seed, int t, int tf){
	if(sumSeed(seed)){
		NumObject a = inputs[0]->derivativeMemo[t];
		NumObject ans = NumObject(a.dimentions);

		int preSum = 1;
		for(int i = 0; i < dimention; i++){
			preSum *= a.dimentions[i];
		}
		int postSum = a.values.size() / (preSum * a.dimentions[dimention]);


		if(tempSeed.rank < (a.rank - dimention)){
			for(int i = 0; i < preSum; i++){
				for(int x = 0; x < a.dimentions[dimention]; x++){
					for(int z = 0; z < postSum; z++){
						if(idx[t].values[i * postSum + z] == x){
							ans.values.push_back(tempSeed.values[(i * postSum + z) % tempSeed.values.size()]);
						}
						else{
							ans.values.push_back(0.0);
						}
					}
				}
			}
		}
		else{
			NumObject temp1 = expandDerivative(tempSeed, dimention - (a.rank - tempSeed.rank) + 1, a.dimentions[dimention]);
			for(int i = 0; i < preSum; i++){
				for(int x = 0; x < a.dimentions[dimention]; x++){
					for(int z = 0; z < postSum; z++){
						if(idx[t].values[i * postSum + z] == x){
							ans.values.push_back(temp1.values[(i * postSum * a.dimentions[dimention] + postSum * x + z) % temp1.values.size()]);
						}
						else{
							ans.values.push_back(0.0);
						}
					}
				}
			}
		}

		inputs[0]->derive(ans, t, tf);
	}
}

string Max::describe(){
	return name + "(" + inputs[0]->describe() + ", " + to_string(dimention) + ")";
}

NumObject Min::getValue(int t, int tf){
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

	NumObject a = inputs[0]->getValue(t, tf);

	vector<int> newDimentions;
	for(int i = 0; i < a.rank; i++){
		if (i != dimention){
			newDimentions.push_back(a.dimentions[i]);
		}
	}
	double lowVal = numeric_limits<double>::infinity();
	NumObject answer = NumObject(newDimentions, lowVal);
	if(t == 0){
		idx.clear();
		idx.resize(tf + 1);
	}
	idx[t] = NumObject(newDimentions, 0.0);

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
					idx[t].values[i * postSum + z] = x;
				}
			}
		}
	}
	return memoize(answer, t, tf);
}


