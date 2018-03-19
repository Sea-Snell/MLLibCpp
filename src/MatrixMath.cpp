#include "MatrixMath.hpp"
#include "Node.hpp"
#include <limits>

MatMul::MatMul(Node* a, Node* b){
	inputs.push_back(a);
	inputs.push_back(b);
	a->outputs.push_back(this);
	b->outputs.push_back(this);
	a->outCount += 1;
	b->outCount += 1;
	name = "MatMul";
}

void MatMul::getDimentions(){
	if(getCount != 0){
		getCount = (getCount + 1) % outCount;
		return;
	}

	inputs[0]->getDimentions();
	inputs[1]->getDimentions();

	if (inputs[0]->resultDims.rank == 1 && inputs[1]->resultDims.rank == 1){
		resultDims.rank = 0;
		resultDims.size = 1;
		resultDims.dimentions = {};
	}
	else if (inputs[0]->resultDims.rank == 1){
		resultDims.rank = 1;
		resultDims.size = inputs[1]->resultDims.dimentions[1];
		resultDims.dimentions = {inputs[1]->resultDims.dimentions[1]};
	}
	else if (inputs[1]->resultDims.rank == 1){
		resultDims.rank = 1;
		resultDims.size = inputs[0]->resultDims.dimentions[0];
		resultDims.dimentions = {inputs[0]->resultDims.dimentions[0]};
	}
	else{
		resultDims.rank = 2;
		resultDims.size = inputs[0]->resultDims.dimentions[0] * inputs[1]->resultDims.dimentions[1];
		resultDims.dimentions = {inputs[0]->resultDims.dimentions[0], inputs[1]->resultDims.dimentions[1]};
	}

	resultDims.setBuf();

	result = cl::Buffer(context, CL_MEM_READ_WRITE, sizeof(float) * resultDims.size);
	getCount = (getCount + 1) % outCount;
}

void MatMul::getValue(){
	if(getCount != 0){
		getCount = (getCount + 1) % outCount;
		return;
	}
	inputs[0]->getValue();
	inputs[1]->getValue();
	if (inputs[0]->resultDims.rank == 1 && inputs[1]->resultDims.rank == 1){
		matMul1x1(cl::EnqueueArgs(queue, cl::NullRange, cl::NDRange(resultDims.size), cl::NullRange), inputs[0]->resultDims.dimBuf, inputs[0]->result, inputs[1]->resultDims.dimBuf, inputs[1]->result, result);
	}
	else if (inputs[0]->resultDims.rank == 1){
		matMul1x2(cl::EnqueueArgs(queue, cl::NullRange, cl::NDRange(resultDims.size), cl::NullRange), inputs[0]->resultDims.dimBuf, inputs[0]->result, inputs[1]->resultDims.dimBuf, inputs[1]->result, result);
	}
	else if (inputs[1]->resultDims.rank == 1){
		matMul2x1(cl::EnqueueArgs(queue, cl::NullRange, cl::NDRange(resultDims.size + (groupSize - resultDims.size % groupSize)), cl::NDRange(groupSize)), inputs[0]->resultDims.dimBuf, inputs[0]->result, inputs[1]->resultDims.dimBuf, inputs[1]->result, result);
	}
	else{
		matMul2x2(cl::EnqueueArgs(queue, cl::NullRange, cl::NDRange(resultDims.size), cl::NullRange), inputs[0]->resultDims.dimBuf, inputs[0]->result, inputs[1]->resultDims.dimBuf, inputs[1]->result, result);
	}
	getCount = (getCount + 1) % outCount;
}

void MatMul::deriveDimentions(GPUDimentions* tempSeed){
	getCount = (getCount + 1) % outCount;
	seedDimAdd(tempSeed);

	if (getCount == 0){
		outDims.push_back(inputs[0]->resultDims);
		outDims.push_back(inputs[1]->resultDims);
		out.push_back(cl::Buffer(context, CL_MEM_READ_WRITE, sizeof(float) * outDims[0].size));
		out.push_back(cl::Buffer(context, CL_MEM_READ_WRITE, sizeof(float) * outDims[1].size));
		inputs[0]->deriveDimentions(&outDims[0]);
		inputs[1]->deriveDimentions(&outDims[1]);
	}
}

void MatMul::derive(){
	getCount = (getCount + 1) % outCount;
	if (getCount == 0){
		if (typeid(*inputs[0]) != typeid(Constant)){
			if (inputs[0]->getCount == 0){
				zeroBuffer(cl::EnqueueArgs(queue, cl::NullRange, cl::NDRange(inputs[0]->seedDims.size), cl::NullRange), inputs[0]->seed);
			}
			if (inputs[0]->resultDims.rank == 1 && inputs[1]->resultDims.rank == 1){
				matMul1x1Derivative0(cl::EnqueueArgs(queue, cl::NullRange, cl::NDRange(outDims[0].size), cl::NullRange), inputs[1]->result, seed, out[0]);
			}
			else if (inputs[0]->resultDims.rank == 1){
				matMul1x2Derivative0(cl::EnqueueArgs(queue, cl::NullRange, cl::NDRange(outDims[0].size), cl::NullRange), inputs[1]->resultDims.dimBuf, inputs[1]->result, seedDims.dimBuf, seed, out[0]);
			}
			else if (inputs[1]->resultDims.rank == 1){
				matMul2x1Derivative0(cl::EnqueueArgs(queue, cl::NullRange, cl::NDRange(outDims[0].size), cl::NullRange), inputs[1]->resultDims.dimBuf, inputs[1]->result, seedDims.dimBuf, seed, out[0]);
			}
			else{
				matMul2x2Derivative0(cl::EnqueueArgs(queue, cl::NullRange, cl::NDRange(outDims[0].size), cl::NullRange), inputs[1]->resultDims.dimBuf, inputs[1]->result, seedDims.dimBuf, seed, out[0]);
			}
			explodeUp(cl::EnqueueArgs(queue, cl::NullRange, cl::NDRange(inputs[0]->seedDims.size), cl::NullRange), outDims[0].dimBuf, out[0], inputs[0]->seedDims.dimBuf, inputs[0]->seed);
			inputs[0]->derive();
		}
		if (typeid(*inputs[1]) != typeid(Constant)){
			if (inputs[1]->getCount == 0){
				zeroBuffer(cl::EnqueueArgs(queue, cl::NullRange, cl::NDRange(inputs[1]->seedDims.size), cl::NullRange), inputs[1]->seed);
			}
			if (inputs[0]->resultDims.rank == 1 && inputs[1]->resultDims.rank == 1){
				matMul1x1Derivative1(cl::EnqueueArgs(queue, cl::NullRange, cl::NDRange(outDims[1].size), cl::NullRange), inputs[0]->result, seed, out[1]);
			}
			else if (inputs[0]->resultDims.rank == 1){
				matMul1x2Derivative1(cl::EnqueueArgs(queue, cl::NullRange, cl::NDRange(outDims[1].size), cl::NullRange), inputs[0]->result, seedDims.dimBuf, seed, outDims[1].dimBuf, out[1]);
			}
			else if (inputs[1]->resultDims.rank == 1){
				matMul2x1Derivative1(cl::EnqueueArgs(queue, cl::NullRange, cl::NDRange(outDims[1].size + (groupSize - outDims[1].size % groupSize)), cl::NDRange(groupSize)), inputs[0]->resultDims.dimBuf, inputs[0]->result, seedDims.dimBuf, seed, out[1]);
			}
			else{
				matMul2x2Derivative1(cl::EnqueueArgs(queue, cl::NullRange, cl::NDRange(outDims[1].size), cl::NullRange), inputs[0]->resultDims.dimBuf, inputs[0]->result, seedDims.dimBuf, seed, outDims[1].dimBuf, out[1]);
			}
			explodeUp(cl::EnqueueArgs(queue, cl::NullRange, cl::NDRange(inputs[1]->seedDims.size), cl::NullRange), outDims[1].dimBuf, out[1], inputs[1]->seedDims.dimBuf, inputs[1]->seed);
			inputs[1]->derive();
		}
	}
}



Sum::Sum(Node* a, int dimentionVal){
	dimention = dimentionVal;
	inputs.push_back(a);
	a->outputs.push_back(this);
	a->outCount += 1;
	name = "Sum";
}

void Sum::getDimentions(){
	if(getCount != 0){
		getCount = (getCount + 1) % outCount;
		return;
	}

	inputs[0]->getDimentions();

	resultDims.rank = inputs[0]->resultDims.rank - 1;
	resultDims.size = inputs[0]->resultDims.size / inputs[0]->resultDims.dimentions[dimention];
	resultDims.dimentions = {};
	for (int i = 0; i < inputs[0]->resultDims.rank; i++){
		if (i != dimention){
			resultDims.dimentions.push_back(inputs[0]->resultDims.dimentions[i]);
		}
	}

	preSum = 1;
	for (int i = dimention + 1; i < inputs[0]->resultDims.rank; i++){
		preSum *= inputs[0]->resultDims.dimentions[i];
	}

	resultDims.setBuf();

	result = cl::Buffer(context, CL_MEM_READ_WRITE, sizeof(float) * resultDims.size);
	getCount = (getCount + 1) % outCount;
}

void Sum::getValue(){
	if(getCount != 0){
		getCount = (getCount + 1) % outCount;
		return;
	}
	inputs[0]->getValue();
	sum_(cl::EnqueueArgs(queue, cl::NullRange, cl::NDRange(resultDims.size), cl::NullRange), inputs[0]->result, result, inputs[0]->resultDims.dimentions[dimention], preSum);
	getCount = (getCount + 1) % outCount;
}

void Sum::deriveDimentions(GPUDimentions* tempSeed){
	getCount = (getCount + 1) % outCount;
	seedDimAdd(tempSeed);

	if (getCount == 0){
		if (seedDims.rank < inputs[0]->resultDims.rank - dimention){
			outDims.push_back(seedDims);
		}
		else{
			GPUDimentions tempDims = GPUDimentions();
			tempDims.rank = seedDims.rank + 1;
			tempDims.size = seedDims.size * inputs[0]->resultDims.dimentions[dimention];
			tempDims.dimentions = {};
			for (int i = 0; i < tempDims.rank; i++){
				tempDims.dimentions.push_back(inputs[0]->resultDims.dimentions[inputs[0]->resultDims.rank - tempDims.rank + i]);
			}
			tempDims.setBuf();
			outDims.push_back(tempDims);
		}
		out.push_back(cl::Buffer(context, CL_MEM_READ_WRITE, sizeof(float) * outDims[0].size));
		inputs[0]->deriveDimentions(&outDims[0]);
	}
}

void Sum::derive(){
	getCount = (getCount + 1) % outCount;
	if (getCount == 0){
		if (typeid(*inputs[0]) != typeid(Constant)){
			if (inputs[0]->getCount == 0){
				zeroBuffer(cl::EnqueueArgs(queue, cl::NullRange, cl::NDRange(inputs[0]->seedDims.size), cl::NullRange), inputs[0]->seed);
			}
			if (seedDims.rank < inputs[0]->resultDims.rank - dimention){
				explodeUp(cl::EnqueueArgs(queue, cl::NullRange, cl::NDRange(inputs[0]->seedDims.size), cl::NullRange), seedDims.dimBuf, seed, inputs[0]->seedDims.dimBuf, inputs[0]->seed);
			}
			else{
				sumDerivative(cl::EnqueueArgs(queue, cl::NullRange, cl::NDRange(outDims[0].size), cl::NullRange), seedDims.dimBuf, seed, out[0], inputs[0]->resultDims.dimentions[dimention], preSum);
				explodeUp(cl::EnqueueArgs(queue, cl::NullRange, cl::NDRange(inputs[0]->seedDims.size), cl::NullRange), outDims[0].dimBuf, out[0], inputs[0]->seedDims.dimBuf, inputs[0]->seed);
			}
			inputs[0]->derive();
		}
	}
}

string Sum::describe(){
	return name + "(" + inputs[0]->describe() + ", " + to_string(dimention) + ")";
}




void Mean::getValue(){
	if(getCount != 0){
		getCount = (getCount + 1) % outCount;
		return;
	}
	inputs[0]->getValue();
	mean_(cl::EnqueueArgs(queue, cl::NullRange, cl::NDRange(resultDims.size), cl::NullRange), inputs[0]->result, result, inputs[0]->resultDims.dimentions[dimention], preSum);
	getCount = (getCount + 1) % outCount;
}

void Mean::derive(){
	getCount = (getCount + 1) % outCount;
	if (getCount == 0){
		if (typeid(*inputs[0]) != typeid(Constant)){
			if (inputs[0]->getCount == 0){
				zeroBuffer(cl::EnqueueArgs(queue, cl::NullRange, cl::NDRange(inputs[0]->seedDims.size), cl::NullRange), inputs[0]->seed);
			}
			if (seedDims.rank < inputs[0]->resultDims.rank - dimention){
				meanDerivativeSmallSeed(cl::EnqueueArgs(queue, cl::NullRange, cl::NDRange(outDims[0].size), cl::NullRange), seed, out[0], inputs[0]->resultDims.dimentions[dimention]);
				explodeUp(cl::EnqueueArgs(queue, cl::NullRange, cl::NDRange(inputs[0]->seedDims.size), cl::NullRange), outDims[0].dimBuf, out[0], inputs[0]->seedDims.dimBuf, inputs[0]->seed);
			}
			else{
				meanDerivative(cl::EnqueueArgs(queue, cl::NullRange, cl::NDRange(outDims[0].size), cl::NullRange), seedDims.dimBuf, seed, out[0], inputs[0]->resultDims.dimentions[dimention], preSum);
				explodeUp(cl::EnqueueArgs(queue, cl::NullRange, cl::NDRange(inputs[0]->seedDims.size), cl::NullRange), outDims[0].dimBuf, out[0], inputs[0]->seedDims.dimBuf, inputs[0]->seed);
			}
			inputs[0]->derive();
		}
	}
}




Trans::Trans(Node* a){
	inputs.push_back(a);
	a->outputs.push_back(this);
	a->outCount += 1;
	name = "Trans";
}

void Trans::getDimentions(){
	if(getCount != 0){
		getCount = (getCount + 1) % outCount;
		return;
	}

	inputs[0]->getDimentions();

	resultDims.rank = inputs[0]->resultDims.rank;
	resultDims.size = inputs[0]->resultDims.size;
	resultDims.dimentions = {};
	resultDims.dimentions.push_back(inputs[0]->resultDims.dimentions[1]);
	resultDims.dimentions.push_back(inputs[0]->resultDims.dimentions[0]);

	resultDims.setBuf();

	result = cl::Buffer(context, CL_MEM_READ_WRITE, sizeof(float) * resultDims.size);
	getCount = (getCount + 1) % outCount;
}

void Trans::getValue(){
	if(getCount != 0){
		getCount = (getCount + 1) % outCount;
		return;
	}
	inputs[0]->getValue();
	trans(cl::EnqueueArgs(queue, cl::NullRange, cl::NDRange(resultDims.size), cl::NullRange), inputs[0]->resultDims.dimBuf, inputs[0]->result, result);
	getCount = (getCount + 1) % outCount;
}

void Trans::deriveDimentions(GPUDimentions* tempSeed){
	getCount = (getCount + 1) % outCount;
	seedDimAdd(tempSeed);

	if (getCount == 0){
		if (seedDims.rank == 0){
			outDims.push_back(seedDims);
		}
		else{
			outDims.push_back(inputs[0]->resultDims);
		}
		out.push_back(cl::Buffer(context, CL_MEM_READ_WRITE, sizeof(float) * outDims[0].size));
		inputs[0]->deriveDimentions(&outDims[0]);
	}
}

void Trans::derive(){
	getCount = (getCount + 1) % outCount;
	if (getCount == 0){
		if (typeid(*inputs[0]) != typeid(Constant)){
			if (inputs[0]->getCount == 0){
				zeroBuffer(cl::EnqueueArgs(queue, cl::NullRange, cl::NDRange(inputs[0]->seedDims.size), cl::NullRange), inputs[0]->seed);
			}
			if (seedDims.rank == 0){
				explodeUp(cl::EnqueueArgs(queue, cl::NullRange, cl::NDRange(inputs[0]->seedDims.size), cl::NullRange), seedDims.dimBuf, seed, inputs[0]->seedDims.dimBuf, inputs[0]->seed);
			}
			else if (seedDims.rank == 1){
				transDerivative1(cl::EnqueueArgs(queue, cl::NullRange, cl::NDRange(outDims[0].size), cl::NullRange), seed, outDims[0].dimBuf, out[0]);
				explodeUp(cl::EnqueueArgs(queue, cl::NullRange, cl::NDRange(inputs[0]->seedDims.size), cl::NullRange), outDims[0].dimBuf, out[0], inputs[0]->seedDims.dimBuf, inputs[0]->seed);
			}
			else{
				transDerivative2(cl::EnqueueArgs(queue, cl::NullRange, cl::NDRange(outDims[0].size), cl::NullRange), seedDims.dimBuf, seed, out[0]);
				explodeUp(cl::EnqueueArgs(queue, cl::NullRange, cl::NDRange(inputs[0]->seedDims.size), cl::NullRange), outDims[0].dimBuf, out[0], inputs[0]->seedDims.dimBuf, inputs[0]->seed);
			}
			inputs[0]->derive();
		}
	}
}


Max::Max(Node* a, int dimentionVal){
	inputs.push_back(a);
	a->outputs.push_back(this);
	a->outCount += 1;
	name = "Max";
	dimention = dimentionVal;
}

void Max::getDimentions(){
	if(getCount != 0){
		getCount = (getCount + 1) % outCount;
		return;
	}

	inputs[0]->getDimentions();

	resultDims.rank = inputs[0]->resultDims.rank - 1;
	resultDims.size = inputs[0]->resultDims.size / inputs[0]->resultDims.dimentions[dimention];
	resultDims.dimentions = {};
	for (int i = 0; i < inputs[0]->resultDims.rank; i++){
		if (i != dimention){
			resultDims.dimentions.push_back(inputs[0]->resultDims.dimentions[i]);
		}
	}

	preSum = 1;
	for (int i = dimention + 1; i < inputs[0]->resultDims.rank; i++){
		preSum *= inputs[0]->resultDims.dimentions[i];
	}

	resultDims.setBuf();

	result = cl::Buffer(context, CL_MEM_READ_WRITE, sizeof(float) * resultDims.size);
	idx = cl::Buffer(context, CL_MEM_READ_WRITE, sizeof(int) * inputs[0]->resultDims.size);
	getCount = (getCount + 1) % outCount;
}

void Max::getValue(){
	if(getCount != 0){
		getCount = (getCount + 1) % outCount;
		return;
	}
	inputs[0]->getValue();
	max_(cl::EnqueueArgs(queue, cl::NullRange, cl::NDRange(resultDims.size), cl::NullRange), inputs[0]->result, result, idx, inputs[0]->resultDims.dimentions[dimention], preSum);
	getCount = (getCount + 1) % outCount;
}

void Max::deriveDimentions(GPUDimentions* tempSeed){
	getCount = (getCount + 1) % outCount;
	seedDimAdd(tempSeed);

	if (getCount == 0){
		outDims.push_back(inputs[0]->resultDims);
		out.push_back(cl::Buffer(context, CL_MEM_READ_WRITE, sizeof(float) * outDims[0].size));
		inputs[0]->deriveDimentions(&outDims[0]);
	}
}

void Max::derive(){
	getCount = (getCount + 1) % outCount;
	if (getCount == 0){
		if (typeid(*inputs[0]) != typeid(Constant)){
			if (inputs[0]->getCount == 0){
				zeroBuffer(cl::EnqueueArgs(queue, cl::NullRange, cl::NDRange(inputs[0]->seedDims.size), cl::NullRange), inputs[0]->seed);
			}
			if (seedDims.rank < inputs[0]->resultDims.rank - dimention){
				maxDerivativeSmallSeed(cl::EnqueueArgs(queue, cl::NullRange, cl::NDRange(outDims[0].size), cl::NullRange), seedDims.dimBuf, seed, out[0], idx);
				explodeUp(cl::EnqueueArgs(queue, cl::NullRange, cl::NDRange(inputs[0]->seedDims.size), cl::NullRange), outDims[0].dimBuf, out[0], inputs[0]->seedDims.dimBuf, inputs[0]->seed);
			}
			else{
				maxDerivative(cl::EnqueueArgs(queue, cl::NullRange, cl::NDRange(outDims[0].size), cl::NullRange), seedDims.dimBuf, seed, out[0], idx, inputs[0]->resultDims.dimentions[dimention], preSum);
				explodeUp(cl::EnqueueArgs(queue, cl::NullRange, cl::NDRange(inputs[0]->seedDims.size), cl::NullRange), outDims[0].dimBuf, out[0], inputs[0]->seedDims.dimBuf, inputs[0]->seed);
			}
			inputs[0]->derive();
		}
	}
}

string Max::describe(){
	return name + "(" + inputs[0]->describe() + ", " + to_string(dimention) + ")";
}

void Min::getValue(){
	if(getCount != 0){
		getCount = (getCount + 1) % outCount;
		return;
	}
	inputs[0]->getValue();
	min_(cl::EnqueueArgs(queue, cl::NullRange, cl::NDRange(resultDims.size), cl::NullRange), inputs[0]->result, result, idx, inputs[0]->resultDims.dimentions[dimention], preSum);
	getCount = (getCount + 1) % outCount;
}


