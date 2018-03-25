#include "Activations.hpp"


void Sigmoid::getValue(){
	if(getCount != 0){
		getCount = (getCount + 1) % outCount;
		return;
	}
	inputs[0]->getValue();
	sigmoid(cl::EnqueueArgs(queue, cl::NullRange, cl::NDRange(resultDims.size), cl::NullRange), inputs[0]->result, result);
	getCount = (getCount + 1) % outCount;
}

void Sigmoid::derive(){
	getCount = (getCount + 1) % outCount;
	if (getCount == 0){
		if (typeid(*inputs[0]) != typeid(Constant)){
			if (inputs[0]->getCount == 0){
				zeroBuffer(cl::EnqueueArgs(queue, cl::NullRange, cl::NDRange(inputs[0]->seedDims.size), cl::NullRange), inputs[0]->seed);
			}
			sigmoidDerivative(cl::EnqueueArgs(queue, cl::NullRange, cl::NDRange(outDims[0].size), cl::NullRange), result, seedDims.dimBuf, seed, out[0]);
			explodeUp(cl::EnqueueArgs(queue, cl::NullRange, cl::NDRange(inputs[0]->seedDims.size), cl::NullRange), outDims[0].dimBuf, out[0], inputs[0]->seedDims.dimBuf, inputs[0]->seed);
			inputs[0]->derive();
		}
	}
}

void ReLU::getValue(){
	if(getCount != 0){
		getCount = (getCount + 1) % outCount;
		return;
	}
	inputs[0]->getValue();
	reLU(cl::EnqueueArgs(queue, cl::NullRange, cl::NDRange(resultDims.size), cl::NullRange), inputs[0]->result, result);
	getCount = (getCount + 1) % outCount;
}

void ReLU::derive(){
	getCount = (getCount + 1) % outCount;
	if (getCount == 0){
		if (typeid(*inputs[0]) != typeid(Constant)){
			if (inputs[0]->getCount == 0){
				zeroBuffer(cl::EnqueueArgs(queue, cl::NullRange, cl::NDRange(inputs[0]->seedDims.size), cl::NullRange), inputs[0]->seed);
			}
			reLUDerivative(cl::EnqueueArgs(queue, cl::NullRange, cl::NDRange(outDims[0].size), cl::NullRange), inputs[0]->result, seedDims.dimBuf, seed, out[0]);
			explodeUp(cl::EnqueueArgs(queue, cl::NullRange, cl::NDRange(inputs[0]->seedDims.size), cl::NullRange), outDims[0].dimBuf, out[0], inputs[0]->seedDims.dimBuf, inputs[0]->seed);
			inputs[0]->derive();
		}
	}
}

void LeakyReLU::getValue(){
	if(getCount != 0){
		getCount = (getCount + 1) % outCount;
		return;
	}
	inputs[0]->getValue();
	leakyReLU(cl::EnqueueArgs(queue, cl::NullRange, cl::NDRange(resultDims.size), cl::NullRange), inputs[0]->result, result);
	getCount = (getCount + 1) % outCount;
}

void LeakyReLU::derive(){
	getCount = (getCount + 1) % outCount;
	if (getCount == 0){
		if (typeid(*inputs[0]) != typeid(Constant)){
			if (inputs[0]->getCount == 0){
				zeroBuffer(cl::EnqueueArgs(queue, cl::NullRange, cl::NDRange(inputs[0]->seedDims.size), cl::NullRange), inputs[0]->seed);
			}
			leakyReLUDerivative(cl::EnqueueArgs(queue, cl::NullRange, cl::NDRange(outDims[0].size), cl::NullRange), inputs[0]->result, seedDims.dimBuf, seed, out[0]);
			explodeUp(cl::EnqueueArgs(queue, cl::NullRange, cl::NDRange(inputs[0]->seedDims.size), cl::NullRange), outDims[0].dimBuf, out[0], inputs[0]->seedDims.dimBuf, inputs[0]->seed);
			inputs[0]->derive();
		}
	}
}

void Gaussian::getValue(){
	if(getCount != 0){
		getCount = (getCount + 1) % outCount;
		return;
	}
	inputs[0]->getValue();
	gaussian(cl::EnqueueArgs(queue, cl::NullRange, cl::NDRange(resultDims.size), cl::NullRange), inputs[0]->result, result);
	getCount = (getCount + 1) % outCount;
}

void Gaussian::derive(){
	getCount = (getCount + 1) % outCount;
	if (getCount == 0){
		if (typeid(*inputs[0]) != typeid(Constant)){
			if (inputs[0]->getCount == 0){
				zeroBuffer(cl::EnqueueArgs(queue, cl::NullRange, cl::NDRange(inputs[0]->seedDims.size), cl::NullRange), inputs[0]->seed);
			}
			gaussianDerivative(cl::EnqueueArgs(queue, cl::NullRange, cl::NDRange(outDims[0].size), cl::NullRange), inputs[0]->result, result, seedDims.dimBuf, seed, out[0]);
			explodeUp(cl::EnqueueArgs(queue, cl::NullRange, cl::NDRange(inputs[0]->seedDims.size), cl::NullRange), outDims[0].dimBuf, out[0], inputs[0]->seedDims.dimBuf, inputs[0]->seed);
			inputs[0]->derive();
		}
	}
}

Softmax::Softmax(Node* a, int dimentionVal){
	inputs.push_back(a);
	a->outputs.push_back(this);
	a->outCount += 1;
	name = "Softmax";
	dimention = dimentionVal;
	GROUP_SIZE = 128;
}

void Softmax::getDimentions(){
	if(getCount != 0){
		getCount = (getCount + 1) % outCount;
		return;
	}

	inputs[0]->getDimentions();

	resultDims.rank = inputs[0]->resultDims.rank;
	resultDims.size = inputs[0]->resultDims.size;
	resultDims.dimentions = inputs[0]->resultDims.dimentions;

	resultDims.setBuf();

	if (dimention == -1){
		dimention = resultDims.rank - 1;
	}

	preSum = 1;
	for (int i = dimention + 1; i < resultDims.rank; i++){
		preSum *= resultDims.dimentions[i];
	}

	int dimSize = resultDims.dimentions[dimention] + (GROUP_SIZE - resultDims.dimentions[dimention] % GROUP_SIZE);
	if (resultDims.dimentions[dimention] % GROUP_SIZE == 0){
		dimSize -= GROUP_SIZE;
	}
	blocksWide = dimSize / GROUP_SIZE;
	globalSize = (resultDims.size / resultDims.dimentions[dimention]) * dimSize;

	result = cl::Buffer(context, CL_MEM_READ_WRITE, sizeof(float) * resultDims.size);
	resedue = cl::Buffer(context, CL_MEM_READ_WRITE, sizeof(float) * ((resultDims.size / resultDims.dimentions[dimention]) * blocksWide));
	getCount = (getCount + 1) % outCount;
}

void Softmax::getValue(){
	if(getCount != 0){
		getCount = (getCount + 1) % outCount;
		return;
	}
	inputs[0]->getValue();
	softmax(cl::EnqueueArgs(queue, cl::NullRange, cl::NDRange(globalSize), cl::NDRange(GROUP_SIZE)), inputs[0]->result, resedue, result, resultDims.dimentions[dimention], preSum, blocksWide);
	getCount = (getCount + 1) % outCount;
}

void Softmax::deriveDimentions(GPUDimentions* tempSeed){
	getCount = (getCount + 1) % outCount;
	seedDimAdd(tempSeed);

	if (getCount == 0){
		outDims.push_back(resultDims);
		out.push_back(cl::Buffer(context, CL_MEM_READ_WRITE, sizeof(float) * outDims[0].size));
		inputs[0]->deriveDimentions(&outDims[0]);
	}
}

void Softmax::derive(){
	getCount = (getCount + 1) % outCount;
	if (getCount == 0){
		if (typeid(*inputs[0]) != typeid(Constant)){
			if (inputs[0]->getCount == 0){
				zeroBuffer(cl::EnqueueArgs(queue, cl::NullRange, cl::NDRange(inputs[0]->seedDims.size), cl::NullRange), inputs[0]->seed);
			}
			softmaxDerivative(cl::EnqueueArgs(queue, cl::NullRange, cl::NDRange(globalSize), cl::NDRange(GROUP_SIZE)), result, seedDims.dimBuf, seed, resedue, out[0], resultDims.dimentions[dimention], preSum, blocksWide);
			explodeUp(cl::EnqueueArgs(queue, cl::NullRange, cl::NDRange(inputs[0]->seedDims.size), cl::NullRange), outDims[0].dimBuf, out[0], inputs[0]->seedDims.dimBuf, inputs[0]->seed);
			inputs[0]->derive();
		}
	}
}

string Softmax::describe(){
	return name + "(" + inputs[0]->describe() + ", " + to_string(dimention) + ")";
}


void TanH::getValue(){
	if(getCount != 0){
		getCount = (getCount + 1) % outCount;
		return;
	}
	inputs[0]->getValue();
	tanH(cl::EnqueueArgs(queue, cl::NullRange, cl::NDRange(resultDims.size), cl::NullRange), inputs[0]->result, result);
	getCount = (getCount + 1) % outCount;
}

void TanH::derive(){
	getCount = (getCount + 1) % outCount;
	if (getCount == 0){
		if (typeid(*inputs[0]) != typeid(Constant)){
			if (inputs[0]->getCount == 0){
				zeroBuffer(cl::EnqueueArgs(queue, cl::NullRange, cl::NDRange(inputs[0]->seedDims.size), cl::NullRange), inputs[0]->seed);
			}
			tanHDerivative(cl::EnqueueArgs(queue, cl::NullRange, cl::NDRange(outDims[0].size), cl::NullRange), result, seedDims.dimBuf, seed, out[0]);
			explodeUp(cl::EnqueueArgs(queue, cl::NullRange, cl::NDRange(inputs[0]->seedDims.size), cl::NullRange), outDims[0].dimBuf, out[0], inputs[0]->seedDims.dimBuf, inputs[0]->seed);
			inputs[0]->derive();
		}
	}
}

void Softsign::getValue(){
	if(getCount != 0){
		getCount = (getCount + 1) % outCount;
		return;
	}
	inputs[0]->getValue();
	softsign(cl::EnqueueArgs(queue, cl::NullRange, cl::NDRange(resultDims.size), cl::NullRange), inputs[0]->result, result);
	getCount = (getCount + 1) % outCount;
}

void Softsign::derive(){
	getCount = (getCount + 1) % outCount;
	if (getCount == 0){
		if (typeid(*inputs[0]) != typeid(Constant)){
			if (inputs[0]->getCount == 0){
				zeroBuffer(cl::EnqueueArgs(queue, cl::NullRange, cl::NDRange(inputs[0]->seedDims.size), cl::NullRange), inputs[0]->seed);
			}
			softsignDerivative(cl::EnqueueArgs(queue, cl::NullRange, cl::NDRange(outDims[0].size), cl::NullRange), inputs[0]->result, seedDims.dimBuf, seed, out[0]);
			explodeUp(cl::EnqueueArgs(queue, cl::NullRange, cl::NDRange(inputs[0]->seedDims.size), cl::NullRange), outDims[0].dimBuf, out[0], inputs[0]->seedDims.dimBuf, inputs[0]->seed);
			inputs[0]->derive();
		}
	}
}


