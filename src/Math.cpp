#include "Math.hpp"
#include "Node.hpp"

void Add::getValue(){
	if(getCount != 0){
		getCount = (getCount + 1) % outCount;
		return;
	}
	inputs[0]->getValue();
	inputs[1]->getValue();
	add(cl::EnqueueArgs(queue, cl::NullRange, cl::NDRange(resultDims.size), cl::NullRange), inputs[0]->resultDims.dimBuf, inputs[0]->result, inputs[1]->resultDims.dimBuf, inputs[1]->result, result);
	getCount = (getCount + 1) % outCount;
}

void Add::deriveDimentions(GPUDimentions* tempSeed){
	getCount = (getCount + 1) % outCount;
	seedDimAdd(tempSeed);

	if (getCount == 0){
		outDims.push_back(seedDims);
		outDims.push_back(seedDims);
		out.push_back(cl::Buffer(context, CL_MEM_READ_WRITE, sizeof(float) * outDims[0].size));
		out.push_back(cl::Buffer(context, CL_MEM_READ_WRITE, sizeof(float) * outDims[1].size));
		inputs[0]->deriveDimentions(&outDims[0]);
		inputs[1]->deriveDimentions(&outDims[1]);
	}
}

void Add::derive(){
	getCount = (getCount + 1) % outCount;
	if (getCount == 0){
		if (typeid(*inputs[0]) != typeid(Constant)){
			if (inputs[0]->getCount == 0){
				zeroBuffer(cl::EnqueueArgs(queue, cl::NullRange, cl::NDRange(inputs[0]->seedDims.size), cl::NullRange), inputs[0]->seed);
			}
			addDerivative(cl::EnqueueArgs(queue, cl::NullRange, cl::NDRange(outDims[0].size), cl::NullRange), seedDims.dimBuf, seed, out[0]);
			if (outDims[0].rank > inputs[0]->seedDims.rank){
				reduceSum(cl::EnqueueArgs(queue, cl::NullRange, cl::NDRange(inputs[0]->seedDims.size), cl::NullRange), outDims[0].dimBuf, out[0], inputs[0]->seedDims.dimBuf, inputs[0]->seed);
			}
			else{
				explodeUp(cl::EnqueueArgs(queue, cl::NullRange, cl::NDRange(inputs[0]->seedDims.size), cl::NullRange), outDims[0].dimBuf, out[0], inputs[0]->seedDims.dimBuf, inputs[0]->seed);
			}
			inputs[0]->derive();
		}
		if (typeid(*inputs[1]) != typeid(Constant)){
			if (inputs[1]->getCount == 0){
				zeroBuffer(cl::EnqueueArgs(queue, cl::NullRange, cl::NDRange(inputs[1]->seedDims.size), cl::NullRange), inputs[1]->seed);
			}
			addDerivative(cl::EnqueueArgs(queue, cl::NullRange, cl::NDRange(outDims[1].size), cl::NullRange), seedDims.dimBuf, seed, out[1]);
			if (outDims[1].rank > inputs[1]->seedDims.rank){
				reduceSum(cl::EnqueueArgs(queue, cl::NullRange, cl::NDRange(inputs[1]->seedDims.size), cl::NullRange), outDims[1].dimBuf, out[1], inputs[1]->seedDims.dimBuf, inputs[1]->seed);
			}
			else{
				explodeUp(cl::EnqueueArgs(queue, cl::NullRange, cl::NDRange(inputs[1]->seedDims.size), cl::NullRange), outDims[1].dimBuf, out[1], inputs[1]->seedDims.dimBuf, inputs[1]->seed);
			}
			inputs[1]->derive();
		}
	}
}



void Subtract::getValue(){
	if(getCount != 0){
		getCount = (getCount + 1) % outCount;
		return;
	}
	inputs[0]->getValue();
	inputs[1]->getValue();
	subtract(cl::EnqueueArgs(queue, cl::NullRange, cl::NDRange(resultDims.size), cl::NullRange), inputs[0]->resultDims.dimBuf, inputs[0]->result, inputs[1]->resultDims.dimBuf, inputs[1]->result, result);
	getCount = (getCount + 1) % outCount;
}

void Subtract::deriveDimentions(GPUDimentions* tempSeed){
	getCount = (getCount + 1) % outCount;
	seedDimAdd(tempSeed);

	if (getCount == 0){
		outDims.push_back(seedDims);
		outDims.push_back(seedDims);
		out.push_back(cl::Buffer(context, CL_MEM_READ_WRITE, sizeof(float) * outDims[0].size));
		out.push_back(cl::Buffer(context, CL_MEM_READ_WRITE, sizeof(float) * outDims[1].size));
		inputs[0]->deriveDimentions(&outDims[0]);
		inputs[1]->deriveDimentions(&outDims[1]);
	}
}

void Subtract::derive(){
	getCount = (getCount + 1) % outCount;
	if (getCount == 0){
		if (typeid(*inputs[0]) != typeid(Constant)){
			if (inputs[0]->getCount == 0){
				zeroBuffer(cl::EnqueueArgs(queue, cl::NullRange, cl::NDRange(inputs[0]->seedDims.size), cl::NullRange), inputs[0]->seed);
			}
			addDerivative(cl::EnqueueArgs(queue, cl::NullRange, cl::NDRange(outDims[0].size), cl::NullRange), seedDims.dimBuf, seed, out[0]);
			if (outDims[0].rank > inputs[0]->seedDims.rank){
				reduceSum(cl::EnqueueArgs(queue, cl::NullRange, cl::NDRange(inputs[0]->seedDims.size), cl::NullRange), outDims[0].dimBuf, out[0], inputs[0]->seedDims.dimBuf, inputs[0]->seed);
			}
			else{
				explodeUp(cl::EnqueueArgs(queue, cl::NullRange, cl::NDRange(inputs[0]->seedDims.size), cl::NullRange), outDims[0].dimBuf, out[0], inputs[0]->seedDims.dimBuf, inputs[0]->seed);
			}
			inputs[0]->derive();
		}
		if (typeid(*inputs[1]) != typeid(Constant)){
			if (inputs[1]->getCount == 0){
				zeroBuffer(cl::EnqueueArgs(queue, cl::NullRange, cl::NDRange(inputs[1]->seedDims.size), cl::NullRange), inputs[1]->seed);
			}
			subtractDerivative1(cl::EnqueueArgs(queue, cl::NullRange, cl::NDRange(outDims[1].size), cl::NullRange), seedDims.dimBuf, seed, out[1]);
			if (outDims[1].rank > inputs[1]->seedDims.rank){
				reduceSum(cl::EnqueueArgs(queue, cl::NullRange, cl::NDRange(inputs[1]->seedDims.size), cl::NullRange), outDims[1].dimBuf, out[1], inputs[1]->seedDims.dimBuf, inputs[1]->seed);
			}
			else{
				explodeUp(cl::EnqueueArgs(queue, cl::NullRange, cl::NDRange(inputs[1]->seedDims.size), cl::NullRange), outDims[1].dimBuf, out[1], inputs[1]->seedDims.dimBuf, inputs[1]->seed);
			}
			inputs[1]->derive();
		}
	}
}



void Multiply::getValue(){
	if(getCount != 0){
		getCount = (getCount + 1) % outCount;
		return;
	}
	inputs[0]->getValue();
	inputs[1]->getValue();
	multiply(cl::EnqueueArgs(queue, cl::NullRange, cl::NDRange(resultDims.size), cl::NullRange), inputs[0]->resultDims.dimBuf, inputs[0]->result, inputs[1]->resultDims.dimBuf, inputs[1]->result, result);
	getCount = (getCount + 1) % outCount;
}

void Multiply::deriveDimentions(GPUDimentions* tempSeed){
	getCount = (getCount + 1) % outCount;
	seedDimAdd(tempSeed);

	if (getCount == 0){
		outDims.push_back(getMaxDimentions(vector<GPUDimentions*>{&seedDims, &inputs[1]->resultDims}));
		outDims.push_back(getMaxDimentions(vector<GPUDimentions*>{&seedDims, &inputs[0]->resultDims}));
		out.push_back(cl::Buffer(context, CL_MEM_READ_WRITE, sizeof(float) * outDims[0].size));
		out.push_back(cl::Buffer(context, CL_MEM_READ_WRITE, sizeof(float) * outDims[1].size));
		inputs[0]->deriveDimentions(&outDims[0]);
		inputs[1]->deriveDimentions(&outDims[1]);
	}
}

void Multiply::derive(){
	getCount = (getCount + 1) % outCount;
	if (getCount == 0){
		if (typeid(*inputs[0]) != typeid(Constant)){
			if (inputs[0]->getCount == 0){
				zeroBuffer(cl::EnqueueArgs(queue, cl::NullRange, cl::NDRange(inputs[0]->seedDims.size), cl::NullRange), inputs[0]->seed);
			}
			multiplyDerivative(cl::EnqueueArgs(queue, cl::NullRange, cl::NDRange(outDims[0].size), cl::NullRange), inputs[1]->resultDims.dimBuf, inputs[1]->result, seedDims.dimBuf, seed, out[0]);
			if (outDims[0].rank > inputs[0]->seedDims.rank){
				reduceSum(cl::EnqueueArgs(queue, cl::NullRange, cl::NDRange(inputs[0]->seedDims.size), cl::NullRange), outDims[0].dimBuf, out[0], inputs[0]->seedDims.dimBuf, inputs[0]->seed);
			}
			else{
				explodeUp(cl::EnqueueArgs(queue, cl::NullRange, cl::NDRange(inputs[0]->seedDims.size), cl::NullRange), outDims[0].dimBuf, out[0], inputs[0]->seedDims.dimBuf, inputs[0]->seed);
			}
			inputs[0]->derive();
		}
		if (typeid(*inputs[1]) != typeid(Constant)){
			if (inputs[1]->getCount == 0){
				zeroBuffer(cl::EnqueueArgs(queue, cl::NullRange, cl::NDRange(inputs[1]->seedDims.size), cl::NullRange), inputs[1]->seed);
			}
			multiplyDerivative(cl::EnqueueArgs(queue, cl::NullRange, cl::NDRange(outDims[1].size), cl::NullRange), inputs[0]->resultDims.dimBuf, inputs[0]->result, seedDims.dimBuf, seed, out[1]);
			if (outDims[1].rank > inputs[1]->seedDims.rank){
				reduceSum(cl::EnqueueArgs(queue, cl::NullRange, cl::NDRange(inputs[1]->seedDims.size), cl::NullRange), outDims[1].dimBuf, out[1], inputs[1]->seedDims.dimBuf, inputs[1]->seed);
			}
			else{
				explodeUp(cl::EnqueueArgs(queue, cl::NullRange, cl::NDRange(inputs[1]->seedDims.size), cl::NullRange), outDims[1].dimBuf, out[1], inputs[1]->seedDims.dimBuf, inputs[1]->seed);
			}
			inputs[1]->derive();
		}
	}
}



void Divide::getValue(){
	if(getCount != 0){
		getCount = (getCount + 1) % outCount;
		return;
	}
	inputs[0]->getValue();
	inputs[1]->getValue();
	divide(cl::EnqueueArgs(queue, cl::NullRange, cl::NDRange(resultDims.size), cl::NullRange), inputs[0]->resultDims.dimBuf, inputs[0]->result, inputs[1]->resultDims.dimBuf, inputs[1]->result, result);
	getCount = (getCount + 1) % outCount;
}

void Divide::deriveDimentions(GPUDimentions* tempSeed){
	getCount = (getCount + 1) % outCount;
	seedDimAdd(tempSeed);

	if (getCount == 0){
		outDims.push_back(getMaxDimentions(vector<GPUDimentions*>{&seedDims, &inputs[1]->resultDims}));
		outDims.push_back(getMaxDimentions(vector<GPUDimentions*>{&seedDims, &inputs[0]->resultDims, &inputs[1]->resultDims}));
		out.push_back(cl::Buffer(context, CL_MEM_READ_WRITE, sizeof(float) * outDims[0].size));
		out.push_back(cl::Buffer(context, CL_MEM_READ_WRITE, sizeof(float) * outDims[1].size));
		inputs[0]->deriveDimentions(&outDims[0]);
		inputs[1]->deriveDimentions(&outDims[1]);
	}
}

void Divide::derive(){
	getCount = (getCount + 1) % outCount;
	if (getCount == 0){
		if (typeid(*inputs[0]) != typeid(Constant)){
			if (inputs[0]->getCount == 0){
				zeroBuffer(cl::EnqueueArgs(queue, cl::NullRange, cl::NDRange(inputs[0]->seedDims.size), cl::NullRange), inputs[0]->seed);
			}
			divideDerivative0(cl::EnqueueArgs(queue, cl::NullRange, cl::NDRange(outDims[0].size), cl::NullRange), inputs[1]->resultDims.dimBuf, inputs[1]->result, seedDims.dimBuf, seed, out[0]);
			if (outDims[0].rank > inputs[0]->seedDims.rank){
				reduceSum(cl::EnqueueArgs(queue, cl::NullRange, cl::NDRange(inputs[0]->seedDims.size), cl::NullRange), outDims[0].dimBuf, out[0], inputs[0]->seedDims.dimBuf, inputs[0]->seed);
			}
			else{
				explodeUp(cl::EnqueueArgs(queue, cl::NullRange, cl::NDRange(inputs[0]->seedDims.size), cl::NullRange), outDims[0].dimBuf, out[0], inputs[0]->seedDims.dimBuf, inputs[0]->seed);
			}
			inputs[0]->derive();
		}
		if (typeid(*inputs[1]) != typeid(Constant)){
			if (inputs[1]->getCount == 0){
				zeroBuffer(cl::EnqueueArgs(queue, cl::NullRange, cl::NDRange(inputs[1]->seedDims.size), cl::NullRange), inputs[1]->seed);
			}
			divideDerivative1(cl::EnqueueArgs(queue, cl::NullRange, cl::NDRange(outDims[1].size), cl::NullRange), inputs[0]->resultDims.dimBuf, inputs[0]->result, inputs[1]->resultDims.dimBuf, inputs[1]->result, seedDims.dimBuf, seed, out[1]);
			if (outDims[1].rank > inputs[1]->seedDims.rank){
				reduceSum(cl::EnqueueArgs(queue, cl::NullRange, cl::NDRange(inputs[1]->seedDims.size), cl::NullRange), outDims[1].dimBuf, out[1], inputs[1]->seedDims.dimBuf, inputs[1]->seed);
			}
			else{
				explodeUp(cl::EnqueueArgs(queue, cl::NullRange, cl::NDRange(inputs[1]->seedDims.size), cl::NullRange), outDims[1].dimBuf, out[1], inputs[1]->seedDims.dimBuf, inputs[1]->seed);
			}
			inputs[1]->derive();
		}
	}
}



void Pow::getValue(){
	if(getCount != 0){
		getCount = (getCount + 1) % outCount;
		return;
	}
	inputs[0]->getValue();
	inputs[1]->getValue();
	pow_(cl::EnqueueArgs(queue, cl::NullRange, cl::NDRange(resultDims.size), cl::NullRange), inputs[0]->resultDims.dimBuf, inputs[0]->result, inputs[1]->resultDims.dimBuf, inputs[1]->result, result);
	getCount = (getCount + 1) % outCount;
}

void Pow::deriveDimentions(GPUDimentions* tempSeed){
	getCount = (getCount + 1) % outCount;
	seedDimAdd(tempSeed);

	if (getCount == 0){
		outDims.push_back(getMaxDimentions(vector<GPUDimentions*>{&seedDims, &inputs[0]->resultDims, &inputs[1]->resultDims}));
		outDims.push_back(getMaxDimentions(vector<GPUDimentions*>{&seedDims, &inputs[0]->resultDims, &resultDims}));
		out.push_back(cl::Buffer(context, CL_MEM_READ_WRITE, sizeof(float) * outDims[0].size));
		out.push_back(cl::Buffer(context, CL_MEM_READ_WRITE, sizeof(float) * outDims[1].size));
		inputs[0]->deriveDimentions(&outDims[0]);
		inputs[1]->deriveDimentions(&outDims[1]);
	}
}

void Pow::derive(){
	getCount = (getCount + 1) % outCount;
	if (getCount == 0){
		if (typeid(*inputs[0]) != typeid(Constant)){
			if (inputs[0]->getCount == 0){
				zeroBuffer(cl::EnqueueArgs(queue, cl::NullRange, cl::NDRange(inputs[0]->seedDims.size), cl::NullRange), inputs[0]->seed);
			}
			powDerivative0(cl::EnqueueArgs(queue, cl::NullRange, cl::NDRange(outDims[0].size), cl::NullRange), inputs[0]->resultDims.dimBuf, inputs[0]->result, inputs[1]->resultDims.dimBuf, inputs[1]->result, seedDims.dimBuf, seed, out[0]);
			if (outDims[0].rank > inputs[0]->seedDims.rank){
				reduceSum(cl::EnqueueArgs(queue, cl::NullRange, cl::NDRange(inputs[0]->seedDims.size), cl::NullRange), outDims[0].dimBuf, out[0], inputs[0]->seedDims.dimBuf, inputs[0]->seed);
			}
			else{
				explodeUp(cl::EnqueueArgs(queue, cl::NullRange, cl::NDRange(inputs[0]->seedDims.size), cl::NullRange), outDims[0].dimBuf, out[0], inputs[0]->seedDims.dimBuf, inputs[0]->seed);
			}
			inputs[0]->derive();
		}
		if (typeid(*inputs[1]) != typeid(Constant)){
			if (inputs[1]->getCount == 0){
				zeroBuffer(cl::EnqueueArgs(queue, cl::NullRange, cl::NDRange(inputs[1]->seedDims.size), cl::NullRange), inputs[1]->seed);
			}
			powDerivative1(cl::EnqueueArgs(queue, cl::NullRange, cl::NDRange(outDims[1].size), cl::NullRange), inputs[0]->resultDims.dimBuf, inputs[0]->result, resultDims.dimBuf, result, seedDims.dimBuf, seed, out[1]);
			if (outDims[1].rank > inputs[1]->seedDims.rank){
				reduceSum(cl::EnqueueArgs(queue, cl::NullRange, cl::NDRange(inputs[1]->seedDims.size), cl::NullRange), outDims[1].dimBuf, out[1], inputs[1]->seedDims.dimBuf, inputs[1]->seed);
			}
			else{
				explodeUp(cl::EnqueueArgs(queue, cl::NullRange, cl::NDRange(inputs[1]->seedDims.size), cl::NullRange), outDims[1].dimBuf, out[1], inputs[1]->seedDims.dimBuf, inputs[1]->seed);
			}
			inputs[1]->derive();
		}
	}
}



void Ln::getValue(){
	if(getCount != 0){
		getCount = (getCount + 1) % outCount;
		return;
	}
	inputs[0]->getValue();
	ln(cl::EnqueueArgs(queue, cl::NullRange, cl::NDRange(resultDims.size), cl::NullRange), inputs[0]->result, result);
	getCount = (getCount + 1) % outCount;
}

void Ln::deriveDimentions(GPUDimentions* tempSeed){
	getCount = (getCount + 1) % outCount;
	seedDimAdd(tempSeed);

	if (getCount == 0){
		outDims.push_back(getMaxDimentions(vector<GPUDimentions*>{&seedDims, &inputs[0]->resultDims}));
		out.push_back(cl::Buffer(context, CL_MEM_READ_WRITE, sizeof(float) * outDims[0].size));
		inputs[0]->deriveDimentions(&outDims[0]);
	}
}

void Ln::derive(){
	getCount = (getCount + 1) % outCount;
	if (getCount == 0){
		if (typeid(*inputs[0]) != typeid(Constant)){
			if (inputs[0]->getCount == 0){
				zeroBuffer(cl::EnqueueArgs(queue, cl::NullRange, cl::NDRange(inputs[0]->seedDims.size), cl::NullRange), inputs[0]->seed);
			}
			divideDerivative0(cl::EnqueueArgs(queue, cl::NullRange, cl::NDRange(outDims[0].size), cl::NullRange), inputs[0]->resultDims.dimBuf, inputs[0]->result, seedDims.dimBuf, seed, out[0]);
			if (outDims[0].rank > inputs[0]->seedDims.rank){
				reduceSum(cl::EnqueueArgs(queue, cl::NullRange, cl::NDRange(inputs[0]->seedDims.size), cl::NullRange), outDims[0].dimBuf, out[0], inputs[0]->seedDims.dimBuf, inputs[0]->seed);
			}
			else{
				explodeUp(cl::EnqueueArgs(queue, cl::NullRange, cl::NDRange(inputs[0]->seedDims.size), cl::NullRange), outDims[0].dimBuf, out[0], inputs[0]->seedDims.dimBuf, inputs[0]->seed);
			}
			inputs[0]->derive();
		}
	}
}



void Exp::getValue(){
	if(getCount != 0){
		getCount = (getCount + 1) % outCount;
		return;
	}
	inputs[0]->getValue();
	exp_(cl::EnqueueArgs(queue, cl::NullRange, cl::NDRange(resultDims.size), cl::NullRange), inputs[0]->result, result);
	getCount = (getCount + 1) % outCount;
}

void Exp::deriveDimentions(GPUDimentions* tempSeed){
	getCount = (getCount + 1) % outCount;
	seedDimAdd(tempSeed);

	if (getCount == 0){
		outDims.push_back(getMaxDimentions(vector<GPUDimentions*>{&seedDims, &resultDims}));
		out.push_back(cl::Buffer(context, CL_MEM_READ_WRITE, sizeof(float) * outDims[0].size));
		inputs[0]->deriveDimentions(&outDims[0]);
	}
}

void Exp::derive(){
	getCount = (getCount + 1) % outCount;
	if (getCount == 0){
		if (typeid(*inputs[0]) != typeid(Constant)){
			if (inputs[0]->getCount == 0){
				zeroBuffer(cl::EnqueueArgs(queue, cl::NullRange, cl::NDRange(inputs[0]->seedDims.size), cl::NullRange), inputs[0]->seed);
			}
			multiplyDerivative(cl::EnqueueArgs(queue, cl::NullRange, cl::NDRange(outDims[0].size), cl::NullRange), resultDims.dimBuf, result, seedDims.dimBuf, seed, out[0]);
			if (outDims[0].rank > inputs[0]->seedDims.rank){
				reduceSum(cl::EnqueueArgs(queue, cl::NullRange, cl::NDRange(inputs[0]->seedDims.size), cl::NullRange), outDims[0].dimBuf, out[0], inputs[0]->seedDims.dimBuf, inputs[0]->seed);
			}
			else{
				explodeUp(cl::EnqueueArgs(queue, cl::NullRange, cl::NDRange(inputs[0]->seedDims.size), cl::NullRange), outDims[0].dimBuf, out[0], inputs[0]->seedDims.dimBuf, inputs[0]->seed);
			}
			inputs[0]->derive();
		}
	}
}



// Log::Log(Node* a, float baseVal): BasicFunction(a){
// 	name = "Log";
// 	base = baseVal;
// }

// void Log::getValue(){
// 	if(getCount != 0){
// 		getCount = (getCount + 1) % outCount;
// 		return;
// 	}
// 	inputs[0]->getValue();
// 	exp_(cl::EnqueueArgs(queue, cl::NullRange, cl::NDRange(resultDims.size), cl::NullRange), inputs[0]->result, result);
// 	getCount = (getCount + 1) % outCount;
// }

// // void Log::derive(NumObject& seed, int t, int tf){
// // 	if(sumSeed(seed)){
// // 		if (typeid(*inputs[0]) != typeid(Constant)){
// // 			vector<NumObject> items1 = {tempSeed, inputs[0]->derivativeMemo[t]};
// // 			NumObject eval1 = mapVals(this, &Log::deriveOperation1, items1);
// // 			inputs[0]->derive(eval1, t, tf);
// // 		}
// // 	}
// // }

// // double Log::deriveOperation1(vector<double>& a){
// // 	return a[0] / (a[1] * log(base));
// // }

// string Log::describe(){
// 	return name + "(" + inputs[0]->describe() + ", " + to_string(base) + ")";
// }

// double Sin::operation(vector<double>& a){
// 	return sin(a[0]);
// }

// void Sin::derive(NumObject& seed, int t, int tf){
// 	if(sumSeed(seed)){
// 		if (typeid(*inputs[0]) != typeid(Constant)){
// 			vector<NumObject> items1 = {tempSeed, inputs[0]->derivativeMemo[t]};
// 			NumObject eval1 = mapVals(this, &Sin::deriveOperation1, items1);
// 			inputs[0]->derive(eval1, t, tf);
// 		}
// 	}
// }

// double Sin::deriveOperation1(vector<double>& a){
// 	return a[0] * cos(a[1]);
// }

// double Cos::operation(vector<double>& a){
// 	return cos(a[0]);
// }

// void Cos::derive(NumObject& seed, int t, int tf){
// 	if(sumSeed(seed)){
// 		if (typeid(*inputs[0]) != typeid(Constant)){
// 			vector<NumObject> items1 = {tempSeed, inputs[0]->derivativeMemo[t]};
// 			NumObject eval1 = mapVals(this, &Cos::deriveOperation1, items1);
// 			inputs[0]->derive(eval1, t, tf);
// 		}
// 	}
// }

// double Cos::deriveOperation1(vector<double>& a){
// 	return -a[0] * sin(a[1]);
// }

// double Tan::operation(vector<double>& a){
// 	return tan(a[0]);
// }

// void Tan::derive(NumObject& seed, int t, int tf){
// 	if(sumSeed(seed)){
// 		if (typeid(*inputs[0]) != typeid(Constant)){
// 			vector<NumObject> items1 = {tempSeed, inputs[0]->derivativeMemo[t]};
// 			NumObject eval1 = mapVals(this, &Tan::deriveOperation1, items1);
// 			inputs[0]->derive(eval1, t, tf);
// 		}
// 	}
// }

// double Tan::deriveOperation1(vector<double>& a){
// 	return a[0] * (1.0 / cos(a[1])) * (1.0 / cos(a[1]));
// }

// double ArcSin::operation(vector<double>& a){
// 	return asin(a[0]);
// }

// void ArcSin::derive(NumObject& seed, int t, int tf){
// 	if(sumSeed(seed)){
// 		if (typeid(*inputs[0]) != typeid(Constant)){
// 			vector<NumObject> items1 = {tempSeed, inputs[0]->derivativeMemo[t]};
// 			NumObject eval1 = mapVals(this, &ArcSin::deriveOperation1, items1);
// 			inputs[0]->derive(eval1, t, tf);
// 		}
// 	}
// }

// double ArcSin::deriveOperation1(vector<double>& a){
// 	return a[0] / pow(1.0 - a[1] * a[1], 0.5);
// }

// double ArcCos::operation(vector<double>& a){
// 	return acos(a[0]);
// }

// void ArcCos::derive(NumObject& seed, int t, int tf){
// 	if(sumSeed(seed)){
// 		if (typeid(*inputs[0]) != typeid(Constant)){
// 			vector<NumObject> items1 = {tempSeed, inputs[0]->derivativeMemo[t]};
// 			NumObject eval1 = mapVals(this, &ArcCos::deriveOperation1, items1);
// 			inputs[0]->derive(eval1, t, tf);
// 		}
// 	}
// }

// double ArcCos::deriveOperation1(vector<double>& a){
// 	return -a[0] / pow(1.0 - a[1] * a[1], 0.5);
// }

// double ArcTan::operation(vector<double>& a){
// 	return atan(a[0]);
// }

// void ArcTan::derive(NumObject& seed, int t, int tf){
// 	if(sumSeed(seed)){
// 		if (typeid(*inputs[0]) != typeid(Constant)){
// 			vector<NumObject> items1 = {tempSeed, inputs[0]->derivativeMemo[t]};
// 			NumObject eval1 = mapVals(this, &ArcTan::deriveOperation1, items1);
// 			inputs[0]->derive(eval1, t, tf);
// 		}
// 	}
// }

// double ArcTan::deriveOperation1(vector<double>& a){
// 	return a[0] / (1.0 + a[1] * a[1]);
// }

