#include "Node.hpp"

cl::Context context;
cl::CommandQueue queue;
cl::Program program;

cl::make_kernel<cl::Buffer> zeroBuffer(cl::Kernel(program, ""));
cl::make_kernel<cl::Buffer, cl::Buffer, cl::Buffer, cl::Buffer> reduceSum(cl::Kernel(program, ""));
cl::make_kernel<cl::Buffer, cl::Buffer, cl::Buffer, cl::Buffer> explodeUp(cl::Kernel(program, ""));
cl::make_kernel<cl::Buffer, cl::Buffer, cl::Buffer, float> gradientDescentStep(cl::Kernel(program, ""));

cl::make_kernel<cl::Buffer, cl::Buffer, cl::Buffer, cl::Buffer, cl::Buffer> add(cl::Kernel(program, ""));
cl::make_kernel<cl::Buffer, cl::Buffer, cl::Buffer, cl::Buffer, cl::Buffer> subtract(cl::Kernel(program, ""));
cl::make_kernel<cl::Buffer, cl::Buffer, cl::Buffer, cl::Buffer, cl::Buffer> multiply(cl::Kernel(program, ""));
cl::make_kernel<cl::Buffer, cl::Buffer, cl::Buffer, cl::Buffer, cl::Buffer> divide(cl::Kernel(program, ""));
cl::make_kernel<cl::Buffer, cl::Buffer, cl::Buffer, cl::Buffer, cl::Buffer> pow_(cl::Kernel(program, ""));
cl::make_kernel<cl::Buffer, cl::Buffer> ln(cl::Kernel(program, ""));
cl::make_kernel<cl::Buffer, cl::Buffer> exp_(cl::Kernel(program, ""));
cl::make_kernel<cl::Buffer, cl::Buffer, float> log_(cl::Kernel(program, ""));
cl::make_kernel<cl::Buffer, cl::Buffer> sin_(cl::Kernel(program, ""));
cl::make_kernel<cl::Buffer, cl::Buffer> cos_(cl::Kernel(program, ""));
cl::make_kernel<cl::Buffer, cl::Buffer> tan_(cl::Kernel(program, ""));
cl::make_kernel<cl::Buffer, cl::Buffer> asin_(cl::Kernel(program, ""));
cl::make_kernel<cl::Buffer, cl::Buffer> acos_(cl::Kernel(program, ""));
cl::make_kernel<cl::Buffer, cl::Buffer> atan_(cl::Kernel(program, ""));
cl::make_kernel<cl::Buffer, cl::Buffer, cl::Buffer, cl::Buffer, cl::Buffer> matMul2x2(cl::Kernel(program, ""));
cl::make_kernel<cl::Buffer, cl::Buffer, cl::Buffer, cl::Buffer, cl::Buffer> matMul2x1(cl::Kernel(program, ""));
cl::make_kernel<cl::Buffer, cl::Buffer, cl::Buffer, cl::Buffer, cl::Buffer> matMul1x2(cl::Kernel(program, ""));
cl::make_kernel<cl::Buffer, cl::Buffer, cl::Buffer, cl::Buffer, cl::Buffer> matMul1x1(cl::Kernel(program, ""));
cl::make_kernel<cl::Buffer, cl::Buffer, int, int> sum_(cl::Kernel(program, ""));
cl::make_kernel<cl::Buffer, cl::Buffer, int, int> mean_(cl::Kernel(program, ""));
cl::make_kernel<cl::Buffer, cl::Buffer, cl::Buffer> trans(cl::Kernel(program, ""));
cl::make_kernel<cl::Buffer, cl::Buffer, cl::Buffer, int, int> max_(cl::Kernel(program, ""));
cl::make_kernel<cl::Buffer, cl::Buffer, cl::Buffer, int, int> min_(cl::Kernel(program, ""));
cl::make_kernel<cl::Buffer, cl::Buffer, cl::Buffer, cl::Buffer> meanSquared(cl::Kernel(program, ""));
cl::make_kernel<cl::Buffer, cl::Buffer, cl::Buffer> crossEntropy(cl::Kernel(program, ""));
cl::make_kernel<cl::Buffer, cl::Buffer> sigmoid(cl::Kernel(program, ""));

cl::make_kernel<cl::Buffer, cl::Buffer, cl::Buffer> addDerivative(cl::Kernel(program, ""));
cl::make_kernel<cl::Buffer, cl::Buffer, cl::Buffer> subtractDerivative1(cl::Kernel(program, ""));
cl::make_kernel<cl::Buffer, cl::Buffer, cl::Buffer, cl::Buffer, cl::Buffer> multiplyDerivative(cl::Kernel(program, ""));
cl::make_kernel<cl::Buffer, cl::Buffer, cl::Buffer, cl::Buffer, cl::Buffer> divideDerivative0(cl::Kernel(program, ""));
cl::make_kernel<cl::Buffer, cl::Buffer, cl::Buffer, cl::Buffer, cl::Buffer, cl::Buffer, cl::Buffer> divideDerivative1(cl::Kernel(program, ""));
cl::make_kernel<cl::Buffer, cl::Buffer, cl::Buffer, cl::Buffer, cl::Buffer, cl::Buffer, cl::Buffer> powDerivative0(cl::Kernel(program, ""));
cl::make_kernel<cl::Buffer, cl::Buffer, cl::Buffer, cl::Buffer, cl::Buffer, cl::Buffer, cl::Buffer> powDerivative1(cl::Kernel(program, ""));
cl::make_kernel<cl::Buffer, float, cl::Buffer, cl::Buffer, cl::Buffer> logDerivative(cl::Kernel(program, ""));
cl::make_kernel<cl::Buffer, cl::Buffer, cl::Buffer, cl::Buffer> sinDerivative(cl::Kernel(program, ""));
cl::make_kernel<cl::Buffer, cl::Buffer, cl::Buffer, cl::Buffer> cosDerivative(cl::Kernel(program, ""));
cl::make_kernel<cl::Buffer, cl::Buffer, cl::Buffer, cl::Buffer> tanDerivative(cl::Kernel(program, ""));
cl::make_kernel<cl::Buffer, cl::Buffer, cl::Buffer, cl::Buffer> asinDerivative(cl::Kernel(program, ""));
cl::make_kernel<cl::Buffer, cl::Buffer, cl::Buffer, cl::Buffer> acosDerivative(cl::Kernel(program, ""));
cl::make_kernel<cl::Buffer, cl::Buffer, cl::Buffer, cl::Buffer> atanDerivative(cl::Kernel(program, ""));
cl::make_kernel<cl::Buffer, cl::Buffer, cl::Buffer, cl::Buffer, cl::Buffer> matMul2x2Derivative0(cl::Kernel(program, ""));
cl::make_kernel<cl::Buffer, cl::Buffer, cl::Buffer, cl::Buffer, cl::Buffer, cl::Buffer> matMul2x2Derivative1(cl::Kernel(program, ""));
cl::make_kernel<cl::Buffer, cl::Buffer, cl::Buffer, cl::Buffer, cl::Buffer> matMul2x1Derivative0(cl::Kernel(program, ""));
cl::make_kernel<cl::Buffer, cl::Buffer, cl::Buffer, cl::Buffer, cl::Buffer> matMul2x1Derivative1(cl::Kernel(program, ""));
cl::make_kernel<cl::Buffer, cl::Buffer, cl::Buffer, cl::Buffer, cl::Buffer> matMul1x2Derivative0(cl::Kernel(program, ""));
cl::make_kernel<cl::Buffer, cl::Buffer, cl::Buffer, cl::Buffer, cl::Buffer> matMul1x2Derivative1(cl::Kernel(program, ""));
cl::make_kernel<cl::Buffer, cl::Buffer, cl::Buffer> matMul1x1Derivative0(cl::Kernel(program, ""));
cl::make_kernel<cl::Buffer, cl::Buffer, cl::Buffer> matMul1x1Derivative1(cl::Kernel(program, ""));
cl::make_kernel<cl::Buffer, cl::Buffer, cl::Buffer, int, int> sumDerivative(cl::Kernel(program, ""));
cl::make_kernel<cl::Buffer, cl::Buffer, cl::Buffer, int, int> meanDerivative(cl::Kernel(program, ""));
cl::make_kernel<cl::Buffer, cl::Buffer, int> meanDerivativeSmallSeed(cl::Kernel(program, ""));
cl::make_kernel<cl::Buffer, cl::Buffer, cl::Buffer> transDerivative1(cl::Kernel(program, ""));
cl::make_kernel<cl::Buffer, cl::Buffer, cl::Buffer> transDerivative2(cl::Kernel(program, ""));
cl::make_kernel<cl::Buffer, cl::Buffer, cl::Buffer, cl::Buffer, int, int> maxDerivative(cl::Kernel(program, ""));
cl::make_kernel<cl::Buffer, cl::Buffer, cl::Buffer, cl::Buffer> maxDerivativeSmallSeed(cl::Kernel(program, ""));
cl::make_kernel<cl::Buffer, cl::Buffer, cl::Buffer, cl::Buffer, int, int> meanSquaredDerivative(cl::Kernel(program, ""));
cl::make_kernel<cl::Buffer, cl::Buffer, cl::Buffer, cl::Buffer, int> meanSquaredDerivativeSmallSeed(cl::Kernel(program, ""));
cl::make_kernel<cl::Buffer, cl::Buffer, cl::Buffer, cl::Buffer, cl::Buffer, int, int> crossEntropyDerivative(cl::Kernel(program, ""));
cl::make_kernel<cl::Buffer, cl::Buffer, cl::Buffer, cl::Buffer, cl::Buffer, int> crossEntropyDerivativeSmallSeed(cl::Kernel(program, ""));

void initialize(){
	vector<cl::Platform> allPlatforms;
	cl::Platform::get(&allPlatforms);

	cl::Platform defaultPlatform = allPlatforms[0];
	cout << "Using platform: " << defaultPlatform.getInfo<CL_PLATFORM_NAME>() << endl;

	vector<cl::Device> allDevices;
	vector<cl::Device> tempDevices;
	defaultPlatform.getDevices(CL_DEVICE_TYPE_ALL, &tempDevices);
	allDevices.push_back(tempDevices[1]);

	cl::Device defaultDevice = allDevices[0];
	cout << "Using device: " << defaultDevice.getInfo<CL_DEVICE_NAME>() << endl;

	context = {defaultDevice};

	cl::Program::Sources sources;

	FILE* fp;
	char* source_str;
	size_t program_size;

	fp = fopen("operations.cl.hpp", "rb");
	fseek(fp, 0, SEEK_END);
	program_size = ftell(fp);
	rewind(fp);
	source_str = (char*)malloc(program_size + 1);
	source_str[program_size] = '\0';
	fread(source_str, sizeof(char), program_size, fp);
	fclose(fp);

	sources.push_back(make_pair(source_str, program_size));

	program = {context, sources};

	if (program.build(allDevices) != CL_SUCCESS){
		cout << " Error building: " << program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(defaultDevice) << endl;
		exit(1);
	}

	queue = {context, defaultDevice};

	zeroBuffer = cl::make_kernel<cl::Buffer>(cl::Kernel(program, "zeroBuffer"));
	reduceSum = cl::make_kernel<cl::Buffer, cl::Buffer, cl::Buffer, cl::Buffer>(cl::Kernel(program, "reduceSum"));
	explodeUp = cl::make_kernel<cl::Buffer, cl::Buffer, cl::Buffer, cl::Buffer>(cl::Kernel(program, "explodeUp"));
	gradientDescentStep = cl::make_kernel<cl::Buffer, cl::Buffer, cl::Buffer, float>(cl::Kernel(program, "gradientDescentStep"));

	add = cl::make_kernel<cl::Buffer, cl::Buffer, cl::Buffer, cl::Buffer, cl::Buffer>(cl::Kernel(program, "add"));
	subtract = cl::make_kernel<cl::Buffer, cl::Buffer, cl::Buffer, cl::Buffer, cl::Buffer>(cl::Kernel(program, "subtract"));
	multiply = cl::make_kernel<cl::Buffer, cl::Buffer, cl::Buffer, cl::Buffer, cl::Buffer>(cl::Kernel(program, "multiply"));
	divide = cl::make_kernel<cl::Buffer, cl::Buffer, cl::Buffer, cl::Buffer, cl::Buffer>(cl::Kernel(program, "divide"));
	pow_ = cl::make_kernel<cl::Buffer, cl::Buffer, cl::Buffer, cl::Buffer, cl::Buffer>(cl::Kernel(program, "pow_"));
	ln = cl::make_kernel<cl::Buffer, cl::Buffer>(cl::Kernel(program, "ln"));
	exp_ = cl::make_kernel<cl::Buffer, cl::Buffer>(cl::Kernel(program, "exp_"));
	log_ = cl::make_kernel<cl::Buffer, cl::Buffer, float>(cl::Kernel(program, "log_"));
	sin_ = cl::make_kernel<cl::Buffer, cl::Buffer>(cl::Kernel(program, "sin_"));
	cos_ = cl::make_kernel<cl::Buffer, cl::Buffer>(cl::Kernel(program, "cos_"));
	tan_ = cl::make_kernel<cl::Buffer, cl::Buffer>(cl::Kernel(program, "tan_"));
	asin_ = cl::make_kernel<cl::Buffer, cl::Buffer>(cl::Kernel(program, "asin_"));
	acos_ = cl::make_kernel<cl::Buffer, cl::Buffer>(cl::Kernel(program, "acos_"));
	atan_ = cl::make_kernel<cl::Buffer, cl::Buffer>(cl::Kernel(program, "atan_"));
	matMul2x2 = cl::make_kernel<cl::Buffer, cl::Buffer, cl::Buffer, cl::Buffer, cl::Buffer>(cl::Kernel(program, "matMul2x2"));
	matMul2x1 = cl::make_kernel<cl::Buffer, cl::Buffer, cl::Buffer, cl::Buffer, cl::Buffer>(cl::Kernel(program, "matMul2x1"));
	matMul1x2 = cl::make_kernel<cl::Buffer, cl::Buffer, cl::Buffer, cl::Buffer, cl::Buffer>(cl::Kernel(program, "matMul1x2"));
	matMul1x1 = cl::make_kernel<cl::Buffer, cl::Buffer, cl::Buffer, cl::Buffer, cl::Buffer>(cl::Kernel(program, "matMul1x1"));
	sum_ = cl::make_kernel<cl::Buffer, cl::Buffer, int, int>(cl::Kernel(program, "sum_"));
	mean_ = cl::make_kernel<cl::Buffer, cl::Buffer, int, int>(cl::Kernel(program, "mean_"));
	trans = cl::make_kernel<cl::Buffer, cl::Buffer, cl::Buffer>(cl::Kernel(program, "trans"));
	max_ = cl::make_kernel<cl::Buffer, cl::Buffer, cl::Buffer, int, int>(cl::Kernel(program, "max_"));
	min_ = cl::make_kernel<cl::Buffer, cl::Buffer, cl::Buffer, int, int>(cl::Kernel(program, "min_"));
	meanSquared = cl::make_kernel<cl::Buffer, cl::Buffer, cl::Buffer, cl::Buffer>(cl::Kernel(program, "meanSquared"));
	crossEntropy = cl::make_kernel<cl::Buffer, cl::Buffer, cl::Buffer>(cl::Kernel(program, "crossEntropy"));
	sigmoid = cl::make_kernel<cl::Buffer, cl::Buffer>(cl::Kernel(program, "sigmoid"));

	addDerivative = cl::make_kernel<cl::Buffer, cl::Buffer, cl::Buffer>(cl::Kernel(program, "addDerivative"));
	subtractDerivative1 = cl::make_kernel<cl::Buffer, cl::Buffer, cl::Buffer>(cl::Kernel(program, "subtractDerivative1"));
	multiplyDerivative = cl::make_kernel<cl::Buffer, cl::Buffer, cl::Buffer, cl::Buffer, cl::Buffer>(cl::Kernel(program, "multiplyDerivative"));
	divideDerivative0 = cl::make_kernel<cl::Buffer, cl::Buffer, cl::Buffer, cl::Buffer, cl::Buffer>(cl::Kernel(program, "divideDerivative0"));
	divideDerivative1 = cl::make_kernel<cl::Buffer, cl::Buffer, cl::Buffer, cl::Buffer, cl::Buffer, cl::Buffer, cl::Buffer>(cl::Kernel(program, "divideDerivative1"));
	powDerivative0 = cl::make_kernel<cl::Buffer, cl::Buffer, cl::Buffer, cl::Buffer, cl::Buffer, cl::Buffer, cl::Buffer>(cl::Kernel(program, "powDerivative0"));
	powDerivative1 = cl::make_kernel<cl::Buffer, cl::Buffer, cl::Buffer, cl::Buffer, cl::Buffer, cl::Buffer, cl::Buffer>(cl::Kernel(program, "powDerivative1"));
	logDerivative = cl::make_kernel<cl::Buffer, float, cl::Buffer, cl::Buffer, cl::Buffer>(cl::Kernel(program, "logDerivative"));
	sinDerivative = cl::make_kernel<cl::Buffer, cl::Buffer, cl::Buffer, cl::Buffer>(cl::Kernel(program, "sinDerivative"));
	cosDerivative = cl::make_kernel<cl::Buffer, cl::Buffer, cl::Buffer, cl::Buffer>(cl::Kernel(program, "cosDerivative"));
	tanDerivative = cl::make_kernel<cl::Buffer, cl::Buffer, cl::Buffer, cl::Buffer>(cl::Kernel(program, "tanDerivative"));
	asinDerivative = cl::make_kernel<cl::Buffer, cl::Buffer, cl::Buffer, cl::Buffer>(cl::Kernel(program, "asinDerivative"));
	acosDerivative = cl::make_kernel<cl::Buffer, cl::Buffer, cl::Buffer, cl::Buffer>(cl::Kernel(program, "acosDerivative"));
	atanDerivative = cl::make_kernel<cl::Buffer, cl::Buffer, cl::Buffer, cl::Buffer>(cl::Kernel(program, "atanDerivative"));
	matMul2x2Derivative0 = cl::make_kernel<cl::Buffer, cl::Buffer, cl::Buffer, cl::Buffer, cl::Buffer>(cl::Kernel(program, "matMul2x2Derivative0"));
	matMul2x2Derivative1 = cl::make_kernel<cl::Buffer, cl::Buffer, cl::Buffer, cl::Buffer, cl::Buffer, cl::Buffer>(cl::Kernel(program, "matMul2x2Derivative1"));
	matMul2x1Derivative0 = cl::make_kernel<cl::Buffer, cl::Buffer, cl::Buffer, cl::Buffer, cl::Buffer>(cl::Kernel(program, "matMul2x1Derivative0"));
	matMul2x1Derivative1 = cl::make_kernel<cl::Buffer, cl::Buffer, cl::Buffer, cl::Buffer, cl::Buffer>(cl::Kernel(program, "matMul2x1Derivative1"));
	matMul1x2Derivative0 = cl::make_kernel<cl::Buffer, cl::Buffer, cl::Buffer, cl::Buffer, cl::Buffer>(cl::Kernel(program, "matMul1x2Derivative0"));
	matMul1x2Derivative1 = cl::make_kernel<cl::Buffer, cl::Buffer, cl::Buffer, cl::Buffer, cl::Buffer>(cl::Kernel(program, "matMul1x2Derivative1"));
	matMul1x1Derivative0 = cl::make_kernel<cl::Buffer, cl::Buffer, cl::Buffer>(cl::Kernel(program, "matMul1x1Derivative0"));
	matMul1x1Derivative1 = cl::make_kernel<cl::Buffer, cl::Buffer, cl::Buffer>(cl::Kernel(program, "matMul1x1Derivative1"));
	sumDerivative = cl::make_kernel<cl::Buffer, cl::Buffer, cl::Buffer, int, int>(cl::Kernel(program, "sumDerivative"));
	meanDerivative = cl::make_kernel<cl::Buffer, cl::Buffer, cl::Buffer, int, int>(cl::Kernel(program, "meanDerivative"));
	meanDerivativeSmallSeed = cl::make_kernel<cl::Buffer, cl::Buffer, int>(cl::Kernel(program, "meanDerivativeSmallSeed"));
	transDerivative1 = cl::make_kernel<cl::Buffer, cl::Buffer, cl::Buffer>(cl::Kernel(program, "transDerivative1"));
	transDerivative2 = cl::make_kernel<cl::Buffer, cl::Buffer, cl::Buffer>(cl::Kernel(program, "transDerivative2"));
	maxDerivative = cl::make_kernel<cl::Buffer, cl::Buffer, cl::Buffer, cl::Buffer, int, int>(cl::Kernel(program, "maxDerivative"));
	maxDerivativeSmallSeed = cl::make_kernel<cl::Buffer, cl::Buffer, cl::Buffer, cl::Buffer>(cl::Kernel(program, "maxDerivativeSmallSeed"));
	meanSquaredDerivative = cl::make_kernel<cl::Buffer, cl::Buffer, cl::Buffer, cl::Buffer, int, int>(cl::Kernel(program, "meanSquaredDerivative"));
	meanSquaredDerivativeSmallSeed = cl::make_kernel<cl::Buffer, cl::Buffer, cl::Buffer, cl::Buffer, int>(cl::Kernel(program, "meanSquaredDerivativeSmallSeed"));
	crossEntropyDerivative = cl::make_kernel<cl::Buffer, cl::Buffer, cl::Buffer, cl::Buffer, cl::Buffer, int, int>(cl::Kernel(program, "crossEntropyDerivative"));
	crossEntropyDerivativeSmallSeed = cl::make_kernel<cl::Buffer, cl::Buffer, cl::Buffer, cl::Buffer, cl::Buffer, int>(cl::Kernel(program, "crossEntropyDerivativeSmallSeed"));
}

NumObject::NumObject(){}

NumObject::NumObject(float val){
	rank = 0;
	size = 1;
	values.push_back(val);
}

NumObject::NumObject(vector<float> val, vector<int> dimentionsList){
	rank = dimentionsList.size();
	size = val.size();
	dimentions = dimentionsList;
	values = val;
}

NumObject::NumObject(vector<int> dimentionsList, float fill){
	rank = dimentionsList.size();
	dimentions = dimentionsList;

	int tempSize = 1;
	for (int i = 0; i < rank; i++){
		tempSize *= dimentions[i];
	}
	size = tempSize;

	values.resize(tempSize, fill);
}

NumObject::NumObject(vector<int> dimentionsList){
	rank = dimentionsList.size();
	dimentions = dimentionsList;

	int tempSize = 1;
	for (int i = 0; i < rank; i++){
		tempSize *= dimentions[i];
	}
	size = tempSize;

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

GPUDimentions::GPUDimentions(){
	rank = 0;
	size = 1;
	dimentions = vector<int>{};
}

GPUDimentions::GPUDimentions(int rankVal, int sizeVal, vector<int> dimentionsVals){
	rank = rankVal;
	size = sizeVal;
	dimentions = dimentionsVals;
	setBuf();
}

void GPUDimentions::setBuf(){
	dimBuf = cl::Buffer(context, CL_MEM_READ_WRITE, sizeof(int) * (rank + 2));
	vector<int> tempDimentions;
	tempDimentions.reserve(rank + 2);
	tempDimentions.push_back(size);
	tempDimentions.push_back(rank);
	for (int i = 0; i < rank; i++){
		tempDimentions.push_back(dimentions[i]);
	}
	queue.enqueueWriteBuffer(dimBuf, CL_TRUE, 0, sizeof(int) * (rank + 2), &tempDimentions[0]);
}

string GPUDimentions::describe(){
	string ans = "size: " + to_string(size) + "\nrank: " + to_string(rank) + "\n" + "{";
	for (int i = 0; i < rank; i++){
		ans += to_string(dimentions[i]);
		if (i != rank - 1){
			ans += ", ";
		}
	}
	ans += "}";
	return ans;
}

Node::Node(){
	outCount = 0;
	getCount = 0;
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

void Node::clean(){
	if (getCount != 0){
		getCount = (getCount + 1) % outCount;
		return;
	}
	seedDims = GPUDimentions();
	resultDims = GPUDimentions();
	outDims = {};
	out = {};

	for (int i = 0; i < inputs.size(); i++){
		inputs[i]->clean();
	}
	getCount = (getCount + 1) % outCount;
}

void Node::seedDimAdd(GPUDimentions* tempSeed){
	if (tempSeed->rank > seedDims.rank){
		if (tempSeed->rank > resultDims.rank){
			seedDims.rank = resultDims.rank;
			seedDims.size = resultDims.size;
			seedDims.dimentions = resultDims.dimentions;
		}
		else{
			seedDims.rank = tempSeed->rank;
			seedDims.size = tempSeed->size;
			seedDims.dimentions = tempSeed->dimentions;
		}
	}


	if (getCount == 0){
		seedDims.setBuf();
		seed = cl::Buffer(context, CL_MEM_READ_WRITE, sizeof(float) * seedDims.size);
	}
}

GPUDimentions Node::getMaxDimentions(vector<GPUDimentions*> dimentionSet){
	int maxPos = 0;
	int maxRank = dimentionSet[0]->rank;
	for (int i = 1; i < dimentionSet.size(); i++){
		if (dimentionSet[i]->rank > maxRank){
			maxRank = dimentionSet[i]->rank;
			maxPos = i;
		}
	}
	return *dimentionSet[maxPos];
}

Constant::Constant(NumObject val, string placeHolder){
	name = placeHolder;
	value = val;
}

void Constant::getValue(){
	return;
}

void Constant::getDimentions(){
	if(getCount != 0){
		getCount = (getCount + 1) % outCount;
		return;
	}

	resultDims.rank = value.rank;
	resultDims.size = value.size;
	resultDims.dimentions = value.dimentions;
	resultDims.setBuf();

	result = cl::Buffer(context, CL_MEM_READ_WRITE, sizeof(float) * resultDims.size);
	queue.enqueueWriteBuffer(result, CL_TRUE, 0, sizeof(float) * resultDims.size, &value.values[0]);
	getCount = (getCount + 1) % outCount;
}

void Constant::deriveDimentions(GPUDimentions* tempSeed){
	return;
}

void Constant::updateHostVals(){
	queue.enqueueReadBuffer(result, CL_TRUE, 0, sizeof(float) * resultDims.size, &value.values[0]);
}

string Constant::describe(){
	if (name == ""){
		return value.describe();
	}
	return name;
}

void Constant::derive(){
	return;
}

void Variable::deriveDimentions(GPUDimentions* tempSeed){
	getCount = (getCount + 1) % outCount;
	seedDimAdd(tempSeed);
}

BasicOperator::BasicOperator(Node* a, Node* b){
	inputs.push_back(a);
	inputs.push_back(b);
	a->outputs.push_back(this);
	b->outputs.push_back(this);
	a->outCount += 1;
	b->outCount += 1;
}

void BasicOperator::getDimentions(){
	if(getCount != 0){
		getCount = (getCount + 1) % outCount;
		return;
	}

	inputs[0]->getDimentions();
	inputs[1]->getDimentions();

	resultDims.rank = max(inputs[0]->resultDims.rank, inputs[1]->resultDims.rank);
	resultDims.size = max(inputs[0]->resultDims.size, inputs[1]->resultDims.size);
	if (inputs[0]->resultDims.rank > inputs[1]->resultDims.rank){
		resultDims.dimentions = inputs[0]->resultDims.dimentions;
	}
	else{
		resultDims.dimentions = inputs[1]->resultDims.dimentions;
	}

	resultDims.setBuf();

	result = cl::Buffer(context, CL_MEM_READ_WRITE, sizeof(float) * resultDims.size);
	getCount = (getCount + 1) % outCount;
}

string BasicOperator::describe(){
	return "(" + inputs[0]->describe() + " " + name + " " + inputs[1]->describe() + ")";
}

BasicFunction::BasicFunction(Node* a){
	inputs.push_back(a);
	a->outputs.push_back(this);
	a->outCount += 1;
}

void BasicFunction::getDimentions(){
	if(getCount != 0){
		getCount = (getCount + 1) % outCount;
		return;
	}

	inputs[0]->getDimentions();

	resultDims.rank = inputs[0]->resultDims.rank;
	resultDims.size = inputs[0]->resultDims.size;
	resultDims.dimentions = inputs[0]->resultDims.dimentions;

	resultDims.setBuf();

	result = cl::Buffer(context, CL_MEM_READ_WRITE, sizeof(float) * resultDims.size);
	getCount = (getCount + 1) % outCount;
}

void BasicFunction::deriveDimentions(GPUDimentions* tempSeed){
	getCount = (getCount + 1) % outCount;
	seedDimAdd(tempSeed);

	if (getCount == 0){
		outDims.push_back(inputs[0]->resultDims);
		out.push_back(cl::Buffer(context, CL_MEM_READ_WRITE, sizeof(float) * outDims[0].size));
		inputs[0]->deriveDimentions(&outDims[0]);
	}
}




