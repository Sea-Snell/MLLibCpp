#ifndef NODE_H
#define NODE_H
#include <vector>
#include <string>
#include <utility>
#include <iostream>
#include <algorithm>
#include <stdarg.h>
#include <math.h>
#include <stdlib.h>
#include <cmath>
#include "openCL.hpp"
using namespace std;

extern cl::Context context;
extern cl::CommandQueue queue;
extern cl::Program program;

extern cl::make_kernel<cl::Buffer> zeroBuffer;
extern cl::make_kernel<cl::Buffer, cl::Buffer, cl::Buffer, cl::Buffer> reduceSum;
extern cl::make_kernel<cl::Buffer, cl::Buffer, cl::Buffer, cl::Buffer> explodeUp;
extern cl::make_kernel<cl::Buffer, cl::Buffer, cl::Buffer, float> gradientDescentStep;

extern cl::make_kernel<cl::Buffer, cl::Buffer, cl::Buffer, cl::Buffer, cl::Buffer> add;
extern cl::make_kernel<cl::Buffer, cl::Buffer, cl::Buffer, cl::Buffer, cl::Buffer> subtract;
extern cl::make_kernel<cl::Buffer, cl::Buffer, cl::Buffer, cl::Buffer, cl::Buffer> multiply;
extern cl::make_kernel<cl::Buffer, cl::Buffer, cl::Buffer, cl::Buffer, cl::Buffer> divide;
extern cl::make_kernel<cl::Buffer, cl::Buffer, cl::Buffer, cl::Buffer, cl::Buffer> pow_;
extern cl::make_kernel<cl::Buffer, cl::Buffer> ln;
extern cl::make_kernel<cl::Buffer, cl::Buffer> exp_;
extern cl::make_kernel<cl::Buffer, cl::Buffer, float> log_;
extern cl::make_kernel<cl::Buffer, cl::Buffer> sin_;
extern cl::make_kernel<cl::Buffer, cl::Buffer> cos_;
extern cl::make_kernel<cl::Buffer, cl::Buffer> tan_;
extern cl::make_kernel<cl::Buffer, cl::Buffer> asin_;
extern cl::make_kernel<cl::Buffer, cl::Buffer> acos_;
extern cl::make_kernel<cl::Buffer, cl::Buffer> atan_;
extern cl::make_kernel<cl::Buffer, cl::Buffer, cl::Buffer, cl::Buffer, cl::Buffer> matMul2x2;
extern cl::make_kernel<cl::Buffer, cl::Buffer, cl::Buffer, cl::Buffer, cl::Buffer> matMul2x1;
extern cl::make_kernel<cl::Buffer, cl::Buffer, cl::Buffer, cl::Buffer, cl::Buffer> matMul1x2;
extern cl::make_kernel<cl::Buffer, cl::Buffer, cl::Buffer, cl::Buffer, cl::Buffer> matMul1x1;
extern cl::make_kernel<cl::Buffer, cl::Buffer, int, int> sum_;
extern cl::make_kernel<cl::Buffer, cl::Buffer, int, int> mean_;
extern cl::make_kernel<cl::Buffer, cl::Buffer, cl::Buffer> trans;
extern cl::make_kernel<cl::Buffer, cl::Buffer, cl::Buffer, int, int> max_;
extern cl::make_kernel<cl::Buffer, cl::Buffer, cl::Buffer, int, int> min_;

extern cl::make_kernel<cl::Buffer, cl::Buffer, cl::Buffer> addDerivative;
extern cl::make_kernel<cl::Buffer, cl::Buffer, cl::Buffer> subtractDerivative1;
extern cl::make_kernel<cl::Buffer, cl::Buffer, cl::Buffer, cl::Buffer, cl::Buffer> multiplyDerivative;
extern cl::make_kernel<cl::Buffer, cl::Buffer, cl::Buffer, cl::Buffer, cl::Buffer> divideDerivative0;
extern cl::make_kernel<cl::Buffer, cl::Buffer, cl::Buffer, cl::Buffer, cl::Buffer, cl::Buffer, cl::Buffer> divideDerivative1;
extern cl::make_kernel<cl::Buffer, cl::Buffer, cl::Buffer, cl::Buffer, cl::Buffer, cl::Buffer, cl::Buffer> powDerivative0;
extern cl::make_kernel<cl::Buffer, cl::Buffer, cl::Buffer, cl::Buffer, cl::Buffer, cl::Buffer, cl::Buffer> powDerivative1;
extern cl::make_kernel<cl::Buffer, cl::Buffer, float, cl::Buffer, cl::Buffer, cl::Buffer> logDerivative;
extern cl::make_kernel<cl::Buffer, cl::Buffer, cl::Buffer, cl::Buffer, cl::Buffer> sinDerivative;
extern cl::make_kernel<cl::Buffer, cl::Buffer, cl::Buffer, cl::Buffer, cl::Buffer> cosDerivative;
extern cl::make_kernel<cl::Buffer, cl::Buffer, cl::Buffer, cl::Buffer, cl::Buffer> tanDerivative;
extern cl::make_kernel<cl::Buffer, cl::Buffer, cl::Buffer, cl::Buffer, cl::Buffer> asinDerivative;
extern cl::make_kernel<cl::Buffer, cl::Buffer, cl::Buffer, cl::Buffer, cl::Buffer> acosDerivative;
extern cl::make_kernel<cl::Buffer, cl::Buffer, cl::Buffer, cl::Buffer, cl::Buffer> atanDerivative;
extern cl::make_kernel<cl::Buffer, cl::Buffer, cl::Buffer, cl::Buffer, cl::Buffer> matMul2x2Derivative0;
extern cl::make_kernel<cl::Buffer, cl::Buffer, cl::Buffer, cl::Buffer, cl::Buffer, cl::Buffer> matMul2x2Derivative1;
extern cl::make_kernel<cl::Buffer, cl::Buffer, cl::Buffer, cl::Buffer, cl::Buffer> matMul2x1Derivative0;
extern cl::make_kernel<cl::Buffer, cl::Buffer, cl::Buffer, cl::Buffer, cl::Buffer> matMul2x1Derivative1;
extern cl::make_kernel<cl::Buffer, cl::Buffer, cl::Buffer, cl::Buffer, cl::Buffer> matMul1x2Derivative0;
extern cl::make_kernel<cl::Buffer, cl::Buffer, cl::Buffer, cl::Buffer, cl::Buffer> matMul1x2Derivative1;
extern cl::make_kernel<cl::Buffer, cl::Buffer, cl::Buffer> matMul1x1Derivative0;
extern cl::make_kernel<cl::Buffer, cl::Buffer, cl::Buffer> matMul1x1Derivative1;
extern cl::make_kernel<cl::Buffer, cl::Buffer, cl::Buffer, int, int> sumDerivative;
extern cl::make_kernel<cl::Buffer, cl::Buffer, cl::Buffer, int, int> meanDerivative;
extern cl::make_kernel<cl::Buffer, cl::Buffer, int> meanDerivativeSmallSeed;
extern cl::make_kernel<cl::Buffer, cl::Buffer, cl::Buffer> transDerivative1;
extern cl::make_kernel<cl::Buffer, cl::Buffer, cl::Buffer> transDerivative2;
extern cl::make_kernel<cl::Buffer, cl::Buffer, cl::Buffer, cl::Buffer, int, int> maxDerivative;
extern cl::make_kernel<cl::Buffer, cl::Buffer, cl::Buffer, cl::Buffer> maxDerivativeSmallSeed;

void initialize();

class NumObject{
public:
	int rank;
	int size;
	vector<int> dimentions;
	vector<float> values;

	NumObject();
	NumObject(float val);
	NumObject(vector<float> val, vector<int> dimentionsList);
	NumObject(vector<int> dimentionsList, float fill);
	NumObject(vector<int> dimentionsList);

	string describe();
};

class GPUDimentions{
public:
	int rank;
	int size;
	vector<int> dimentions;
	cl::Buffer dimBuf;

	GPUDimentions();
	GPUDimentions(int rankVal, int sizeVal, vector<int> dimentionsVals);
	void setBuf();
};

class Node{
public:
	vector<Node*> inputs;
	vector<Node*> outputs;
	string name;

	GPUDimentions resultDims;
	cl::Buffer result;

	GPUDimentions seedDims;
	cl::Buffer seed;

	vector<GPUDimentions> outDims;
	vector<cl::Buffer> out;

	int outCount;
	int getCount;

	Node();
	virtual void getValue() = 0;
	virtual void getDimentions() = 0;
	virtual void deriveDimentions(GPUDimentions* tempSeed) = 0;
	virtual string describe();
	void clean();

	void seedDimAdd(GPUDimentions* tempSeed);
	GPUDimentions getMaxDimentions(vector<GPUDimentions*> dimentionSet);

	virtual void derive() = 0;

	// Node* operator+(Node& param);
	// Node* operator-(Node& param);
	// Node* operator/(Node& param);
	// Node* operator*(Node& param);
};

class Constant: public Node{
public:
	NumObject value;
	Constant(NumObject val, string placeHolder = "");
	void getValue();
	void getDimentions();
	void deriveDimentions(GPUDimentions* tempSeed);
	void derive();
	void updateHostVals();
	string describe();
};

class Variable: public Constant{
public:
	Variable(NumObject val, string placeHolder = ""): Constant(val, placeHolder){}
	void deriveDimentions(GPUDimentions* tempSeed);
};

class BasicOperator: public Node{
public:
	BasicOperator(Node* a, Node* b);
	void getDimentions();
	virtual string describe();
};

class BasicFunction: public Node{
public:
	BasicFunction(Node* a);
	void getDimentions();
};



#endif
