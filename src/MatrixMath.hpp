#ifndef MATRIXMATH_H
#define MATRIXMATH_H
#include "Node.hpp"

class MatMul: public Node{
public:
	int GROUP_SIZE;
	int globalSize;
	int blockSide;
	int workPerBlockSideSquared;
	int heightSize;
	int widthSize;
	int blocksWide;

	int GROUP_SIZE_DERIVATIVE_0;
	int globalSizeDerivative0;
	int blockSideDerivative0;
	int workPerBlockSideSquaredDerivative0;
	int heightSizeDerivative0;
	int widthSizeDerivative0;
	int blocksWideDerivative0;

	int GROUP_SIZE_DERIVATIVE_1;
	int globalSizeDerivative1;
	int blockSideDerivative1;
	int workPerBlockSideSquaredDerivative1;
	int heightSizeDerivative1;
	int widthSizeDerivative1;
	int blocksWideDerivative1;
	MatMul(Node* a, Node* b);

	void getDimentions();
	void getValue();
	void deriveDimentions(GPUDimentions* tempSeed);
	void derive();
};

class Trans: public Node{
public:
	Trans(Node* a);

	void getDimentions();
	void getValue();
	void deriveDimentions(GPUDimentions* tempSeed);
	void derive();
};

class Sum: public Node{
public:
	vector<cl::Buffer> resedue;
	int dimention;
	int GROUP_SIZE;
	int globalSize;
	int blocksWide;
	int preSum;
	Sum(Node* a, int dimentionVal = 0);

	void getDimentions();
	void getValue();
	void deriveDimentions(GPUDimentions* tempSeed);
	void derive();
	string describe();
};

class Mean: public Sum{
public:
	Mean(Node* a, int dimentionVal = 0): Sum(a, dimentionVal){name = "Mean";}

	void getValue();
	void derive();
};

class Max: public Node{
public:
	int dimention;
	int preSum;
	vector<cl::Buffer> idx;
	Max(Node* a, int dimentionVal = 0);

	void getDimentions();
	void getValue();
	void deriveDimentions(GPUDimentions* tempSeed);
	void derive();
	string describe();
};

class Min: public Max{
public:
	Min(Node* a, int dimentionVal = 0): Max(a, dimentionVal){name = "Min";}
	void getValue();
};

#endif