#ifndef COSTFUNCTIONS_H
#define COSTFUNCTIONS_H
#include "Node.hpp"

class CostFunction: public Node{
public:
	vector<cl::Buffer> resedue;
	int dimention;
	int GROUP_SIZE;
	int globalSize;
	int numGroups;

	CostFunction(Node* hypothesis, Node* y, int dimentionVal);
	void getDimentions();
	void deriveDimentions(GPUDimentions* tempSeed);
	string describe();
};

class MeanSquared: public CostFunction{
public:
	vector<cl::Buffer> differenceMemo;

	MeanSquared(Node* hypothesis, Node* y, int dimentionVal = 0): CostFunction(hypothesis, y, dimentionVal){name = "MeanSquared";}
	void getDimentions();
	void getValue();
	void derive();
};


class CrossEntropy: public CostFunction{
public:

	CrossEntropy(Node* hypothesis, Node* y, int dimentionVal = 0): CostFunction(hypothesis, y, dimentionVal){name = "CrossEntropy";}
	void getValue();
	void derive();
};

class CrossEntropySoftmax: public CostFunction{
public:
	vector<cl::Buffer> softmaxMemo;
	vector<cl::Buffer> resultResedue;
	int meanDimention;
	int preSum;
	int blocksWide;

	CrossEntropySoftmax(Node* hypothesis, Node* y, int dimentionVal = -1, int meanDimentionVal = 0);
	void getDimentions();
	void getValue();
	void derive();
	string describe();
};

#endif