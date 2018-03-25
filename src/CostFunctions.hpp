#ifndef COSTFUNCTIONS_H
#define COSTFUNCTIONS_H
#include "Node.hpp"

class MeanSquared: public Node{
public:
	cl::Buffer differenceMemo;
	cl::Buffer diffSquaredResedue;
	int dimention;
	int GROUP_SIZE;

	MeanSquared(Node* hypothesis, Node* y, int dimentionVal = 0);
	void getDimentions();
	void getValue();
	void deriveDimentions(GPUDimentions* tempSeed);
	void derive();
	string describe();
};


class CrossEntropy: public Node{
public:
	cl::Buffer crossResultResedue;
	int dimention;
	int GROUP_SIZE;

	CrossEntropy(Node* hypothesis, Node* y, int dimentionVal = 0);
	void getDimentions();
	void getValue();
	void deriveDimentions(GPUDimentions* tempSeed);
	void derive();
	string describe();
};

class CrossEntropySoftmax: public Node{
public:
	cl::Buffer softmaxMemo;
	cl::Buffer resedue;
	int dimention;
	int meanDimention;
	int GROUP_SIZE;
	int preSum;
	int globalSize;
	int blocksWide;

	CrossEntropySoftmax(Node* hypothesis, Node* y, int dimentionVal = -1, int meanDimentionVal = 0);
	void getDimentions();
	void getValue();
	void deriveDimentions(GPUDimentions* tempSeed);
	void derive();
	string describe();
};

#endif