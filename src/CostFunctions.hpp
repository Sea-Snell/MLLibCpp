#ifndef COSTFUNCTIONS_H
#define COSTFUNCTIONS_H
#include "Node.hpp"

class MeanSquared: public Node{
public:
	cl::Buffer differenceMemo;
	cl::Buffer diffSquared;
	int dimention;
	int preSum;

	MeanSquared(Node* hypothesis, Node* y, int dimentionVal = 0);
	void getDimentions();
	void getValue();
	void deriveDimentions(GPUDimentions* tempSeed);
	void derive();
	string describe();
};


class CrossEntropy: public Node{
public:
	cl::Buffer crossResult;
	int dimention;
	int preSum;

	CrossEntropy(Node* hypothesis, Node* y, int dimentionVal = 0);
	void getDimentions();
	void getValue();
	void deriveDimentions(GPUDimentions* tempSeed);
	void derive();
	string describe();
};

// class CrossEntropySoftmax: public Node{
// public:
// 	int dimention;
// 	int meanDimention;
// 	vector<NumObject> softmaxMemo;

// 	CrossEntropySoftmax(Node* hypothesis, Node* y, int dimentionVal = -1, int meanDimentionVal = 0);
// 	NumObject getValue(int t = 0, int tf = 0);
// 	double operation1(vector<double>& a);
// 	void derive(NumObject& seed, int t = 0, int tf = 0);
// 	double deriveOperation1(vector<double>& a);
// 	string describe();
// };

#endif