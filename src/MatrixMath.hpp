#ifndef MATRIXMATH_H
#define MATRIXMATH_H
#include "Node.hpp"

class MatMul: public Node{
public:
	MatMul(Node* a, Node* b);

	void getDimentions();
	void getValue();
	void deriveDimentions(GPUDimentions* tempSeed);
	void derive();
};


class Sum: public Node{
public:
	int dimention;
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


class Trans: public Node{
public:
	Trans(Node* a);

	void getDimentions();
	void getValue();
	void deriveDimentions(GPUDimentions* tempSeed);
	void derive();
};

class Max: public Node{
public:
	int dimention;
	int preSum;
	cl::Buffer idx;
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