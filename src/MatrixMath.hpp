#ifndef MATRIXMATH_H
#define MATRIXMATH_H
#include "Node.hpp"

class MatMul: public Node{
public:
	int dimentionA;
	int dimentionB;
	vector<int> newDimentionsMemo;
	int adjustedDimentionAMemo;

	MatMul(Node* a, Node* b, int dimentionAVal, int dimentionBVal);
	NumObject getValue();
	vector<vector<int>> getDimentions(vector<int> a, vector<int>, int adjustedDimentionA);
	void derive(NumObject& seed);
	string describe();
};


class Sum: public Node{
public:
	int dimention;

	Sum(Node* a, int dimentionVal = 0);
	NumObject getValue();
	void derive(NumObject& seed);
	string describe();
};

class Mean: public Node{
public:
	int dimention;

	Mean(Node* a, int dimentionVal = 0);
	NumObject getValue();
	double operation(vector<double>& a);
	void derive(NumObject& seed);
	string describe();
};

NumObject expandDerivative(NumObject& a, int dimention, int size);


class Trans: public Node{
public:
	vector<int> perm;

	Trans(Node* a, vector<int> permutations);
	NumObject getValue();
	int flipIdx(int num, NumObject& a, NumObject& b);
	int flipIdxDerive(int num, NumObject& a, NumObject& b);
	void derive(NumObject& seed);
	string describe();
};

class Max: public Node{
public:
	int dimention;
	NumObject idx;

	Max(Node* a, int dimentionVal = 0);
	NumObject getValue();
	void derive(NumObject& seed);
	string describe();
};

class Min: public Max{
public:
	Min(Node* a, int dimentionVal = 0): Max(a, dimentionVal){name = "Min";}
	NumObject getValue();
};

#endif