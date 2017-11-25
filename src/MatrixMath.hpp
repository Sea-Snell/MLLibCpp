#ifndef MATRIXMATH_H
#define MATRIXMATH_H
#include "Node.hpp"

class MatMul: public Node{
public:
	MatMul(Node* a, Node* b);
	NumObject getValue(int t = 0, int tf = 0);
	void derive(NumObject& seed, int t = 0, int tf = 0);
};


class Sum: public Node{
public:
	int dimention;

	Sum(Node* a, int dimentionVal = 0);
	NumObject getValue(int t = 0, int tf = 0);
	void derive(NumObject& seed, int t = 0, int tf = 0);
	string describe();
};

class Mean: public Node{
public:
	int dimention;

	Mean(Node* a, int dimentionVal = 0);
	NumObject getValue(int t = 0, int tf = 0);
	double operation(vector<double>& a);
	void derive(NumObject& seed, int t = 0, int tf = 0);
	string describe();
};

NumObject expandDerivative(NumObject& a, int dimention, int size);


class Trans: public Node{
public:
	vector<int> perm;

	Trans(Node* a, vector<int> permutations);
	NumObject getValue(int t = 0, int tf = 0);
	int flipIdx(int num, NumObject& a, NumObject& b);
	int flipIdxDerive(int num, NumObject& a, NumObject& b);
	void derive(NumObject& seed, int t = 0, int tf = 0);
	string describe();
};

class Max: public Node{
public:
	int dimention;
	vector<NumObject> idx;

	Max(Node* a, int dimentionVal = 0);
	NumObject getValue(int t = 0, int tf = 0);
	void derive(NumObject& seed, int t = 0, int tf = 0);
	string describe();
};

class Min: public Max{
public:
	Min(Node* a, int dimentionVal = 0): Max(a, dimentionVal){name = "Min";}
	NumObject getValue(int t = 0, int tf = 0);
};

#endif