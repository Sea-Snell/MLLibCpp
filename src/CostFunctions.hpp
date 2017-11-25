#ifndef COSTFUNCTIONS_H
#define COSTFUNCTIONS_H
#include "Node.hpp"

class MeanSquared: public Node{
public:
	vector<NumObject> differenceMemo;
	int dimention;

	MeanSquared(Node* hypothesis, Node* y, int dimentionVal = 0);
	NumObject getValue(int t = 0, int tf = 0);
	double differenceOperation(vector<double>& a);
	double powerOperation(vector<double>& a);
	void derive(NumObject& seed, int t = 0, int tf = 0);
	double deriveOperation1(vector<double>& a);
	string describe();
};


class CrossEntropy: public Node{
public:
	int dimention;

	CrossEntropy(Node* hypothesis, Node* y, int dimentionVal = 0);
	NumObject getValue(int t = 0, int tf = 0);
	double operation(vector<double>& a);
	void derive(NumObject& seed, int t = 0, int tf = 0);
	double deriveOperation1(vector<double>& a);
	string describe();
};

class CrossEntropySoftmax: public Node{
public:
	int dimention;
	int meanDimention;
	vector<NumObject> softmaxMemo;

	CrossEntropySoftmax(Node* hypothesis, Node* y, int dimentionVal = -1, int meanDimentionVal = 0);
	NumObject getValue(int t = 0, int tf = 0);
	double operation1(vector<double>& a);
	void derive(NumObject& seed, int t = 0, int tf = 0);
	double deriveOperation1(vector<double>& a);
	string describe();
};

#endif