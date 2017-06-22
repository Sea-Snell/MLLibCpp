#ifndef COSTFUNCTIONS_H
#define COSTFUNCTIONS_H
#include "Node.hpp"

class MeanSquared: public Node{
public:
	vector<double> differenceMemo;
	int dimention;

	vector<int> ansDimentions;
	int ansRank;
	int ansSize;
	vector<double> ans;

	double average;

	MeanSquared(Node* hypothesis, Node* y, int dimentionVal = 0);
	void getValue();
	void getValueDimentions();
	void derive(vector<double>& seed);
	void deriveDimentions(vector<int>& seedDimentionsVal);
	string describe();
};


class CrossEntropy: public Node{
public:
	int dimention;

	vector<int> ansDimentions;
	int ansRank;
	int ansSize;
	vector<double> ans;

	double average;

	CrossEntropy(Node* hypothesis, Node* y, int dimentionVal = 0);
	void getValue();
	void getValueDimentions();
	void derive(vector<double>& seed);
	void deriveDimentions(vector<int>& seedDimentionsVal);
	string describe();
};

class CrossEntropySoftmax: public Node{
public:
	int dimention;
	int meanDimention;
	vector<double> softmaxMemo;

	vector<int> newDimentions;
	vector<double> maxVals;
	int newSize;

	int preSum;
	int postSum;

	vector<double> temp;
	vector<double> temp2;
	vector<double> ans;

	vector<int> ansDimentions2;
	int ansRank2;
	int ansSize2;
	vector<double> ans2;

	double average;

	CrossEntropySoftmax(Node* hypothesis, Node* y, int dimentionVal = -1, int meanDimentionVal = 0);
	void getValue();
	void getValueDimentions();
	void derive(vector<double>& seed);
	void deriveDimentions(vector<int>& seedDimentionsVal);
	string describe();
};

class KL: public CrossEntropy{
public:
	KL(Node* hypothesis, Node* y, int dimentionVal = 0): CrossEntropy(hypothesis, y, dimentionVal){name = "KL";};
	void getValue();
	void derive(vector<double>& seed);
};

#endif