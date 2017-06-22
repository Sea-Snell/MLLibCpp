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
using namespace std;

class Node{
public:
	vector<Node*> inputs;
	string name;
	vector<double> derivativeMemo;
	vector<int> outDimentions;
	int outRank;
	int outSize;
	vector<int> seedDimentions;
	int seedRank;
	int seedSize;


	virtual void getValue() = 0;
	virtual string describe();
	virtual void derive(vector<double>& seed) = 0;
	virtual void getValueDimentions() = 0;
	virtual void deriveDimentions(vector<int>& seedDimentionsVal) = 0;
};

class Constant: public Node{
public:
	Constant(vector<double> val, vector<int> dimentionsVal, string placeHolder = "");
	Constant(double val, string placeHolder = "");
	Constant(const Constant &a);
	void getValue();
	string describe();
	void derive(vector<double>& seed);
	void getValueDimentions();
	void deriveDimentions(vector<int>& seedDimentionsVal);
};

class Variable: public Constant{
public:
	vector<double> derivative;
	vector<int> seedSizes;
	int location;

	Variable(vector<double> val, vector<int> dimentions, string placeHolder = "");
	Variable(double val, string placeHolder = "");
	Variable(const Constant &a);
	void getValue();
	void getValueDimentions();
	void derive(vector<double>& seed);
	void deriveDimentions(vector<int>& seedDimentionsVal);
};

class BasicOperator: public Node{
public:
	int tempSize1;
	vector<int> tempDimentions1;
	int tempRank1;
	vector<double> temp1;

	int tempSize2;
	vector<int> tempDimentions2;
	int tempRank2;
	vector<double> temp2;

	int ansSize1;
	int ansRank1;
	vector<int> ansDimentions1;
	vector<double> ans1;

	int ansSize2;
	int ansRank2;
	vector<int> ansDimentions2;
	vector<double> ans2;

	BasicOperator(Node* a, Node* b);
	void getValue();
	virtual double operation(double a, double b) = 0;
	virtual string describe();
	void getValueDimentions();
	void helpDeriveDimentions(vector<vector<int>> dependentDimentions1, vector<vector<int>> dependentDimentions2);
};

class BasicFunction: public Node{
public:
	int tempSize1;
	vector<int> tempDimentions1;
	int tempRank1;
	vector<double> temp1;

	int ansSize1;
	int ansRank1;
	vector<int> ansDimentions1;
	vector<double> ans1;

	BasicFunction(Node* a);
	void getValue();
	virtual double operation(double a) = 0;
	void getValueDimentions();
	void helpDeriveDimentions(vector<vector<int>> dependentDimentions1);
};

vector<int> helpMapDimentions(vector<vector<int>> dimentions);
vector<int> helpReduceDimentions(vector<int> a, vector<int> b);
int getSize(vector<int> dimentions);



#endif

