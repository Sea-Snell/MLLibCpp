#ifndef MATRIXMATH_H
#define MATRIXMATH_H
#include "Node.hpp"

class MatMul: public Node{
public:
	int dimentionA;
	int dimentionB;

	vector<int> newDimentions;
	int lowIdx;
	int highIdx;
	int adjustedDimentionA;
	int bNormalizeProdct;

	int firstProduct = 1;
	int secondProduct = 1;
	int thirdProduct = 1;

	int prodA;
	int prodB;
	int prodC;
	int prodD;

	int prodE;
	int prodF;
	int prodG;

	int prodH;
	int prodI;
	int prodJ;

	vector<int> newDimentions1;
	vector<int> tempBDimentions1;

	int bNormalizeProdct1;

	int firstProduct1;
	int secondProduct1;
	int thirdProduct1;

	int prodA1;
	int prodB1;
	int prodC1;
	int prodD1;

	int prodE1;
	int prodF1;
	int prodG1;

	int prodH1;
	int prodI1;
	int prodJ1;

	vector<double> ans1;


	vector<int> newDimentions2;
	vector<int> tempADimentions2;

	int bNormalizeProdct2;

	int firstProduct2;
	int secondProduct2;
	int thirdProduct2;

	int prodA2;
	int prodB2;
	int prodC2;
	int prodD2;

	int prodE2;
	int prodF2;
	int prodG2;

	int prodH2;
	int prodI2;
	int prodJ2;

	vector<double> ans2;

	MatMul(Node* a, Node* b, int dimentionAVal, int dimentionBVal);

	void getValue();
	void getValueDimentions();
	vector<vector<int>> getDimentions(vector<int> a, vector<int>, int adjustedDimentionA);
	void derive(vector<double>& seed);
	void deriveDimentions(vector<int>& seedDimentionsVal);

	string describe();
};


class Sum: public Node{
public:
	int dimention;

	int preSum;
	int postSum;

	int preSum1;
	int postSum1;

	int adjustedDimention;

	vector<int> ansDimentions;
	int ansRank;
	int ansSize;
	vector<double> ans;

	Sum(Node* a, int dimentionVal = 0);
	void getValue();
	void getValueDimentions();
	void derive(vector<double>& seed);
	void deriveDimentions(vector<int>& seedDimentionsVal);
	string describe();
};

class Mean: public Sum{
public:

	vector<double> temp;

	Mean(Node* a, int dimentionVal = 0): Sum(a, dimentionVal){name = "Mean";}
	void getValue();
	void derive(vector<double>& seed);
	void deriveDimentions(vector<int>& seedDimentionsVal);
	string describe();
};

class Trans: public Node{
public:
	vector<int> perm;

	vector<int> idx;

	bool isInternalized;

	vector<int> ansDimentions;
	int ansRank;
	int ansSize;
	vector<double> ans;

	vector<int> idx1;
	vector<int> newIdx1;

	Trans(Node* a, vector<int> permutations);
	void getValue();
	void getValueDimentions();
	void derive(vector<double>& seed);
	void deriveDimentions(vector<int>& seedDimentionsVal);
	string describe();
};

class Max: public Node{
public:
	int dimention;
	vector<int> idx;

	int preSum;
	int postSum;

	vector<double> ans;
	vector<int> ansDimentions;
	int ansRank;
	int ansSize;

	int adjustedDimention;

	vector<double> temp1;
	vector<int> tempDimentions1;
	int tempRank1;
	int tempSize1;

	int preSum1;
	int postSum1;

	Max(Node* a, int dimentionVal = 0);
	void getValue();
	void getValueDimentions();
	void derive(vector<double>& seed);
	void deriveDimentions(vector<int>& seedDimentionsVal);
	string describe();
};

class Min: public Max{
public:
	Min(Node* a, int dimentionVal = 0): Max(a, dimentionVal){name = "Min";}
	void getValue();
	void getValueDimentions();
};

#endif