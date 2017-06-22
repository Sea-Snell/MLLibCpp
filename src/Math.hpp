#ifndef MATH_H
#define MATH_H
#include "Node.hpp"
#include <math.h>

using namespace std;

class Add: public BasicOperator{
public:
	Add(Node* a, Node* b): BasicOperator(a, b){name = "+";}

	double operation(double a, double b);
	void derive(vector<double>& seed);
	void deriveDimentions(vector<int>& seedDimentionsVal);
};

class Subtract: public Add{
public:
	Subtract(Node* a, Node* b): Add(a, b){name = "-";}

	double operation(double a, double b);
	void derive(vector<double>& seed);
};


class Multiply: public BasicOperator{
public:
	Multiply(Node* a, Node* b): BasicOperator(a, b){name = "*";}

	double operation(double a, double b);
	void derive(vector<double>& seed);
	void deriveDimentions(vector<int>& seedDimentionsVal);
};


class Divide: public BasicOperator{
public:
	Divide(Node* a, Node* b): BasicOperator(a, b){name = "/";}

	double operation(double a, double b);
	void derive(vector<double>& seed);
	void deriveDimentions(vector<int>& seedDimentionsVal);
};

class Pow: public BasicOperator{
public:
	Pow(Node* a, Node* b): BasicOperator(a, b){name = "^";}

	double operation(double a, double b);
	void derive(vector<double>& seed);
	void deriveDimentions(vector<int>& seedDimentionsVal);
};

class Ln: public BasicFunction{
public:
	Ln(Node* a): BasicFunction(a){name = "Ln";}

	double operation(double a);
	void derive(vector<double>& seed);
	void deriveDimentions(vector<int>& seedDimentionsVal);
};

class Exp: public BasicFunction{
public:
	Exp(Node* a): BasicFunction(a){name = "Exp";}

	double operation(double a);
	void derive(vector<double>& seed);
	void deriveDimentions(vector<int>& seedDimentionsVal);
};

class Log: public BasicFunction{
public:
	double base;
	Log(Node* a, double baseVal = 10);
	double operation(double a);
	void derive(vector<double>& seed);
	string describe();
	void deriveDimentions(vector<int>& seedDimentionsVal);
};

class Sin: public BasicFunction{
public:
	Sin(Node* a): BasicFunction(a){name = "Sin";}

	double operation(double a);
	void derive(vector<double>& seed);
	void deriveDimentions(vector<int>& seedDimentionsVal);
};

class Cos: public Sin{
public:
	Cos(Node* a): Sin(a){name = "Cos";}

	double operation(double a);
	void derive(vector<double>& seed);
};

class Tan: public Sin{
public:
	Tan(Node* a): Sin(a){name = "Tan";}

	double operation(double a);
	void derive(vector<double>& seed);
};

class ArcSin: public Sin{
public:
	ArcSin(Node* a): Sin(a){name = "ArcSin";}

	double operation(double a);
	void derive(vector<double>& seed);
};

class ArcCos: public Sin{
public:
	ArcCos(Node* a): Sin(a){name = "ArcCos";}

	double operation(double a);
	void derive(vector<double>& seed);
};

class ArcTan: public Sin{
public:
	ArcTan(Node* a): Sin(a){name = "ArcTan";}

	double operation(double a);
	void derive(vector<double>& seed);
};


#endif