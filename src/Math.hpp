#ifndef MATH_H
#define MATH_H
#include "Node.hpp"
#include <math.h>

using namespace std;

class Add: public BasicOperator{
public:
	Add(Node* a, Node* b): BasicOperator(a, b){name = "+";}

	double operation(vector<double>& a);
	void derive(NumObject& seed, int t = 0, int tf = 0);
};

class Subtract: public BasicOperator{
public:
	Subtract(Node* a, Node* b): BasicOperator(a, b){name = "-";}

	double operation(vector<double>& a);
	void derive(NumObject& seed, int t = 0, int tf = 0);
	double deriveOperation2(vector<double>& a);
};


class Multiply: public BasicOperator{
public:
	Multiply(Node* a, Node* b): BasicOperator(a, b){name = "*";}

	double operation(vector<double>& a);
	void derive(NumObject& seed, int t = 0, int tf = 0);
	double deriveOperation1(vector<double>& a);
};


class Divide: public BasicOperator{
public:
	Divide(Node* a, Node* b): BasicOperator(a, b){name = "/";}

	double operation(vector<double>& a);
	void derive(NumObject& seed, int t = 0, int tf = 0);
	double deriveOperation1(vector<double>& a);
	double deriveOperation2(vector<double>& a);
};

class Pow: public BasicOperator{
public:
	Pow(Node* a, Node* b): BasicOperator(a, b){name = "^";}

	double operation(vector<double>& a);
	void derive(NumObject& seed, int t = 0, int tf = 0);
	double deriveOperation1(vector<double>& a);
	double deriveOperation2(vector<double>& a);
};

class Ln: public BasicFunction{
public:
	Ln(Node* a): BasicFunction(a){name = "Ln";}

	double operation(vector<double>& a);
	void derive(NumObject& seed, int t = 0, int tf = 0);
	double deriveOperation1(vector<double>& a);
};

class Exp: public BasicFunction{
public:
	Exp(Node* a): BasicFunction(a){name = "Exp";}

	double operation(vector<double>& a);
	void derive(NumObject& seed, int t = 0, int tf = 0);
	double deriveOperation1(vector<double>& a);
};

class Log: public BasicFunction{
public:
	double base;
	Log(Node* a, double baseVal = 10);
	double operation(vector<double>& a);
	void derive(NumObject& seed, int t = 0, int tf = 0);
	double deriveOperation1(vector<double>& a);
	string describe();
};

class Sin: public BasicFunction{
public:
	Sin(Node* a): BasicFunction(a){name = "Sin";}

	double operation(vector<double>& a);
	void derive(NumObject& seed, int t = 0, int tf = 0);
	double deriveOperation1(vector<double>& a);
};

class Cos: public BasicFunction{
public:
	Cos(Node* a): BasicFunction(a){name = "Cos";}

	double operation(vector<double>& a);
	void derive(NumObject& seed, int t = 0, int tf = 0);
	double deriveOperation1(vector<double>& a);
};

class Tan: public BasicFunction{
public:
	Tan(Node* a): BasicFunction(a){name = "Tan";}

	double operation(vector<double>& a);
	void derive(NumObject& seed, int t = 0, int tf = 0);
	double deriveOperation1(vector<double>& a);
};

class ArcSin: public BasicFunction{
public:
	ArcSin(Node* a): BasicFunction(a){name = "ArcSin";}

	double operation(vector<double>& a);
	void derive(NumObject& seed, int t = 0, int tf = 0);
	double deriveOperation1(vector<double>& a);
};

class ArcCos: public BasicFunction{
public:
	ArcCos(Node* a): BasicFunction(a){name = "ArcCos";}

	double operation(vector<double>& a);
	void derive(NumObject& seed, int t = 0, int tf = 0);
	double deriveOperation1(vector<double>& a);
};

class ArcTan: public BasicFunction{
public:
	ArcTan(Node* a): BasicFunction(a){name = "ArcTan";}

	double operation(vector<double>& a);
	void derive(NumObject& seed, int t = 0, int tf = 0);
	double deriveOperation1(vector<double>& a);
};


#endif