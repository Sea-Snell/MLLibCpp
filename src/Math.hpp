#ifndef MATH_H
#define MATH_H
#include "Node.hpp"
#include <math.h>

using namespace std;

class Add: public BasicOperator{
public:
	Add(Node* a, Node* b): BasicOperator(a, b){name = "+";}

	void getValue();
	void deriveDimentions(GPUDimentions* tempSeed);
	void derive();
};

class Subtract: public BasicOperator{
public:
	Subtract(Node* a, Node* b): BasicOperator(a, b){name = "-";}

	void getValue();
	void deriveDimentions(GPUDimentions* tempSeed);
	void derive();
};


class Multiply: public BasicOperator{
public:
	Multiply(Node* a, Node* b): BasicOperator(a, b){name = "*";}

	void getValue();
	void deriveDimentions(GPUDimentions* tempSeed);
	void derive();
};


class Divide: public BasicOperator{
public:
	Divide(Node* a, Node* b): BasicOperator(a, b){name = "/";}

	void getValue();
	void deriveDimentions(GPUDimentions* tempSeed);
	void derive();
};

class Pow: public BasicOperator{
public:
	Pow(Node* a, Node* b): BasicOperator(a, b){name = "^";}

	void getValue();
	void deriveDimentions(GPUDimentions* tempSeed);
	void derive();
};

class Ln: public BasicFunction{
public:
	Ln(Node* a): BasicFunction(a){name = "Ln";}

	void getValue();
	void deriveDimentions(GPUDimentions* tempSeed);
	void derive();
};

class Exp: public BasicFunction{
public:
	Exp(Node* a): BasicFunction(a){name = "Exp";}

	void getValue();
	void deriveDimentions(GPUDimentions* tempSeed);
	void derive();
};

class Log: public BasicFunction{
public:
	float base;
	Log(Node* a, float baseVal = 10.0);
	void getValue();
	void deriveDimentions(GPUDimentions* tempSeed);
	void derive();
	string describe();
};

class Sin: public BasicFunction{
public:
	Sin(Node* a): BasicFunction(a){name = "Sin";}
	
	void getValue();
	void deriveDimentions(GPUDimentions* tempSeed);
	void derive();
};

class Cos: public BasicFunction{
public:
	Cos(Node* a): BasicFunction(a){name = "Cos";}

	void getValue();
	void deriveDimentions(GPUDimentions* tempSeed);
	void derive();
};

class Tan: public BasicFunction{
public:
	Tan(Node* a): BasicFunction(a){name = "Tan";}

	void getValue();
	void deriveDimentions(GPUDimentions* tempSeed);
	void derive();
};

class ArcSin: public BasicFunction{
public:
	ArcSin(Node* a): BasicFunction(a){name = "ArcSin";}

	void getValue();
	void deriveDimentions(GPUDimentions* tempSeed);
	void derive();
};

class ArcCos: public BasicFunction{
public:
	ArcCos(Node* a): BasicFunction(a){name = "ArcCos";}

	void getValue();
	void deriveDimentions(GPUDimentions* tempSeed);
	void derive();
};

class ArcTan: public BasicFunction{
public:
	ArcTan(Node* a): BasicFunction(a){name = "ArcTan";}

	void getValue();
	void deriveDimentions(GPUDimentions* tempSeed);
	void derive();
};


#endif