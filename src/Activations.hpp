#ifndef ACTIVATIONS_H
#define ACTIVATIONS_H
#include "Node.hpp"

class Sigmoid: public BasicFunction{
public:
	Sigmoid(Node* a): BasicFunction(a){name = "Sigmoid";}

	void getValue();
	void derive();
};

class ReLU: public BasicFunction{
public:
	ReLU(Node* a): BasicFunction(a){name = "ReLU";}

	void getValue();
	void derive();
};

class LeakyReLU: public BasicFunction{
public:
	LeakyReLU(Node* a): BasicFunction(a){name = "LeakyReLU";}

	void getValue();
	void derive();
};

class Gaussian: public BasicFunction{
public:
	Gaussian(Node* a): BasicFunction(a){name = "Gaussian";}

	void getValue();
	void derive();
};

class Softmax: public Node{
public:
	int dimention;
	int preSum;
	int GROUP_SIZE;
	int globalSize;
	int blocksWide;
	cl::Buffer resedue;
	cl::Buffer resultResedue;
	
	Softmax(Node* a, int dimentionVal = -1);
	
	void getDimentions();
	void getValue();
	void deriveDimentions(GPUDimentions* tempSeed);
	void derive();
	string describe();
};

class TanH: public BasicFunction{
public:
	TanH(Node* a): BasicFunction(a){name = "TanH";}

	void getValue();
	void derive();
};

class Softsign: public BasicFunction{
public:
	Softsign(Node* a): BasicFunction(a){name = "Softsign";}

	void getValue();
	void derive();
};

#endif