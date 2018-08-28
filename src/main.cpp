#include "Node.hpp"
#include "Math.hpp"
#include "HelperFunctions.hpp"
#include "MatrixMath.hpp"
#include "Optimizers.hpp"
#include "CostFunctions.hpp"
#include "Activations.hpp"
#include "MNISTLoad.hpp"
#include "TextLoad.hpp"
#include "Regularization.hpp"
#include <time.h>
#include <sys/stat.h>
#include <fstream>

// void rubixNet();
void testWordLSTM();
void wordLSTM();
vector<vector<NumObject>> breakUpData(vector<vector<float>> rawData, int t, int n);
void sinLSTM();
void sinRNN();
void WordRNN();
void charRNN();
void MNISTFFNN();
vector<NumObject> getTrain(int n);
vector<NumObject> getTest(int n);
void linearReg();

int main(){

	srand(time(NULL));

	initialize();

	// linearReg();
	// MNISTFFNN();
	// charRNN();
	// WordRNN();
	// sinRNN();
	// sinLSTM();
	wordLSTM();
	// testWordLSTM();

	return 0;
}

void testWordLSTM(){
	int t = 1;
	int n = 1;
	int charCount;

	vector<char> vocab = vocabFromFolder("../TextData/THE RESERECTION OF CHRIST");

	charCount = (int)(vocab.size());

	int layer1Size = 128;
	int layer2Size = 128;
	int layer3Size = 128;

	Constant&& xData = Constant(NumObject(), "x");
	Constant&& yData = Constant(NumObject(), "y");

	Variable&& h1 = Variable(NumObject(), "h1");
	Variable&& c1 = Variable(NumObject(), "c1");

	Variable&& h2 = Variable(NumObject(), "h2");
	Variable&& c2 = Variable(NumObject(), "c2");

	Variable&& h3 = Variable(NumObject(), "h2");
	Variable&& c3 = Variable(NumObject(), "c2");
	
	Variable&& wf1 = Variable(trunGaussianRandomNums(vector<int>{charCount, layer1Size}, 0.0, 1.0 / sqrt((float)(charCount))), "wf1");
	Variable&& uf1 = Variable(trunGaussianRandomNums(vector<int>{layer1Size, layer1Size}, 0.0, 1.0 / sqrt((float)(layer1Size))), "uf1");
	Variable&& bf1 = Variable(NumObject(vector<int>{layer1Size}, 1.0), "bf1");

	Variable&& wi1 = Variable(trunGaussianRandomNums(vector<int>{charCount, layer1Size}, 0.0, 1.0 / sqrt((float)(charCount))), "wi1");
	Variable&& ui1 = Variable(trunGaussianRandomNums(vector<int>{layer1Size, layer1Size}, 0.0, 1.0 / sqrt((float)(layer1Size))), "ui1");
	Variable&& bi1 = Variable(NumObject(vector<int>{layer1Size}, 0.0), "bi1");

	Variable&& wo1 = Variable(trunGaussianRandomNums(vector<int>{charCount, layer1Size}, 0.0, 1.0 / sqrt((float)(charCount))), "wo1");
	Variable&& uo1 = Variable(trunGaussianRandomNums(vector<int>{layer1Size, layer1Size}, 0.0, 1.0 / sqrt((float)(layer1Size))), "uo1");
	Variable&& bo1 = Variable(NumObject(vector<int>{layer1Size}, 0.0), "bo1");

	Variable&& wc1 = Variable(trunGaussianRandomNums(vector<int>{charCount, layer1Size}, 0.0, 1.0 / sqrt((float)(charCount))), "wc1");
	Variable&& uc1 = Variable(trunGaussianRandomNums(vector<int>{layer1Size, layer1Size}, 0.0, 1.0 / sqrt((float)(layer1Size))), "uc1");
	Variable&& bc1 = Variable(NumObject(vector<int>{layer1Size}, 0.0), "bc1");

	Variable&& wf2 = Variable(trunGaussianRandomNums(vector<int>{layer1Size, layer2Size}, 0.0, 1.0 / sqrt((float)(layer1Size))), "wf2");
	Variable&& uf2 = Variable(trunGaussianRandomNums(vector<int>{layer2Size, layer2Size}, 0.0, 1.0 / sqrt((float)(layer2Size))), "uf2");
	Variable&& bf2 = Variable(NumObject(vector<int>{layer2Size}, 1.0), "bf2");

	Variable&& wi2 = Variable(trunGaussianRandomNums(vector<int>{layer1Size, layer2Size}, 0.0, 1.0 / sqrt((float)(layer1Size))), "wi2");
	Variable&& ui2 = Variable(trunGaussianRandomNums(vector<int>{layer2Size, layer2Size}, 0.0, 1.0 / sqrt((float)(layer2Size))), "ui2");
	Variable&& bi2 = Variable(NumObject(vector<int>{layer2Size}, 0.0), "bi2");

	Variable&& wo2 = Variable(trunGaussianRandomNums(vector<int>{layer1Size, layer2Size}, 0.0, 1.0 / sqrt((float)(layer1Size))), "wo2");
	Variable&& uo2 = Variable(trunGaussianRandomNums(vector<int>{layer2Size, layer2Size}, 0.0, 1.0 / sqrt((float)(layer2Size))), "uo2");
	Variable&& bo2 = Variable(NumObject(vector<int>{layer2Size}, 0.0), "bo2");

	Variable&& wc2 = Variable(trunGaussianRandomNums(vector<int>{layer2Size, layer3Size}, 0.0, 1.0 / sqrt((float)(layer1Size))), "wc2");
	Variable&& uc2 = Variable(trunGaussianRandomNums(vector<int>{layer3Size, layer3Size}, 0.0, 1.0 / sqrt((float)(layer2Size))), "uc2");
	Variable&& bc2 = Variable(NumObject(vector<int>{layer3Size}, 0.0), "bc2");

	Variable&& wf3 = Variable(trunGaussianRandomNums(vector<int>{layer2Size, layer3Size}, 0.0, 1.0 / sqrt((float)(layer2Size))), "wf3");
	Variable&& uf3 = Variable(trunGaussianRandomNums(vector<int>{layer3Size, layer3Size}, 0.0, 1.0 / sqrt((float)(layer3Size))), "uf3");
	Variable&& bf3 = Variable(NumObject(vector<int>{layer3Size}, 1.0), "bf3");

	Variable&& wi3 = Variable(trunGaussianRandomNums(vector<int>{layer2Size, layer3Size}, 0.0, 1.0 / sqrt((float)(layer2Size))), "wi3");
	Variable&& ui3 = Variable(trunGaussianRandomNums(vector<int>{layer3Size, layer3Size}, 0.0, 1.0 / sqrt((float)(layer3Size))), "ui3");
	Variable&& bi3 = Variable(NumObject(vector<int>{layer3Size}, 0.0), "bi3");

	Variable&& wo3 = Variable(trunGaussianRandomNums(vector<int>{layer2Size, layer3Size}, 0.0, 1.0 / sqrt((float)(layer2Size))), "wo3");
	Variable&& uo3 = Variable(trunGaussianRandomNums(vector<int>{layer3Size, layer3Size}, 0.0, 1.0 / sqrt((float)(layer3Size))), "uo3");
	Variable&& bo3 = Variable(NumObject(vector<int>{layer3Size}, 0.0), "bo3");

	Variable&& wc3 = Variable(trunGaussianRandomNums(vector<int>{layer2Size, layer3Size}, 0.0, 1.0 / sqrt((float)(layer2Size))), "wc3");
	Variable&& uc3 = Variable(trunGaussianRandomNums(vector<int>{layer3Size, layer3Size}, 0.0, 1.0 / sqrt((float)(layer3Size))), "uc3");
	Variable&& bc3 = Variable(NumObject(vector<int>{layer3Size}, 0.0), "bc3");

	Variable&& ow1 = Variable(trunGaussianRandomNums(vector<int>{layer3Size, charCount}, 0.0, 1.0 / sqrt((float)(layer3Size))), "ow1");
	Variable&& ob1 = Variable(NumObject(vector<int>{charCount}, 0.0), "ob1");


	Node* f1 = new Sigmoid(new Add(new Add(new MatMul(&xData, &wf1), new MatMul(&h1, &uf1)), &bf1));
	Node* i1 = new Sigmoid(new Add(new Add(new MatMul(&xData, &wi1), new MatMul(&h1, &ui1)), &bi1));
	Node* o1 = new Sigmoid(new Add(new Add(new MatMul(&xData, &wo1), new MatMul(&h1, &uo1)), &bo1));
	Node* tempC1 = new TanH(new Add(new Add(new MatMul(&xData, &wc1), new MatMul(&h1, &uc1)), &bc1));
	Node* newC1 = new Set(new Add(new Multiply(f1, &c1), new Multiply(i1, tempC1)), &c1);
	Node* newH1 = new Multiply(new Set(new Multiply(o1, new TanH(newC1)), &h1), new Constant(NumObject(0.5)));

	Node* f2 = new Sigmoid(new Add(new Add(new MatMul(newH1, &wf2), new MatMul(&h2, &uf2)), &bf2));
	Node* i2 = new Sigmoid(new Add(new Add(new MatMul(newH1, &wi2), new MatMul(&h2, &ui2)), &bi2));
	Node* o2 = new Sigmoid(new Add(new Add(new MatMul(newH1, &wo2), new MatMul(&h2, &uo2)), &bo2));
	Node* tempC2 = new TanH(new Add(new Add(new MatMul(newH1, &wc2), new MatMul(&h2, &uc2)), &bc2));
	Node* newC2 = new Set(new Add(new Multiply(f2, &c2), new Multiply(i2, tempC2)), &c2);
	Node* newH2 = new Multiply(new Set(new Multiply(o2, new TanH(newC2)), &h2), new Constant(NumObject(0.5)));

	Node* f3 = new Sigmoid(new Add(new Add(new MatMul(newH2, &wf3), new MatMul(&h3, &uf3)), &bf3));
	Node* i3 = new Sigmoid(new Add(new Add(new MatMul(newH2, &wi3), new MatMul(&h3, &ui3)), &bi3));
	Node* o3 = new Sigmoid(new Add(new Add(new MatMul(newH2, &wo3), new MatMul(&h3, &uo3)), &bo3));
	Node* tempC3 = new TanH(new Add(new Add(new MatMul(newH2, &wc3), new MatMul(&h3, &uc3)), &bc3));
	Node* newC3 = new Set(new Add(new Multiply(f3, &c3), new Multiply(i3, tempC3)), &c3);
	Node* newH3 = new Multiply(new Set(new Multiply(o3, new TanH(newC3)), &h3), new Constant(NumObject(0.5)));

	Node* o = new Add(new MatMul(newH3, &ow1), &ob1);

	Node* cost = new CrossEntropySoftmax(o, &yData);

	xData.value = {};
	yData.value = {};
	h1.value = {};
	c1.value = {};
	h2.value = {};
	c2.value = {};
	h3.value = {};
	c3.value = {};
	for (int i = 0; i < t; i++){
		xData.value.push_back(NumObject(vector<int> {n, charCount}, 0.0));
		yData.value.push_back(NumObject(vector<int> {n, charCount}, 0.0));
		h1.value.push_back(NumObject(vector<int>{n, layer1Size}, 0.0));
		c1.value.push_back(NumObject(vector<int>{n, layer1Size}, 0.0));
		h2.value.push_back(NumObject(vector<int>{n, layer2Size}, 0.0));
		c2.value.push_back(NumObject(vector<int>{n, layer2Size}, 0.0));
		h3.value.push_back(NumObject(vector<int>{n, layer3Size}, 0.0));
		c3.value.push_back(NumObject(vector<int>{n, layer3Size}, 0.0));
	}
	xData.timeSteps = t;
	yData.timeSteps = t;
	h1.timeSteps = t;
	c1.timeSteps = t;
	h2.timeSteps = t;
	c2.timeSteps = t;
	h3.timeSteps = t;
	c3.timeSteps = t;

	vector<Variable*> variables = {&wf1, &uf1, &bf1, &wi1, &ui1, &bi1, &wo1, &uo1, &bo1, &wc1, &uc1, &bc1, &wf2, &uf2, &bf2, &wi2, &ui2, &bi2, &wo2, &uo2, &bo2, &wc2, &uc2, &bc2, &wf3, &uf3, &bf3, &wi3, &ui3, &bi3, &wo3, &uo3, &bo3, &wc3, &uc3, &bc3, &ow1, &ob1};

	for (int i = 0; i < variables.size(); i++){
		variables[i]->value[0] = loadData("../Weights/CHRIST1_/" + variables[i]->name + ".txt");
	}

	initalize(cost);

	int generationSize = 1000;
	vector<vector<float>> context = textToVec("../Text-io/Context.txt", vocab);
	ofstream output;
	output.open("../Text-io/Output.txt", ios::out | ios::trunc);

	cout << "starting..." << endl;

	for (int i = 0; i < context.size(); i++){
		xData.value[0] = NumObject(context[i], vector<int>{1, charCount});
		xData.updateDeviceVals();

		// output << vecToChar(xData.value[0].values, vocab);

		getValue(cost);

		h1.value[0] = showValue(newH1->inputs[0])[0];
		c1.value[0] = showValue(newC1->inputs[0])[0];
		h2.value[0] = showValue(newH2->inputs[0])[0];
		c2.value[0] = showValue(newC2->inputs[0])[0];
		h3.value[0] = showValue(newH3->inputs[0])[0];
		c3.value[0] = showValue(newC3->inputs[0])[0];
		h1.updateDeviceVals();
		c1.updateDeviceVals();
		h2.updateDeviceVals();
		c2.updateDeviceVals();
		h3.updateDeviceVals();
		c3.updateDeviceVals();
	}

	for (int i = 0; i < generationSize; i++){
		NumObject outVal = showValue(o)[0];
		int maxIdx = 0;
		float maxVal = outVal.values[0];
		NumObject newVals = NumObject(outVal.dimentions, 0.0);

		for (int x = 1; x < outVal.size; x++){
			if (outVal.values[x] > maxVal){
				maxVal = outVal.values[x];
				maxIdx = x;
			}
		}

		newVals.values[maxIdx] = 1.0;

		output << vecToChar(newVals.values, vocab);

		xData.value[0] = newVals;
		xData.updateDeviceVals();

		getValue(cost);

		h1.value[0] = showValue(newH1->inputs[0])[0];
		c1.value[0] = showValue(newC1->inputs[0])[0];
		h2.value[0] = showValue(newH2->inputs[0])[0];
		c2.value[0] = showValue(newC2->inputs[0])[0];
		h3.value[0] = showValue(newH3->inputs[0])[0];
		c3.value[0] = showValue(newC3->inputs[0])[0];
		h1.updateDeviceVals();
		c1.updateDeviceVals();
		h2.updateDeviceVals();
		c2.updateDeviceVals();
		h3.updateDeviceVals();
		c3.updateDeviceVals();
	}

	output.close();
}

void wordLSTM(){
	int t = 50;
	int n = 100;
	int charCount;

	vector<char> vocab = vocabFromFolder("../TextData/THE RESERECTION OF CHRIST");
	vector<vector<float>> data = loadFromFolder("../TextData/THE RESERECTION OF CHRIST", vocab);
	vector<vector<NumObject>> formattedData = breakUpData(data, t, n);

	charCount = (int)(vocab.size());

	int layer1Size = 128;
	int layer2Size = 128;
	int layer3Size = 128;

	Constant&& xData = Constant(NumObject(), "x");
	Constant&& yData = Constant(NumObject(), "y");

	Variable&& h1 = Variable(NumObject(), "h1");
	Variable&& c1 = Variable(NumObject(), "c1");

	Variable&& h2 = Variable(NumObject(), "h2");
	Variable&& c2 = Variable(NumObject(), "c2");

	Variable&& h3 = Variable(NumObject(), "h2");
	Variable&& c3 = Variable(NumObject(), "c2");
	
	Variable&& wf1 = Variable(trunGaussianRandomNums(vector<int>{charCount, layer1Size}, 0.0, 1.0 / sqrt((float)(charCount))), "wf1");
	Variable&& uf1 = Variable(trunGaussianRandomNums(vector<int>{layer1Size, layer1Size}, 0.0, 1.0 / sqrt((float)(layer1Size))), "uf1");
	Variable&& bf1 = Variable(NumObject(vector<int>{layer1Size}, 1.0), "bf1");

	Variable&& wi1 = Variable(trunGaussianRandomNums(vector<int>{charCount, layer1Size}, 0.0, 1.0 / sqrt((float)(charCount))), "wi1");
	Variable&& ui1 = Variable(trunGaussianRandomNums(vector<int>{layer1Size, layer1Size}, 0.0, 1.0 / sqrt((float)(layer1Size))), "ui1");
	Variable&& bi1 = Variable(NumObject(vector<int>{layer1Size}, 0.0), "bi1");

	Variable&& wo1 = Variable(trunGaussianRandomNums(vector<int>{charCount, layer1Size}, 0.0, 1.0 / sqrt((float)(charCount))), "wo1");
	Variable&& uo1 = Variable(trunGaussianRandomNums(vector<int>{layer1Size, layer1Size}, 0.0, 1.0 / sqrt((float)(layer1Size))), "uo1");
	Variable&& bo1 = Variable(NumObject(vector<int>{layer1Size}, 0.0), "bo1");

	Variable&& wc1 = Variable(trunGaussianRandomNums(vector<int>{charCount, layer1Size}, 0.0, 1.0 / sqrt((float)(charCount))), "wc1");
	Variable&& uc1 = Variable(trunGaussianRandomNums(vector<int>{layer1Size, layer1Size}, 0.0, 1.0 / sqrt((float)(layer1Size))), "uc1");
	Variable&& bc1 = Variable(NumObject(vector<int>{layer1Size}, 0.0), "bc1");

	Variable&& wf2 = Variable(trunGaussianRandomNums(vector<int>{layer1Size, layer2Size}, 0.0, 1.0 / sqrt((float)(layer1Size))), "wf2");
	Variable&& uf2 = Variable(trunGaussianRandomNums(vector<int>{layer2Size, layer2Size}, 0.0, 1.0 / sqrt((float)(layer2Size))), "uf2");
	Variable&& bf2 = Variable(NumObject(vector<int>{layer2Size}, 1.0), "bf2");

	Variable&& wi2 = Variable(trunGaussianRandomNums(vector<int>{layer1Size, layer2Size}, 0.0, 1.0 / sqrt((float)(layer1Size))), "wi2");
	Variable&& ui2 = Variable(trunGaussianRandomNums(vector<int>{layer2Size, layer2Size}, 0.0, 1.0 / sqrt((float)(layer2Size))), "ui2");
	Variable&& bi2 = Variable(NumObject(vector<int>{layer2Size}, 0.0), "bi2");

	Variable&& wo2 = Variable(trunGaussianRandomNums(vector<int>{layer1Size, layer2Size}, 0.0, 1.0 / sqrt((float)(layer1Size))), "wo2");
	Variable&& uo2 = Variable(trunGaussianRandomNums(vector<int>{layer2Size, layer2Size}, 0.0, 1.0 / sqrt((float)(layer2Size))), "uo2");
	Variable&& bo2 = Variable(NumObject(vector<int>{layer2Size}, 0.0), "bo2");

	Variable&& wc2 = Variable(trunGaussianRandomNums(vector<int>{layer2Size, layer3Size}, 0.0, 1.0 / sqrt((float)(layer1Size))), "wc2");
	Variable&& uc2 = Variable(trunGaussianRandomNums(vector<int>{layer3Size, layer3Size}, 0.0, 1.0 / sqrt((float)(layer2Size))), "uc2");
	Variable&& bc2 = Variable(NumObject(vector<int>{layer3Size}, 0.0), "bc2");

	Variable&& wf3 = Variable(trunGaussianRandomNums(vector<int>{layer2Size, layer3Size}, 0.0, 1.0 / sqrt((float)(layer2Size))), "wf3");
	Variable&& uf3 = Variable(trunGaussianRandomNums(vector<int>{layer3Size, layer3Size}, 0.0, 1.0 / sqrt((float)(layer3Size))), "uf3");
	Variable&& bf3 = Variable(NumObject(vector<int>{layer3Size}, 1.0), "bf3");

	Variable&& wi3 = Variable(trunGaussianRandomNums(vector<int>{layer2Size, layer3Size}, 0.0, 1.0 / sqrt((float)(layer2Size))), "wi3");
	Variable&& ui3 = Variable(trunGaussianRandomNums(vector<int>{layer3Size, layer3Size}, 0.0, 1.0 / sqrt((float)(layer3Size))), "ui3");
	Variable&& bi3 = Variable(NumObject(vector<int>{layer3Size}, 0.0), "bi3");

	Variable&& wo3 = Variable(trunGaussianRandomNums(vector<int>{layer2Size, layer3Size}, 0.0, 1.0 / sqrt((float)(layer2Size))), "wo3");
	Variable&& uo3 = Variable(trunGaussianRandomNums(vector<int>{layer3Size, layer3Size}, 0.0, 1.0 / sqrt((float)(layer3Size))), "uo3");
	Variable&& bo3 = Variable(NumObject(vector<int>{layer3Size}, 0.0), "bo3");

	Variable&& wc3 = Variable(trunGaussianRandomNums(vector<int>{layer2Size, layer3Size}, 0.0, 1.0 / sqrt((float)(layer2Size))), "wc3");
	Variable&& uc3 = Variable(trunGaussianRandomNums(vector<int>{layer3Size, layer3Size}, 0.0, 1.0 / sqrt((float)(layer3Size))), "uc3");
	Variable&& bc3 = Variable(NumObject(vector<int>{layer3Size}, 0.0), "bc3");

	Variable&& ow1 = Variable(trunGaussianRandomNums(vector<int>{layer3Size, charCount}, 0.0, 1.0 / sqrt((float)(layer3Size))), "ow1");
	Variable&& ob1 = Variable(NumObject(vector<int>{charCount}, 0.0), "ob1");


	Node* f1 = new Sigmoid(new Add(new Add(new MatMul(&xData, &wf1), new MatMul(&h1, &uf1)), &bf1));
	Node* i1 = new Sigmoid(new Add(new Add(new MatMul(&xData, &wi1), new MatMul(&h1, &ui1)), &bi1));
	Node* o1 = new Sigmoid(new Add(new Add(new MatMul(&xData, &wo1), new MatMul(&h1, &uo1)), &bo1));
	Node* tempC1 = new TanH(new Add(new Add(new MatMul(&xData, &wc1), new MatMul(&h1, &uc1)), &bc1));
	Node* newC1 = new Set(new Add(new Multiply(f1, &c1), new Multiply(i1, tempC1)), &c1);
	Dropout* newH1 = new Dropout(new Set(new Multiply(o1, new TanH(newC1)), &h1), 0.5);

	Node* f2 = new Sigmoid(new Add(new Add(new MatMul(newH1, &wf2), new MatMul(&h2, &uf2)), &bf2));
	Node* i2 = new Sigmoid(new Add(new Add(new MatMul(newH1, &wi2), new MatMul(&h2, &ui2)), &bi2));
	Node* o2 = new Sigmoid(new Add(new Add(new MatMul(newH1, &wo2), new MatMul(&h2, &uo2)), &bo2));
	Node* tempC2 = new TanH(new Add(new Add(new MatMul(newH1, &wc2), new MatMul(&h2, &uc2)), &bc2));
	Node* newC2 = new Set(new Add(new Multiply(f2, &c2), new Multiply(i2, tempC2)), &c2);
	Dropout* newH2 = new Dropout(new Set(new Multiply(o2, new TanH(newC2)), &h2), 0.5);

	Node* f3 = new Sigmoid(new Add(new Add(new MatMul(newH2, &wf3), new MatMul(&h3, &uf3)), &bf3));
	Node* i3 = new Sigmoid(new Add(new Add(new MatMul(newH2, &wi3), new MatMul(&h3, &ui3)), &bi3));
	Node* o3 = new Sigmoid(new Add(new Add(new MatMul(newH2, &wo3), new MatMul(&h3, &uo3)), &bo3));
	Node* tempC3 = new TanH(new Add(new Add(new MatMul(newH2, &wc3), new MatMul(&h3, &uc3)), &bc3));
	Node* newC3 = new Set(new Add(new Multiply(f3, &c3), new Multiply(i3, tempC3)), &c3);
	Dropout* newH3 = new Dropout(new Set(new Multiply(o3, new TanH(newC3)), &h3), 0.5);

	Node* o = new Add(new MatMul(newH3, &ow1), &ob1);

	Node* cost = new CrossEntropySoftmax(o, &yData);

	xData.value = {};
	yData.value = {};
	h1.value = {};
	c1.value = {};
	h2.value = {};
	c2.value = {};
	h3.value = {};
	c3.value = {};
	for (int i = 0; i < t; i++){
		xData.value.push_back(NumObject(vector<int> {n, charCount}, 0.0));
		yData.value.push_back(NumObject(vector<int> {n, charCount}, 0.0));
		h1.value.push_back(NumObject(vector<int>{n, layer1Size}, 0.0));
		c1.value.push_back(NumObject(vector<int>{n, layer1Size}, 0.0));
		h2.value.push_back(NumObject(vector<int>{n, layer2Size}, 0.0));
		c2.value.push_back(NumObject(vector<int>{n, layer2Size}, 0.0));
		h3.value.push_back(NumObject(vector<int>{n, layer3Size}, 0.0));
		c3.value.push_back(NumObject(vector<int>{n, layer3Size}, 0.0));
	}
	xData.timeSteps = t;
	yData.timeSteps = t;
	h1.timeSteps = t;
	c1.timeSteps = t;
	h2.timeSteps = t;
	c2.timeSteps = t;
	h3.timeSteps = t;
	c3.timeSteps = t;

	vector<Variable*> variables = {&wf1, &uf1, &bf1, &wi1, &ui1, &bi1, &wo1, &uo1, &bo1, &wc1, &uc1, &bc1, &wf2, &uf2, &bf2, &wi2, &ui2, &bi2, &wo2, &uo2, &bo2, &wc2, &uc2, &bc2, &wf3, &uf3, &bf3, &wi3, &ui3, &bi3, &wo3, &uo3, &bo3, &wc3, &uc3, &bc3, &ow1, &ob1};

	for (int i = 0; i < variables.size(); i++){
		variables[i]->value[0] = loadData("../Weights/CHRIST1_/" + variables[i]->name + ".txt");
	}

	initalize(cost);

	RMSProp optimizer = RMSProp(0.001, 0.9, 0.00000001, variables);

	cout << "starting..." << endl;

	for (int i = 0; i < 100000; i++){
		for (int x = 0; x < formattedData.size() - 1; x++){
			for (int z = 0; z < t - 1; z++){
				xData.value[z] = formattedData[x][z];
				yData.value[z] = formattedData[x][z + 1];
			}
			xData.value[t - 1] = formattedData[x][t - 1];
			yData.value[t - 1] = formattedData[x + 1][0];

			xData.updateDeviceVals();
			yData.updateDeviceVals();

			newH1->updateDrop();
			newH2->updateDrop();
			newH3->updateDrop();
			derive(cost);
			optimizer.minimize();


			h1.value[0] = showValue(newH1)[t - 1];
			c1.value[0] = showValue(newC1)[t - 1];
			h2.value[0] = showValue(newH2)[t - 1];
			c2.value[0] = showValue(newC2)[t - 1];
			h3.value[0] = showValue(newH3)[t - 1];
			c3.value[0] = showValue(newC3)[t - 1];
			for (int z = 1; z < t; z++){
				h1.value[z] = NumObject(vector<int>{n, layer1Size}, 0.0);
				c1.value[z] = NumObject(vector<int>{n, layer1Size}, 0.0);
				h2.value[z] = NumObject(vector<int>{n, layer2Size}, 0.0);
				c2.value[z] = NumObject(vector<int>{n, layer2Size}, 0.0);
				h3.value[z] = NumObject(vector<int>{n, layer3Size}, 0.0);
				c3.value[z] = NumObject(vector<int>{n, layer3Size}, 0.0);
			}
			h1.updateDeviceVals();
			c1.updateDeviceVals();
			h2.updateDeviceVals();
			c2.updateDeviceVals();
			h3.updateDeviceVals();
			c3.updateDeviceVals();

			float total = 0.0;
			vector<NumObject> timeResults = showValue(cost);
			for (int z = 0; z < t; z++){
				total += timeResults[z].values[0];
			}
			cout << i << ", " << x << ", " << to_string(total / (float)(t)) << endl;
		}

		cout << "saving weights..." << endl;

		string directoryName = "../Weights/CHRIST1_";
		for (int z = 0; z < variables.size(); z++){
			variables[z]->updateHostVals();
			saveData(variables[z]->value[0], directoryName + "/" + variables[z]->name + ".txt");
		}
	}
}

vector<vector<NumObject>> breakUpData(vector<vector<float>> rawData, int t, int n){
	int realSize = rawData.size() - rawData.size() % (t * n);
	vector<vector<NumObject>> newData = {};

	for (int i = 0 ; i < realSize / (n * t); i++){
		vector<NumObject> temp = {};
		for (int x = 0; x < t; x++){
			temp.push_back(NumObject(vector<int> {n, (int)(rawData[0].size())}));
		}
		newData.push_back(temp);
	}

	for (int i = 0; i < realSize; i++){
		for (int x = 0; x < rawData[0].size(); x++){
			newData[(i % (realSize / n)) / t][(i % (realSize / n)) % t].values.push_back(rawData[i][x]);
		}
	}

	return newData;
}

void sinLSTM(){
	int t = 100;
	int n = 50;

	Constant&& xData = Constant(NumObject(), "x");
	Constant&& yData = Constant(NumObject(), "y");

	Variable&& h1 = Variable(NumObject(), "h1");
	Variable&& c1 = Variable(NumObject(), "c1");

	Variable&& h2 = Variable(NumObject(), "h2");
	Variable&& c2 = Variable(NumObject(), "c2");
	
	Variable&& wf1 = Variable(trunGaussianRandomNums(vector<int>{1, 64}, 0.0, 1.0 / sqrt(1.0)), "wf1");
	Variable&& uf1 = Variable(trunGaussianRandomNums(vector<int>{64, 64}, 0.0, 1.0 / sqrt(64.0)), "uf1");
	Variable&& bf1 = Variable(NumObject(vector<int>{64}, 0.0), "bf1");

	Variable&& wi1 = Variable(trunGaussianRandomNums(vector<int>{1, 64}, 0.0, 1.0 / sqrt(1.0)), "wi1");
	Variable&& ui1 = Variable(trunGaussianRandomNums(vector<int>{64, 64}, 0.0, 1.0 / sqrt(64.0)), "ui1");
	Variable&& bi1 = Variable(NumObject(vector<int>{64}, 0.0), "bi1");

	Variable&& wo1 = Variable(trunGaussianRandomNums(vector<int>{1, 64}, 0.0, 1.0 / sqrt(1.0)), "wo1");
	Variable&& uo1 = Variable(trunGaussianRandomNums(vector<int>{64, 64}, 0.0, 1.0 / sqrt(64.0)), "uo1");
	Variable&& bo1 = Variable(NumObject(vector<int>{64}, 0.0), "bo1");

	Variable&& wc1 = Variable(trunGaussianRandomNums(vector<int>{1, 64}, 0.0, 1.0 / sqrt(1.0)), "wc1");
	Variable&& uc1 = Variable(trunGaussianRandomNums(vector<int>{64, 64}, 0.0, 1.0 / sqrt(64.0)), "uc1");
	Variable&& bc1 = Variable(NumObject(vector<int>{64}, 0.0), "bc1");

	Variable&& wf2 = Variable(trunGaussianRandomNums(vector<int>{64, 64}, 0.0, 1.0 / sqrt(64.0)), "wf2");
	Variable&& uf2 = Variable(trunGaussianRandomNums(vector<int>{64, 64}, 0.0, 1.0 / sqrt(64.0)), "uf2");
	Variable&& bf2 = Variable(NumObject(vector<int>{64}, 0.0), "bf2");

	Variable&& wi2 = Variable(trunGaussianRandomNums(vector<int>{64, 64}, 0.0, 1.0 / sqrt(64.0)), "wi2");
	Variable&& ui2 = Variable(trunGaussianRandomNums(vector<int>{64, 64}, 0.0, 1.0 / sqrt(64.0)), "ui2");
	Variable&& bi2 = Variable(NumObject(vector<int>{64}, 0.0), "bi2");

	Variable&& wo2 = Variable(trunGaussianRandomNums(vector<int>{64, 64}, 0.0, 1.0 / sqrt(64.0)), "wo2");
	Variable&& uo2 = Variable(trunGaussianRandomNums(vector<int>{64, 64}, 0.0, 1.0 / sqrt(64.0)), "uo2");
	Variable&& bo2 = Variable(NumObject(vector<int>{64}, 0.0), "bo2");

	Variable&& wc2 = Variable(trunGaussianRandomNums(vector<int>{64, 64}, 0.0, 1.0 / sqrt(64.0)), "wc2");
	Variable&& uc2 = Variable(trunGaussianRandomNums(vector<int>{64, 64}, 0.0, 1.0 / sqrt(64.0)), "uc2");
	Variable&& bc2 = Variable(NumObject(vector<int>{64}, 0.0), "bc2");

	Variable&& ow1 = Variable(trunGaussianRandomNums(vector<int>{64, 1}, 0.0, 1.0 / sqrt(64.0)), "ow1");
	Variable&& ob1 = Variable(NumObject(vector<int>{1}, 0.0), "ob1");


	Node* f1 = new Sigmoid(new Add(new Add(new MatMul(&xData, &wf1), new MatMul(&h1, &uf1)), &bf1));
	Node* i1 = new Sigmoid(new Add(new Add(new MatMul(&xData, &wi1), new MatMul(&h1, &ui1)), &bi1));
	Node* o1 = new Sigmoid(new Add(new Add(new MatMul(&xData, &wo1), new MatMul(&h1, &uo1)), &bo1));
	Node* tempC1 = new Softsign(new Add(new Add(new MatMul(&xData, &wc1), new MatMul(&h1, &uc1)), &bc1));
	Node* newC1 = new Set(new Add(new Multiply(f1, &c1), new Multiply(i1, tempC1)), &c1);
	Node* newH1 = new Set(new Multiply(o1, new Softsign(newC1)), &h1);

	Node* f2 = new Sigmoid(new Add(new Add(new MatMul(newH1, &wf2), new MatMul(&h2, &uf2)), &bf2));
	Node* i2 = new Sigmoid(new Add(new Add(new MatMul(newH1, &wi2), new MatMul(&h2, &ui2)), &bi2));
	Node* o2 = new Sigmoid(new Add(new Add(new MatMul(newH1, &wo2), new MatMul(&h2, &uo2)), &bo2));
	Node* tempC2 = new Softsign(new Add(new Add(new MatMul(newH1, &wc2), new MatMul(&h2, &uc2)), &bc2));
	Node* newC2 = new Set(new Add(new Multiply(f2, &c2), new Multiply(i2, tempC2)), &c2);
	Node* newH2 = new Set(new Multiply(o2, new Softsign(newC2)), &h2);

	Node* o = new TanH(new Add(new MatMul(newH2, &ow1), &ob1));

	Node* cost = new MeanSquared(o, &yData);

	xData.value = {};
	yData.value = {};
	h1.value = {};
	c1.value = {};
	h2.value = {};
	c2.value = {};
	for (int i = 0; i < t; i++){
		xData.value.push_back(NumObject(vector<int> {n, 1}, 0.0));
		yData.value.push_back(NumObject(vector<int> {n, 1}, 0.0));
		h1.value.push_back(NumObject(vector<int>{n, 64}, 0.0));
		c1.value.push_back(NumObject(vector<int>{n, 64}, 0.0));
		h2.value.push_back(NumObject(vector<int>{n, 64}, 0.0));
		c2.value.push_back(NumObject(vector<int>{n, 64}, 0.0));
	}
	xData.timeSteps = t;
	yData.timeSteps = t;
	h1.timeSteps = t;
	c1.timeSteps = t;
	h2.timeSteps = t;
	c2.timeSteps = t;

	initalize(cost);

	vector<Variable*> variables = {&wf1, &uf1, &bf1, &wi1, &ui1, &bi1, &wo1, &uo1, &bo1, &wc1, &uc1, &bc1, &wf2, &uf2, &bf2, &wi2, &ui2, &bi2, &wo2, &uo2, &bo2, &wc2, &uc2, &bc2, &ow1, &ob1};

	RMSProp optimizer = RMSProp(0.001, 0.9, 0.0, variables);

	cout << "starting..." << endl;

	for (int i = 0; i < 5001; i++){
		for (int z = 0; z < t; z++){
			h1.value[z] = NumObject(vector<int>{n, 64}, 0.0);
			c1.value[z] = NumObject(vector<int>{n, 64}, 0.0);
			h2.value[z] = NumObject(vector<int>{n, 64}, 0.0);
			c2.value[z] = NumObject(vector<int>{n, 64}, 0.0);
		}
		h1.updateDeviceVals();
		c1.updateDeviceVals();
		h2.updateDeviceVals();
		c2.updateDeviceVals();

		for (int x = 0; x < t; x++){
			for (int z = 0; z < n; z++){
				xData.value[x].values[z] = sin(((float)(t) * 0.2 * (float)(z + (i % 101))) + (float)(x + (i % 11)) * 0.2);
				yData.value[x].values[z] = sin(((float)(t) * 0.2 * (float)(z + (i % 101))) + (float)(x + (i % 11) + 1) * 0.2);
			}
		}

		xData.updateDeviceVals();
		yData.updateDeviceVals();

		derive(cost);
		optimizer.minimize();

		float total = 0.0;
		vector<NumObject> timeResults = showValue(cost);
		for (int x = 0; x < t; x++){
			total += timeResults[x].values[0];
		}

		cout << i << ", " << to_string(total) << endl;
	}

	xData.value = {};
	yData.value = {};
	h1.value = {};
	c1.value = {};
	h2.value = {};
	c2.value = {};
	for (int i = 0; i < 1; i++){
		xData.value.push_back(NumObject(vector<int> {1, 1}, 0.0));
		yData.value.push_back(NumObject(vector<int> {1, 1}, 0.0));
		h1.value.push_back(NumObject(vector<int>{1, 64}, 0.0));
		c1.value.push_back(NumObject(vector<int>{1, 64}, 0.0));
		h2.value.push_back(NumObject(vector<int>{1, 64}, 0.0));
		c2.value.push_back(NumObject(vector<int>{1, 64}, 0.0));
	}
	xData.timeSteps = t;
	yData.timeSteps = t;
	h1.timeSteps = t;
	c1.timeSteps = t;
	h2.timeSteps = t;
	c2.timeSteps = t;

	wf1.updateHostVals();
	uf1.updateHostVals();
	bf1.updateHostVals();
	wi1.updateHostVals();
	ui1.updateHostVals();
	bi1.updateHostVals();
	wo1.updateHostVals();
	uo1.updateHostVals();
	bo1.updateHostVals();
	wc1.updateHostVals();
	uc1.updateHostVals();
	bc1.updateHostVals();
	wf2.updateHostVals();
	uf2.updateHostVals();
	bf2.updateHostVals();
	wi2.updateHostVals();
	ui2.updateHostVals();
	bi2.updateHostVals();
	wo2.updateHostVals();
	uo2.updateHostVals();
	bo2.updateHostVals();
	wc2.updateHostVals();
	uc2.updateHostVals();
	bc2.updateHostVals();
	ow1.updateHostVals();
	ob1.updateHostVals();

	initalize(cost);

	for (int i = 0; i < 11; i++){
		xData.value[0] = sin(i * 0.2);
		xData.updateDeviceVals();

		getValue(cost);

		h1.value[0] = showValue(newH1)[0];
		c1.value[0] = showValue(newC1)[0];
		h2.value[0] = showValue(newH2)[0];
		c2.value[0] = showValue(newC2)[0];
		h1.updateDeviceVals();
		c1.updateDeviceVals();
		h2.updateDeviceVals();
		c2.updateDeviceVals();
	}

	for (int i = 0; i < 51; i++){
		xData.value[0] = showValue(o)[0];
		xData.updateDeviceVals();

		cout << (i + 11) * 0.2 << ", " << xData.value[0].describe() << endl;

		getValue(cost);

		h1.value[0] = showValue(newH1)[0];
		c1.value[0] = showValue(newC1)[0];
		h2.value[0] = showValue(newH2)[0];
		c2.value[0] = showValue(newC2)[0];
		h1.updateDeviceVals();
		c1.updateDeviceVals();
		h2.updateDeviceVals();
		c2.updateDeviceVals();
	}
}

void sinRNN(){
	int t = 20;
	int n = 50;

	Constant&& xData = Constant(NumObject(), "x");
	Constant&& yData = Constant(NumObject(), "y");

	Variable&& outVal1 = Variable(NumObject(), "outVal1");
	Variable&& outVal2 = Variable(NumObject(), "outVal2");

	Variable&& w1 = Variable(trunGaussianRandomNums(vector<int>{1, 64}, 0.0, 1.0 / sqrt(1.0)), "w1");
	Variable&& u1 = Variable(trunGaussianRandomNums(vector<int>{64, 64}, 0.0, 1.0 / sqrt(64.0)), "u1");
	Variable&& b1 = Variable(gaussianRandomNums(vector<int>{64}, 0.0, 1.0), "b1");

	Variable&& w2 = Variable(trunGaussianRandomNums(vector<int>{64, 64}, 0.0, 1.0 / sqrt(64.0)), "w2");
	Variable&& u2 = Variable(trunGaussianRandomNums(vector<int>{64, 64}, 0.0, 1.0 / sqrt(64.0)), "u2");
	Variable&& b2 = Variable(gaussianRandomNums(vector<int>{64}, 0.0, 1.0), "b2");

	Variable&& w3 = Variable(trunGaussianRandomNums(vector<int>{64, 1}, 0.0, 1.0 / sqrt(64.0)), "w3");
	Variable&& b3 = Variable(gaussianRandomNums(vector<int>{1}, 0.0, 1.0), "b3");

	Node* layer1 = new Set(new TanH(new Add(new Add(new MatMul(&xData, &w1), new MatMul(&outVal1, &u1)), &b1)), &outVal1);
	Node* layer2 = new Set(new TanH(new Add(new Add(new MatMul(layer1, &w2), new MatMul(&outVal2, &u2)), &b2)), &outVal2);
	Node* layer3 = new TanH(new Add(new MatMul(layer2, &w3), &b3));

	Node* cost = new MeanSquared(layer3, &yData);

	xData.value = {};
	yData.value = {};
	outVal1.value = {};
	outVal2.value = {};
	for (int i = 0; i < t; i++){
		xData.value.push_back(NumObject(vector<int> {n, 1}, 0.0));
		yData.value.push_back(NumObject(vector<int> {n, 1}, 0.0));
		outVal1.value.push_back(NumObject(vector<int>{n, 64}, 0.0));
		outVal2.value.push_back(NumObject(vector<int>{n, 64}, 0.0));
	}
	xData.timeSteps = t;
	yData.timeSteps = t;
	outVal1.timeSteps = t;
	outVal2.timeSteps = t;

	initalize(cost);

	vector<Variable*> variables = {&w1, &u1, &b1, &w2, &u2, &b2, &w3, &b3};

	RMSProp optimizer = RMSProp(0.001, 0.9, 0.0, variables);

	cout << "starting..." << endl;

	for (int i = 0; i < 5001; i++){
		for (int z = 0; z < t; z++){
			outVal1.value[z] = NumObject(vector<int>{n, 64}, 0.0);
			outVal2.value[z] = NumObject(vector<int>{n, 64}, 0.0);
		}
		outVal1.updateDeviceVals();
		outVal2.updateDeviceVals();

		for (int x = 0; x < t; x++){
			for (int z = 0; z < n; z++){
				xData.value[x].values[z] = sin(((float)(t) * 0.2 * (float)(z + (i % 101))) + (float)(x + (i % 11)) * 0.2);
				yData.value[x].values[z] = sin(((float)(t) * 0.2 * (float)(z + (i % 101))) + (float)(x + (i % 11) + 1) * 0.2);
			}
		}

		xData.updateDeviceVals();
		yData.updateDeviceVals();

		derive(cost);
		optimizer.minimize();

		float total = 0.0;
		vector<NumObject> timeResults = showValue(cost);
		for (int x = 0; x < t; x++){
			total += timeResults[x].values[0];
		}

		cout << i << ", " << to_string(total) << endl;
	}

	xData.value = {};
	yData.value = {};
	outVal1.value = {};
	outVal2.value = {};
	for (int i = 0; i < 1; i++){
		xData.value.push_back(NumObject(vector<int> {1, 1}, 0.0));
		yData.value.push_back(NumObject(vector<int> {1, 1}, 0.0));
		outVal1.value.push_back(NumObject(vector<int>{1, 64}, 0.0));
		outVal2.value.push_back(NumObject(vector<int>{1, 64}, 0.0));
	}
	xData.timeSteps = 1;
	yData.timeSteps = 1;
	outVal1.timeSteps = 1;
	outVal2.timeSteps = 1;

	w1.updateHostVals();
	u1.updateHostVals();
	b1.updateHostVals();
	w2.updateHostVals();
	u2.updateHostVals();
	b2.updateHostVals();
	w3.updateHostVals();
	b3.updateHostVals();

	initalize(cost);

	for (int i = 0; i < 11; i++){
		xData.value[0] = sin(i * 0.2);
		xData.updateDeviceVals();

		getValue(cost);

		outVal1.value[0] = showValue(layer1)[0];
		outVal2.value[0] = showValue(layer2)[0];
		outVal1.updateDeviceVals();
		outVal2.updateDeviceVals();
	}

	for (int i = 0; i < 51; i++){
		xData.value[0] = showValue(layer3)[0];
		xData.updateDeviceVals();

		cout << (i + 11) * 0.2 << ", " << xData.value[0].describe() << endl;

		getValue(cost);

		outVal1.value[0] = showValue(layer1)[0];
		outVal2.value[0] = showValue(layer2)[0];
		outVal1.updateDeviceVals();
		outVal2.updateDeviceVals();
	}
}

// void WordRNN(){
// 	vector<vector<vector<float>>> allData = loadFromFolder("../TextData/Rap/Kanye West/");
// 	vector<vector<vector<float>>> rawTextData = {};
// 	vector<int> indicies = {};
// 	int maxSize = 0;

// 	int t = 100;
// 	int n = 50;
// 	int charCount = 74;


// 	Constant&& xData = Constant(NumObject(), "x");
// 	Constant&& yData = Constant(NumObject(), "y");

// 	Variable&& outVal1 = Variable(NumObject(), "outVal1");
// 	Variable&& outVal2 = Variable(NumObject(), "outVal2");

// 	Variable&& w1 = Variable(trunGaussianRandomNums(vector<int>{charCount, 64}, 0.0, 1.0 / sqrt((float)(charCount))), "w1");
// 	Variable&& u1 = Variable(trunGaussianRandomNums(vector<int>{64, 64}, 0.0, 1.0 / sqrt(64.0)), "u1");
// 	Variable&& b1 = Variable(gaussianRandomNums(vector<int>{64}, 0.0, 1.0), "b1");

// 	Variable&& w2 = Variable(trunGaussianRandomNums(vector<int>{64, 64}, 0.0, 1.0 / sqrt(64.0)), "w2");
// 	Variable&& u2 = Variable(trunGaussianRandomNums(vector<int>{64, 64}, 0.0, 1.0 / sqrt(64.0)), "u2");
// 	Variable&& b2 = Variable(gaussianRandomNums(vector<int>{64}, 0.0, 1.0), "b2");

// 	Variable&& w3 = Variable(trunGaussianRandomNums(vector<int>{64, charCount}, 0.0, 1.0 / sqrt(64.0)), "w3");
// 	Variable&& b3 = Variable(gaussianRandomNums(vector<int>{charCount}, 0.0, 1.0), "b3");

// 	Node* layer1 = new Set(new TanH(new Add(new Add(new MatMul(&xData, &w1), new MatMul(&outVal1, &u1)), &b1)), &outVal1);
// 	Node* layer2 = new Set(new TanH(new Add(new Add(new MatMul(layer1, &w2), new MatMul(&outVal2, &u2)), &b2)), &outVal2);
// 	Node* layer3 = new Add(new MatMul(layer2, &w3), &b3);

// 	Node* cost = new CrossEntropySoftmax(layer3, &yData);

// 	xData.value = {};
// 	yData.value = {};
// 	outVal1.value = {};
// 	outVal2.value = {};
// 	for (int i = 0; i < t; i++){
// 		xData.value.push_back(NumObject(vector<int> {n, charCount}, 0.0));
// 		yData.value.push_back(NumObject(vector<int> {n, charCount}, 0.0));
// 		outVal1.value.push_back(NumObject(vector<int>{n, 64}, 0.0));
// 		outVal2.value.push_back(NumObject(vector<int>{n, 64}, 0.0));
// 	}
// 	xData.timeSteps = t;
// 	yData.timeSteps = t;
// 	outVal1.timeSteps = t;
// 	outVal2.timeSteps = t;

// 	initalize(cost);

// 	vector<Variable*> variables = {&w1, &u1, &b1, &w2, &u2, &b2, &w3, &b3};

// 	RMSProp optimizer = RMSProp(0.001, 0.9, 0.00000001, variables);

// 	cout << "starting..." << endl;

// 	for (int i = 0; i < 10001; i++){
// 		if ((i - 1) % 10 == 0){
// 			xData.value = {};
// 			yData.value = {};
// 			outVal1.value = {};
// 			outVal2.value = {};
// 			for (int z = 0; z < t; z++){
// 				xData.value.push_back(NumObject(vector<int> {n, charCount}, 0.0));
// 				yData.value.push_back(NumObject(vector<int> {n, charCount}, 0.0));
// 				outVal1.value.push_back(NumObject(vector<int>{n, 64}, 0.0));
// 				outVal2.value.push_back(NumObject(vector<int>{n, 64}, 0.0));
// 			}
// 			xData.timeSteps = t;
// 			yData.timeSteps = t;
// 			outVal1.timeSteps = t;
// 			outVal2.timeSteps = t;

// 			w1.updateHostVals();
// 			u1.updateHostVals();
// 			b1.updateHostVals();
// 			w2.updateHostVals();
// 			u2.updateHostVals();
// 			b2.updateHostVals();
// 			w3.updateHostVals();
// 			b3.updateHostVals();

// 			initalize(cost);
// 		}

// 		for (int z = 0; z < t; z++){
// 			outVal1.value[z] = NumObject(vector<int>{n, 64}, 0.0);
// 			outVal2.value[z] = NumObject(vector<int>{n, 64}, 0.0);
// 		}
// 		outVal1.updateDeviceVals();
// 		outVal2.updateDeviceVals();



// 		rawTextData = {};
// 		indicies = {};
// 		maxSize = 0;

// 		for (int i = 0; i < n; i++){
// 			indicies.push_back((int) ((((float) rand()) / (float) RAND_MAX) * (float)(allData.size())));
// 			// indicies.push_back(i % allData.size());
// 		}

// 		for (int i = 0; i < n; i++){
// 			rawTextData.push_back(allData[indicies[i]]);
// 			maxSize = max(maxSize, (int)(allData[indicies[i]].size()));
// 		}

// 		for (int i = 0; i < n; i++){
// 			while(rawTextData[i].size() < maxSize){
// 				vector<float> temp = {};
// 				for (int i = 0; i < charCount; i++){
// 					temp.push_back(0.0);
// 				}
// 				rawTextData[i].push_back(temp);
// 			}
// 		}

// 		float total = 0.0;
// 		for (int x = 0; x < (maxSize - 1); x += t){
// 			for (int z = 0; z < t; z++){
// 				if (x + z < rawTextData[0].size()){
// 					vector<float> dataVals = {};
// 					for (int b = 0; b < n; b++){
// 						for (int a = 0; a < charCount; a++){
// 							dataVals.push_back(rawTextData[b][x + z][a]);
// 						}
// 					}
// 					xData.value[z] = NumObject(dataVals, vector<int> {n, charCount});
// 				}
// 				else{
// 					xData.value[z] = NumObject(vector<int> {n, charCount}, 0.0);
// 				}
// 				if (x + z + 1 < rawTextData[0].size()){
// 					vector<float> dataVals = {};
// 					for (int b = 0; b < n; b++){
// 						for (int a = 0; a < charCount; a++){
// 							dataVals.push_back(rawTextData[b][x + z + 1][a]);
// 						}
// 					}
// 					yData.value[z] = NumObject(dataVals, vector<int> {n, charCount});
// 				}
// 				else{
// 					yData.value[z] = NumObject(vector<int> {n, charCount}, 0.0);
// 				}
// 			}
// 			xData.updateDeviceVals();
// 			yData.updateDeviceVals();

// 			derive(cost);
// 			optimizer.minimize();
			

// 			outVal1.value[0] = showValue(layer1)[t - 1];
// 			outVal2.value[0] = showValue(layer2)[t - 1];
// 			for (int z = 1; z < t; z++){
// 				outVal1.value[z] = NumObject(vector<int>{n, 64}, 0.0);
// 				outVal2.value[z] = NumObject(vector<int>{n, 64}, 0.0);
// 			}
// 			outVal1.updateDeviceVals();
// 			outVal2.updateDeviceVals();



// 			vector<NumObject> timeResults = showValue(cost);
// 			for (int z = 0; z < t; z++){
// 				total += timeResults[z].values[0];
// 			}
// 		}
// 		cout << i << ", " << to_string(total / (maxSize - 1)) << endl;


// 		if (i % 10 == 0){
// 			string context = "[Hook: Syleena Johnson + Kanye West]\nOh, when it all, it all falls down\nYeah, this the real one, baby\nI'm tellin' you all, it all falls down\nUh, Chi-Town, stand up!\nOh, when it all, it all falls down\nSouthside, Southside\nWe gon' set this party off right\nI'm tellin' you all, it all falls down\nWestside, Westside\nWe gon' set this party off right\nOh, when it all";

// 			xData.value = {NumObject(vector<int> {1, charCount}, 0.0)};
// 			yData.value = {NumObject(vector<int> {1, charCount}, 0.0)};
// 			outVal1.value = {NumObject(vector<int>{1, 64}, 0.0)};
// 			outVal2.value = {NumObject(vector<int>{1, 64}, 0.0)};
// 			xData.timeSteps = 1;
// 			yData.timeSteps = 1;
// 			outVal1.timeSteps = 1;
// 			outVal2.timeSteps = 1;

// 			w1.updateHostVals();
// 			u1.updateHostVals();
// 			b1.updateHostVals();
// 			w2.updateHostVals();
// 			u2.updateHostVals();
// 			b2.updateHostVals();
// 			w3.updateHostVals();
// 			b3.updateHostVals();

// 			initalize(cost);

// 			for (int x = 0; x < context.size(); x++){
// 				xData.value[0].values = charToVec(context[x]);

// 				xData.updateDeviceVals();

// 				getValue(cost);

// 				outVal1.value[0] = showValue(layer1)[0];
// 				outVal2.value[0] = showValue(layer2)[0];
// 				outVal1.updateDeviceVals();
// 				outVal2.updateDeviceVals();
// 			}

// 			for (int x = 0; x < 200; x++){
// 				NumObject ans = showValue(layer3)[0];
// 				int maxIdx = 0;
// 				float maxVal = -100000.0;
// 				for (int z = 0; z < ans.size; z++){
// 					if (ans.values[z] > maxVal){
// 						maxVal = ans.values[z];
// 						maxIdx = z;
// 					}
// 				}


// 				vector<float> newX = {};
// 				for (int z = 0; z < ans.size; z++){
// 					if (z == maxIdx){
// 						newX.push_back(1.0);
// 					}
// 					else{
// 						newX.push_back(0.0);
// 					}
// 				}

// 				char newChar = vecToChar(newX);
// 				context += newChar;

// 				xData.value[0].values = newX;

// 				xData.updateDeviceVals();

// 				getValue(cost);

// 				outVal1.value[0] = showValue(layer1)[0];
// 				outVal2.value[0] = showValue(layer2)[0];
// 				outVal1.updateDeviceVals();
// 				outVal2.updateDeviceVals();
// 			}

// 			cout << context << endl;
// 		}
// 	}
// }

void charRNN(){
	Constant&& xData = Constant(NumObject(), "x");
	Constant&& yData = Constant(NumObject(), "y");

	Variable&& outVal1 = Variable(NumObject(), "outVal1");
	Variable&& outVal2 = Variable(NumObject(), "outVal2");

	Variable&& w1 = Variable(trunGaussianRandomNums(vector<int>{26, 256}, 0.0, 1.0 / sqrt(26.0)), "w1");
	Variable&& u1 = Variable(trunGaussianRandomNums(vector<int>{256, 256}, 0.0, 1.0 / sqrt(256.0)), "u1");
	Variable&& b1 = Variable(gaussianRandomNums(vector<int>{256}, 0.0, 1.0), "b1");

	Variable&& w2 = Variable(trunGaussianRandomNums(vector<int>{256, 256}, 0.0, 1.0 / sqrt(256.0)), "w2");
	Variable&& u2 = Variable(trunGaussianRandomNums(vector<int>{256, 256}, 0.0, 1.0 / sqrt(256.0)), "u2");
	Variable&& b2 = Variable(gaussianRandomNums(vector<int>{256}, 0.0, 1.0), "b2");

	Variable&& w3 = Variable(trunGaussianRandomNums(vector<int>{256, 26}, 0.0, 1.0 / sqrt(256.0)), "w3");
	Variable&& b3 = Variable(gaussianRandomNums(vector<int>{26}, 0.0, 1.0), "b3");

	Node* layer1 = new Set(new TanH(new Add(new Add(new MatMul(&xData, &w1), new MatMul(&outVal1, &u1)), &b1)), &outVal1);
	Node* layer2 = new Set(new TanH(new Add(new Add(new MatMul(layer1, &w2), new MatMul(&outVal2, &u2)), &b2)), &outVal2);
	Node* layer3 = new Add(new MatMul(layer2, &w3), &b3);

	Node* cost = new CrossEntropySoftmax(layer3, &yData);

	vector<vector<NumObject>> textData = load5LetterWords();

	xData.value = textData[0];
	yData.value = textData[1];
	xData.timeSteps = xData.value.size();
	yData.timeSteps = yData.value.size();

	outVal1.value = {};
	outVal2.value = {};
	for (int i = 0; i < xData.value.size(); i++){
		outVal1.value.push_back(NumObject(vector<int>{xData.value[0].dimentions[0], 256}, 0.0));
		outVal2.value.push_back(NumObject(vector<int>{xData.value[0].dimentions[0], 256}, 0.0));
	}
	outVal1.timeSteps = outVal1.value.size();
	outVal2.timeSteps = outVal2.value.size();

	initalize(cost);

	vector<Variable*> variables = {&w1, &u1, &b1, &w2, &u2, &b2, &w3, &b3};

	cout << "starting..." << endl;

	for (int i =  0; i < 5001; i++){
		derive(cost);
		gradientDescent(variables, 0.01);

		if (i % 10 == 0){
			vector<NumObject> timeResults = showValue(cost);
			float total = 0.0;
			for (int x = 0; x < xData.value.size(); x++){
				total += timeResults[x].values[0];
			}
			cout << i << ", " << to_string(total) << endl;
		}
	}
}

void MNISTFFNN(){
	Constant&& xData = Constant(NumObject(), "x");
	Constant&& yData = Constant(NumObject(), "y");
	Variable&& weights1 = Variable(gaussianRandomNums(vector<int>{784, 100}, 0.0, 1.0 / sqrt(784.0)), "w1");
	Variable&& weights2 = Variable(gaussianRandomNums(vector<int>{100, 100}, 0.0, 1.0 / sqrt(100.0)), "w2");
	Variable&& weights3 = Variable(gaussianRandomNums(vector<int>{100, 10}, 0.0, 1.0 / sqrt(100.0)), "w3");
	Variable&& bias1 = Variable(gaussianRandomNums(vector<int>{100}, 0.0, 1.0), "bias1");
	Variable&& bias2 = Variable(gaussianRandomNums(vector<int>{100}, 0.0, 1.0), "bias2");
	Variable&& bias3 = Variable(gaussianRandomNums(vector<int>{10}, 0.0, 1.0), "bias3");

	Node* layer1 = new ReLU(new Add(new MatMul(&xData, &weights1), &bias1));
	Node* layer2 = new ReLU(new Add(new MatMul(layer1, &weights2), &bias2));
	Node* layer3 = new Add(new MatMul(layer2, &weights3), &bias3);

	Node* cost = new CrossEntropySoftmax(layer3, &yData);


	xData.value = {};
	yData.value = {};
	for (int i = 0; i < 10; i++){
		vector<NumObject> trainData = getTrain(100);
		xData.value.push_back(trainData[0]);
		yData.value.push_back(trainData[1]);
	}
	xData.timeSteps = 10;
	yData.timeSteps = 10;

	initalize(cost);

	vector<Variable*> variables = {&weights1, &bias1, &weights2, &bias2, &weights3, &bias3};

	cout << "starting..." << endl;
	for(int i = 0; i < 5001; i++){
		xData.value = {};
		yData.value = {};
		for (int i = 0; i < 10; i++){
			vector<NumObject> trainData = getTrain(100);
			xData.value.push_back(trainData[0]);
			yData.value.push_back(trainData[1]);
		}
		xData.timeSteps = 10;
		yData.timeSteps = 10;

		xData.updateDeviceVals();
		yData.updateDeviceVals();

		derive(cost);
		gradientDescent(variables, 0.01);

		if(i % 10 == 0){
			vector<NumObject> timeResults = showValue(cost);
			float total = 0.0;
			for (int x = 0; x < 10; x++){
				total += timeResults[x].values[0];
			}
			cout << i << ", " << total << endl;
		}
	}

	vector<NumObject> testData = getTest(20000);
	xData.value[0] = testData[0];
	yData.value[0] = testData[1];
	xData.timeSteps = 1;
	yData.timeSteps = 1;
	weights1.updateHostVals();
	bias1.updateHostVals();
	weights2.updateHostVals();
	bias2.updateHostVals();
	weights3.updateHostVals();
	bias3.updateHostVals();

	initalize(cost);

	getValue(cost);

	NumObject prediction = showValue(layer3)[0];

	float total = 0.0;
	float currentVal;
	float maxVal;
	int indexVal;
	for (int i = 0; i < prediction.dimentions[0]; i++){
		indexVal = -1;
		maxVal = -numeric_limits<float>::max();
		for (int x = 0; x < prediction.dimentions[1]; x++){
			currentVal = prediction.values[i * prediction.dimentions[1] + x];
			if (currentVal > maxVal){
				maxVal = currentVal;
				indexVal = x;
			}
		}
		if (yData.value[0].values[i * prediction.dimentions[1] + indexVal] == 1){
			total += 1.0;
		}
	}

	cout << total / (float)(prediction.dimentions[0]) << endl;
}

vector<NumObject> getTrain(int n){
	vector<NumObject> temp = randomTrainSet(n);

	for(int i = 0; i < temp[0].values.size(); i++){
		temp[0].values[i] = (temp[0].values[i] - 128.0) / 256.0;
	}

	NumObject yFinal = oneHot(temp[1], 0, 9);

	vector<NumObject> ans = {temp[0], yFinal};

	return ans;
}

vector<NumObject> getTest(int n){
	vector<NumObject> temp = randomTestSet(n);
	
	for(int i = 0; i < temp[0].values.size(); i++){
		temp[0].values[i] = (temp[0].values[i] - 128.0) / 256.0;
	}

	NumObject yFinal = oneHot(temp[1], 0, 9);

	vector<NumObject> ans = {temp[0], yFinal};

	return ans;
}






void linearReg(){
	Constant&& xData = Constant(gaussianRandomNums(vector<int>{640, 640}, -10.0, 10.0));
	for(int i = 0; i < xData.value[0].dimentions[0]; i++){
		xData.value[0].values[i * xData.value[0].dimentions[1] + 639] = 1.0;
	}

	Node* yValExpression = new Sum(new Multiply(&xData, new Constant(NumObject(0.5))), 1);
	initalize(yValExpression);
	Constant&& yData = Constant(getValue(yValExpression));
	clearHistory(&xData);

	Variable&& weights = Variable(gaussianRandomNums(vector<int>{640}, -0.5, 0.5));

	Node* hypothesis = new MatMul(&xData, &weights);
	Node* cost = new MeanSquared(hypothesis, &yData);
	// Node* cost = new Mean(new Pow(new Subtract(hypothesis, &yData), new Constant(2.0)));

	initalize(cost);

	vector<Variable*> variables = {&weights};

	cout << "starting..." << endl;
	for(int i = 0; i < 50000; i++){
		derive(cost);
		gradientDescent(variables, 0.0000001);

		if(i % 10 == 0){
			cout << i << ", " << showValue(cost)[0].describe() << endl;
		}
	}
	cout << showValue(cost)[0].describe() << endl;
	// cout << showValue(cost).describe() << endl;
	// weights.updateHostVals();
	// cout << weights.describe() << endl;
}

// void rubixNet(){
// 	Constant&& xData = Constant(NumObject(), "x");
// 	Constant&& yData = Constant(NumObject(), "y");

// 	Variable&& h1 = Variable(NumObject(), "h1");
// 	Variable&& c1 = Variable(NumObject(), "c1");

// 	Variable&& h2 = Variable(NumObject(), "h2");
// 	Variable&& c2 = Variable(NumObject(), "c2");
	
// 	Variable&& wf1 = Variable(trunGaussianRandomNums(vector<int>{147, 64}, -0.1, 0.1), "wf1");
// 	Variable&& uf1 = Variable(trunGaussianRandomNums(vector<int>{64, 64}, -0.1, 0.1), "uf1");
// 	Variable&& bf1 = Variable(gaussianRandomNums(vector<int>{64}, 0.0, 1.0), "bf1");

// 	Variable&& wi1 = Variable(trunGaussianRandomNums(vector<int>{147, 64}, -0.1, 0.1), "wi1");
// 	Variable&& ui1 = Variable(trunGaussianRandomNums(vector<int>{64, 64}, -0.1, 0.1), "ui1");
// 	Variable&& bi1 = Variable(gaussianRandomNums(vector<int>{64}, 0.0, 1.0), "bi1");

// 	Variable&& wo1 = Variable(trunGaussianRandomNums(vector<int>{147, 64}, -0.1, 0.1), "wo1");
// 	Variable&& uo1 = Variable(trunGaussianRandomNums(vector<int>{64, 64}, -0.1, 0.1), "uo1");
// 	Variable&& bo1 = Variable(gaussianRandomNums(vector<int>{64}, 0.0, 1.0), "bo1");

// 	Variable&& wc1 = Variable(trunGaussianRandomNums(vector<int>{147, 64}, -0.1, 0.1), "wc1");
// 	Variable&& uc1 = Variable(trunGaussianRandomNums(vector<int>{64, 64}, -0.1, 0.1), "uc1");
// 	Variable&& bc1 = Variable(gaussianRandomNums(vector<int>{64}, 0.0, 1.0), "bc1");

// 	Variable&& wf2 = Variable(trunGaussianRandomNums(vector<int>{64, 64}, -0.1, 0.1), "wf2");
// 	Variable&& uf2 = Variable(trunGaussianRandomNums(vector<int>{64, 64}, -0.1, 0.1), "uf2");
// 	Variable&& bf2 = Variable(NumObject(vector<int>{64}, 0.0), "bf2");

// 	Variable&& wi2 = Variable(trunGaussianRandomNums(vector<int>{64, 64}, -0.1, 0.1), "wi2");
// 	Variable&& ui2 = Variable(trunGaussianRandomNums(vector<int>{64, 64}, -0.1, 0.1), "ui2");
// 	Variable&& bi2 = Variable(NumObject(vector<int>{64}, 0.0), "bi2");

// 	Variable&& wo2 = Variable(trunGaussianRandomNums(vector<int>{64, 64}, -0.1, 0.1), "wo2");
// 	Variable&& uo2 = Variable(trunGaussianRandomNums(vector<int>{64, 64}, -0.1, 0.1), "uo2");
// 	Variable&& bo2 = Variable(NumObject(vector<int>{64}, 0.0), "bo2");

// 	Variable&& wc2 = Variable(trunGaussianRandomNums(vector<int>{64, 64}, -0.1, 0.1), "wc2");
// 	Variable&& uc2 = Variable(trunGaussianRandomNums(vector<int>{64, 64}, -0.1, 0.1), "uc2");
// 	Variable&& bc2 = Variable(NumObject(vector<int>{64}, 0.0), "bc2");

// 	Variable&& ow1 = Variable(trunGaussianRandomNums(vector<int>{64, 7}, -0.1, 0.1), "ow1");
// 	Variable&& ob1 = Variable(NumObject(vector<int>{7}, 0.0), "ob1");


// 	Node* f1 = new Sigmoid(new Add(new Add(new MatMul(&xData, &wf1), new MatMul(&h1, &uf1)), &bf1));
// 	Node* i1 = new Sigmoid(new Add(new Add(new MatMul(&xData, &wi1), new MatMul(&h1, &ui1)), &bi1));
// 	Node* o1 = new Sigmoid(new Add(new Add(new MatMul(&xData, &wo1), new MatMul(&h1, &uo1)), &bo1));
// 	Node* tempC1 = new Softsign(new Add(new Add(new MatMul(&xData, &wc1), new MatMul(&h1, &uc1)), &bc1));
// 	Node* newC1 = new Set(new Add(new Multiply(f1, &c1), new Multiply(i1, tempC1)), &c1);
// 	Node* newH1 = new Set(new Multiply(o1, new Softsign(newC1)), &h1);

// 	Node* f2 = new Sigmoid(new Add(new Add(new MatMul(newH1, &wf2), new MatMul(&h2, &uf2)), &bf2));
// 	Node* i2 = new Sigmoid(new Add(new Add(new MatMul(newH1, &wi2), new MatMul(&h2, &ui2)), &bi2));
// 	Node* o2 = new Sigmoid(new Add(new Add(new MatMul(newH1, &wo2), new MatMul(&h2, &uo2)), &bo2));
// 	Node* tempC2 = new Softsign(new Add(new Add(new MatMul(newH1, &wc2), new MatMul(&h2, &uc2)), &bc2));
// 	Node* newC2 = new Set(new Add(new Multiply(f2, &c2), new Multiply(i2, tempC2)), &c2);
// 	Node* newH2 = new Set(new Multiply(o2, new Softsign(newC2)), &h2);

// 	Node* o = new Add(new MatMul(newH2, &ow1), &ob1);

// 	Node* cost = new CrossEntropySoftmax(o, &yData);

// 	GradientDescent trainer = GradientDescent(0.1);

// 	vector<Variable*> variables = {&wf1, &uf1, &bf1, &wi1, &ui1, &bi1, &wo1, &uo1, &bo1, &wc1, &uc1, &bc1, &wf2, &uf2, &bf2, &wi2, &ui2, &bi2, &wo2, &uo2, &bo2, &wc2, &uc2, &bc2, &ow1, &ob1};
// 	vector<Constant*> constants = {&xData, &yData};

// 	for(int i = 0; i < 1001; i++){
// 		h1.value = NumObject(vector<int> {100, 64}, 0.0);
// 		c1.value = NumObject(vector<int> {100, 64}, 0.0);
// 		h2.value = NumObject(vector<int> {100, 64}, 0.0);
// 		c2.value = NumObject(vector<int> {100, 64}, 0.0);

// 		vector<vector<NumObject>> finalData = generateData(100);

// 		NumObject costVal = deriveTime(cost, finalData, constants);
// 		trainer.minimize(variables);

// 		cout << i << ", " << costVal.describe() << endl;
// 	}

// 	cout << "testing..." << endl;

// 	h1.value = NumObject(vector<int> {10000, 64}, 0.0);
// 	c1.value = NumObject(vector<int> {10000, 64}, 0.0);
// 	h2.value = NumObject(vector<int> {10000, 64}, 0.0);
// 	c2.value = NumObject(vector<int> {10000, 64}, 0.0);

// 	vector<vector<NumObject>> testData = generateData(10000);
// 	getValueTime(cost, testData, constants);

// 	double ans = 0.0;
// 	for(int i = 0; i < o->derivativeMemo.size(); i++){
// 		Max a = Max(new Constant(o->derivativeMemo[i]), 1);
// 		Max b = Max(new Constant(yData.derivativeMemo[i]), 1);
// 		a.getValue();
// 		b.getValue();
// 		ans += Mean(new Constant(equal(a.idx[0], b.idx[0]))).getValue().values[0];
// 	}

// 	cout << ans / o->derivativeMemo.size() << endl;
// 	cout << "done testing" << endl;

// 	// NumObject data = NumObject(vector<int>{1, 147});
// 	// data.values = convertToData("ggggyyyybbbbwwwwoooorrrr").values;

// 	// h1.value = NumObject(vector<int> {10000, 2}, 0.0);
// 	// c1.value = NumObject(vector<int> {10000, 2}, 0.0);
// 	// h2.value = NumObject(vector<int> {10000, 2}, 0.0);
// 	// c2.value = NumObject(vector<int> {10000, 2}, 0.0);

// 	// for(int i = 0; i <= 40; i++){
// 	// 	xData.value = data;

// 	// 	NumObject ans = o->getValue(0, 40);

// 	// 	Max tempExpression = Max(new Constant(ans), 1);
// 	// 	tempExpression.getValue();

// 	// 	if(tempExpression.idx[0].values[0] == 0){
// 	// 		data.values = convertToData(left(convertToString(data))).values;
// 	// 		cout << "left" << endl;
// 	// 	}
// 	// 	if(tempExpression.idx[0].values[0] == 1){
// 	// 		data.values = convertToData(right(convertToString(data))).values;
// 	// 		cout << "right" << endl;
// 	// 	}
// 	// 	if(tempExpression.idx[0].values[0] == 2){
// 	// 		data.values = convertToData(up(convertToString(data))).values;
// 	// 		cout << "up" << endl;
// 	// 	}
// 	// 	if(tempExpression.idx[0].values[0] == 3){
// 	// 		data.values = convertToData(down(convertToString(data))).values;
// 	// 		cout << "down" << endl;
// 	// 	}
// 	// 	if(tempExpression.idx[0].values[0] == 4){
// 	// 		data.values = convertToData(rotateLeft(convertToString(data))).values;
// 	// 		cout << "rotate left" << endl;
// 	// 	}
// 	// 	if(tempExpression.idx[0].values[0] == 5){
// 	// 		data.values = convertToData(rotateRight(convertToString(data))).values;
// 	// 		cout << "rotate right" << endl;
// 	// 	}
// 	// 	if(tempExpression.idx[0].values[0] == 6){
// 	// 		cout << "done" << endl;
// 	// 		break;
// 	// 	}
// 	// }
// }


