#include "Node.hpp"
#include "Math.hpp"
#include "MapVals.hpp"
#include "HelperFunctions.hpp"
#include "MatrixMath.hpp"
#include "Optimizers.hpp"
#include "Activations.hpp"
#include "CostFunctions.hpp"
#include "MNISTLoad.hpp"
#include "Regularization.hpp"

void MNISTFFNN();
vector<NumObject> getTrain(int n);
vector<NumObject> getTest(int n);
void linearReg();

int main(){
	
	MNISTFFNN();


	return 0;
}

void MNISTFFNN(){
	Constant&& xData = Constant(NumObject(), "x");
	Constant&& yData = Constant(NumObject(), "y");
	Variable&& weights1 = Variable(randomNums(vector<int>{784, 40}, 0.0, 1.0 / sqrt(784.0)), "w1");
	Variable&& weights2 = Variable(randomNums(vector<int>{40, 10}, 0.0, 1.0 / sqrt(40.0)), "w2");
	Variable&& bias1 = Variable(randomNums(vector<int>{40}, 0.0, 1.0), "bias1");
	Variable&& bias2 = Variable(randomNums(vector<int>{10}, 0.0, 1.0), "bias2");

	Node* layer1 = new ReLU(new Add(new MatMul(&xData, &weights1, 1, 0), &bias1));
	Node* layer2 = new Add(new MatMul(layer1, &weights2, 1, 0), &bias2);

	Node* cost = new CrossEntropySoftmax(layer2, &yData);

	GradientDescent trainer = GradientDescent(0.1);

	vector<Variable*> variables = {&weights1, &bias1, &weights2, &bias2};

	for(int i = 0; i < 201; i++){

		vector<NumObject> trainData = getTrain(100);
		xData.value = trainData[0];
		yData.value = trainData[1];

		trainer.minimize(cost, variables);

		if(i % 10 == 0){
			cout << cost->getValue().describe() << endl;
		}
	}

	vector<NumObject> testData = getTest(1000);
	xData.value = testData[0];
	yData.value = testData[1];

	NumObject prediction = layer2->getValue();

	Max a = Max(new Constant(prediction), 1);
	Max b = Max(&yData, 1);
	a.getValue();
	b.getValue();

	cout << Mean(new Constant(equal(a.idx, b.idx))).getValue().describe() << endl;
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
	Constant&& xData = Constant(randomNums(vector<int>{100, 101}, -100.0, 100.0));
	for(int i = 0; i < xData.value.dimentions[0]; i++){
		xData.value.values[i * xData.value.dimentions[1] + 100] = 1.0;
	}
	Constant&& yData = Constant(Add(new Sum(new Multiply(&xData, new Constant(NumObject(0.5))), 1), new Constant(NumObject(0.5))).getValue());
	Variable&& weights = Variable(randomNums(vector<int>{101}, -0.5, 0.5));

	Node* hypothesis = new MatMul(&xData, &weights, 1, 0);
	Node* cost = new MeanSquared(hypothesis, &yData);

	GradientDescent trainer = GradientDescent(0.00001);

	vector<Variable*> variables = {&weights};
	for(int i = 0; i < 100; i++){
		trainer.minimize(cost, variables);

		if(i % 10 == 0){
			cout << cost->getValue().describe() << endl;
		}
	}

	cout << weights.describe() << endl;
}