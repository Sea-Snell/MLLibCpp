#include "Node.hpp"
#include "Math.hpp"
#include "HelperFunctions.hpp"
#include "MatrixMath.hpp"
#include "Optimizers.hpp"
#include "Activations.hpp"
#include "CostFunctions.hpp"
#include "MNISTLoad.hpp"
#include "Regularization.hpp"
#include "TextLoad.hpp"
#include <map>
#include <iostream>
#include <fstream>
#include <dirent.h>

void MNISTFFNN();
vector<vector<double>> getTrain(int n);
vector<vector<double>> getTest(int n);
void linearReg();
void autoEncoderMNISTNN();
void LSTMRapper();

int main(){

	LSTMRapper();
	

	return 0;
}

void LSTMRapper(){
	StepSongs songs = StepSongs();
	vector<double> newSong = songs.getSong("Good Morning.txt");

	int batchSize = 1;

	Constant&& xData = Constant(0.0, "x");
	xData.outDimentions = {batchSize, songs.numChars};
	xData.outRank = 2;
	xData.outSize = batchSize * songs.numChars;

	Constant&& yData = Constant(0.0, "x");
	yData.outDimentions = {batchSize, songs.numChars};
	yData.outRank = 2;
	yData.outSize = batchSize * songs.numChars;

	Constant&& c1 = Constant(0.0, "c1");
	c1.outDimentions = {batchSize, 200};
	c1.outRank = 2;
	c1.outSize = batchSize * 200;
	c1.derivativeMemo.clear();
	c1.derivativeMemo.resize(c1.outSize, 0.0);

	Constant&& h1 = Constant(0.0, "h1");
	h1.outDimentions = {batchSize, 200};
	h1.outRank = 2;
	h1.outSize = batchSize * 200;
	h1.derivativeMemo.clear();
	h1.derivativeMemo.resize(h1.outSize, 0.0);

	Constant&& c2 = Constant(0.0, "c2");
	c2.outDimentions = {batchSize, songs.numChars};
	c2.outRank = 2;
	c2.outSize = batchSize * songs.numChars;
	c2.derivativeMemo.clear();
	c2.derivativeMemo.resize(c2.outSize, 0.0);

	Constant&& h2 = Constant(0.0, "h2");
	h2.outDimentions = {batchSize, songs.numChars};
	h2.outRank = 2;
	h2.outSize = batchSize * songs.numChars;
	h2.derivativeMemo.clear();
	h2.derivativeMemo.resize(h2.outSize, 0.0);




	Variable&& w11 = Variable(trunGaussianRandomNums(vector<int>{songs.numChars, 200}, -0.1, 0.1));
	Variable&& u11 = Variable(trunGaussianRandomNums(vector<int>{songs.numChars, 200}, -0.1, 0.1));
	Variable&& b11 = Variable(vector<double> {0.0}, vector<int> {200});
	b11.derivativeMemo.clear();
	b11.derivativeMemo.resize(200, 0.0);

	Variable&& w21 = Variable(trunGaussianRandomNums(vector<int>{songs.numChars, 200}, -0.1, 0.1));
	Variable&& u21 = Variable(trunGaussianRandomNums(vector<int>{songs.numChars, 200}, -0.1, 0.1));
	Variable&& b21 = Variable(vector<double> {0.0}, vector<int> {200});
	b21.derivativeMemo.clear();
	b21.derivativeMemo.resize(200, 0.0);

	Variable&& w31 = Variable(trunGaussianRandomNums(vector<int>{songs.numChars, 200}, -0.1, 0.1));
	Variable&& u31 = Variable(trunGaussianRandomNums(vector<int>{songs.numChars, 200}, -0.1, 0.1));
	Variable&& b31 = Variable(vector<double> {0.0}, vector<int> {200});
	b31.derivativeMemo.clear();
	b31.derivativeMemo.resize(200, 0.0);

	Variable&& w41 = Variable(trunGaussianRandomNums(vector<int>{songs.numChars, 200}, -0.1, 0.1));
	Variable&& u41 = Variable(trunGaussianRandomNums(vector<int>{songs.numChars, 200}, -0.1, 0.1));
	Variable&& b41 = Variable(vector<double> {0.0}, vector<int> {200});
	b41.derivativeMemo.clear();
	b41.derivativeMemo.resize(200, 0.0);



	Variable&& w12 = Variable(trunGaussianRandomNums(vector<int>{200, songs.numChars}, -0.1, 0.1));
	Variable&& u12 = Variable(trunGaussianRandomNums(vector<int>{200, songs.numChars}, -0.1, 0.1));
	Variable&& b12 = Variable(vector<double> {0.0}, vector<int> {songs.numChars});
	b12.derivativeMemo.clear();
	b12.derivativeMemo.resize(songs.numChars, 0.0);

	Variable&& w22 = Variable(trunGaussianRandomNums(vector<int>{200, songs.numChars}, -0.1, 0.1));
	Variable&& u22 = Variable(trunGaussianRandomNums(vector<int>{200, songs.numChars}, -0.1, 0.1));
	Variable&& b22 = Variable(vector<double> {0.0}, vector<int> {songs.numChars});
	b22.derivativeMemo.clear();
	b22.derivativeMemo.resize(songs.numChars, 0.0);

	Variable&& w32 = Variable(trunGaussianRandomNums(vector<int>{200, songs.numChars}, -0.1, 0.1));
	Variable&& u32 = Variable(trunGaussianRandomNums(vector<int>{200, songs.numChars}, -0.1, 0.1));
	Variable&& b32 = Variable(vector<double> {0.0}, vector<int> {songs.numChars});
	b32.derivativeMemo.clear();
	b32.derivativeMemo.resize(songs.numChars, 0.0);

	Variable&& w42 = Variable(trunGaussianRandomNums(vector<int>{200, songs.numChars}, -0.1, 0.1));
	Variable&& u42 = Variable(trunGaussianRandomNums(vector<int>{200, songs.numChars}, -0.1, 0.1));
	Variable&& b42 = Variable(vector<double> {0.0}, vector<int> {songs.numChars});
	b42.derivativeMemo.clear();
	b42.derivativeMemo.resize(songs.numChars, 0.0);

	Variable&& v = Variable(trunGaussianRandomNums(vector<int>{songs.numChars, songs.numChars}, -0.1, 0.1));
	Variable&& vb = Variable(vector<double> {0.0}, vector<int> {songs.numChars});
	vb.derivativeMemo.clear();
	vb.derivativeMemo.resize(songs.numChars, 0.0);


	Node* f1 = new Sigmoid(new Add(new Add(new MatMul(&xData, &w11, 1, 0), new MatMul(&h1, &u11, 1, 0)), &b11));
	Node* i1 = new Sigmoid(new Add(new Add(new MatMul(&xData, &w21, 1, 0), new MatMul(&h1, &u21, 1, 0)), &b21));
	Node* o1 = new Sigmoid(new Add(new Add(new MatMul(&xData, &w31, 1, 0), new MatMul(&h1, &u31, 1, 0)), &b31));
	Node* tempC1 = new TanH(new Add(new Add(new MatMul(&xData, &w41, 1, 0), new MatMul(&h1, &u41, 1, 0)), &b41));

	Node* newC1 = new Add(new Multiply(f1, &c1), new Multiply(i1, tempC1));
	Node* newH1 = new Multiply(o1, new TanH(newC1));


	Node* f2 = new Sigmoid(new Add(new Add(new MatMul(newH1, &w12, 1, 0), new MatMul(&h2, &u12, 1, 0)), &b12));
	Node* i2 = new Sigmoid(new Add(new Add(new MatMul(newH1, &w22, 1, 0), new MatMul(&h2, &u22, 1, 0)), &b22));
	Node* o2 = new Sigmoid(new Add(new Add(new MatMul(newH1, &w32, 1, 0), new MatMul(&h2, &u32, 1, 0)), &b32));
	Node* tempC2 = new TanH(new Add(new Add(new MatMul(newH1, &w42, 1, 0), new MatMul(&h2, &u42, 1, 0)), &b42));

	Node* newC2 = new Add(new Multiply(f2, &c2), new Multiply(i2, tempC2));
	Node* newH2 = new Multiply(o2, new TanH(newC2));

	Node* outVal = new Add(new MatMul(newH2, &v, 1, 0), &vb);





	Node* cost = new CrossEntropySoftmax(outVal, &yData);

	initalize(cost);

	GradientDescent trainer = GradientDescent(0.001);

	vector<Variable*> variables = {&v, &vb};
	vector<Variable*> noClear1 = {&u11, &u21, &u31, &u41, &w11, &b11, &w21, &b21, &w31, &b31, &w41, &b41};
	vector<Variable*> noClear2 = {&u12, &u22, &u32, &u42, &w12, &b12, &w22, &b22, &w32, &b32, &w42, &b42};

	for(int i = 0; i < 100; i++){
		// batchSize = 1;

		// xData.outDimentions = {batchSize, songs.numChars};
		// xData.outRank = 2;
		// xData.outSize = batchSize * songs.numChars;

		// c.outDimentions = {batchSize, songs.numChars};
		// c.outRank = 2;
		// c.outSize = batchSize * songs.numChars;
		// c.derivativeMemo.clear();
		// c.derivativeMemo.resize(c.outSize, 0.0);

		// songs.getRandomSet(batchSize);
		// xData.derivativeMemo.clear();
		// xData.derivativeMemo.resize(xData.outSize, 0.0);
		// yData.derivativeMemo = songs.getNext();









		c1.derivativeMemo.clear();
		c1.derivativeMemo.resize(c1.outSize, 0.0);
		h1.derivativeMemo.clear();
		h1.derivativeMemo.resize(h1.outSize, 0.0);
		c2.derivativeMemo.clear();
		c2.derivativeMemo.resize(c2.outSize, 0.0);
		h2.derivativeMemo.clear();
		h2.derivativeMemo.resize(h2.outSize, 0.0);
		xData.derivativeMemo.clear();
		xData.derivativeMemo.resize(xData.outSize, 0.0);
		yData.derivativeMemo.clear();
		yData.derivativeMemo.resize(yData.outSize, 0.0);

		vector<vector<double>> history1;
		vector<vector<double>> history2;

		for(int x = 0; x < noClear1.size(); x++){
			noClear1[x]->derivative.clear();
			noClear1[x]->derivative.resize(noClear1[x]->outSize, 0.0);

			vector<double> temp1;
			temp1.resize(noClear1[x]->outSize, 0.0);
			history1.push_back(temp1);
		}

		for(int x = 0; x < noClear2.size(); x++){
			noClear2[x]->derivative.clear();
			noClear2[x]->derivative.resize(noClear2[x]->outSize, 0.0);

			vector<double> temp2;
			temp2.resize(noClear2[x]->outSize, 0.0);
			history2.push_back(temp2);
		}

		for(int z = 1; z < newSong.size() / songs.numChars; z++){
			vector<vector<double>> gradient;

			for(int x = 0; x < songs.numChars; x++){
				xData.derivativeMemo[x] = newSong[songs.numChars * (z - 1) + x];
				yData.derivativeMemo[x] = newSong[songs.numChars * z + x];
			}

			// trainer.minimize(cost, variables, noClear);
			cost->getValue();
			



			if(z % 50 == 0){
				cout << getValue(cost).describe() << endl;
			}

			c1.derivativeMemo = newC1->derivativeMemo;
			h1.derivativeMemo = newH1->derivativeMemo;
			c2.derivativeMemo = newC2->derivativeMemo;
			h2.derivativeMemo = newH2->derivativeMemo;
		}

		cout << "done" << endl;





		vector<double> data;
		string seed = "[Produced by Kanye West]\n\n[Intro]\nUhh.. uhh\nUhh.. uhh\nGood morning!\nGood morning!\nGood morning!\nGood morning!](4905654)\n\n[Verse 1]\nWake up Mr. West, Mr. West";
		// batchSize = 1;

		// xData.outDimentions = {batchSize, songs.numChars};
		// xData.outRank = 2;
		// xData.outSize = batchSize * songs.numChars;

		// c.outDimentions = {batchSize, songs.numChars};
		// c.outRank = 2;
		// c.outSize = batchSize * songs.numChars;
		// c.derivativeMemo.clear();
		// c.derivativeMemo.resize(c.outSize, 0.0);


		c1.derivativeMemo.clear();
		c1.derivativeMemo.resize(c1.outSize, 0.0);
		h1.derivativeMemo.clear();
		h1.derivativeMemo.resize(h1.outSize, 0.0);
		c2.derivativeMemo.clear();
		c2.derivativeMemo.resize(c2.outSize, 0.0);
		h2.derivativeMemo.clear();
		h2.derivativeMemo.resize(h2.outSize, 0.0);
		xData.derivativeMemo.clear();
		xData.derivativeMemo.resize(xData.outSize, 0.0);
		xData.derivativeMemo = songs.charToVector(seed[0]);
		for(int x = 0; x < xData.outSize; x++){
			data.push_back(xData.derivativeMemo[x]);
		}

		for(int i = 1; i < seed.size(); i++){
			outVal->getValue();

			xData.derivativeMemo = songs.charToVector(seed[i]);
			c1.derivativeMemo = newC1->derivativeMemo;
			h1.derivativeMemo = newH1->derivativeMemo;
			c2.derivativeMemo = newC2->derivativeMemo;
			h2.derivativeMemo = newH2->derivativeMemo;
			for(int x = 0; x < xData.outSize; x++){
				data.push_back(xData.derivativeMemo[x]);
			}
		}

		int loopCount = 0;
		vector<double> tempData;
		tempData.clear();
		tempData.resize(songs.numChars, 0.0);

		while(tempData[songs.numChars - 1] != 1.0 && loopCount < 1000){
			Constant results = getValue(outVal);

			Max tempExpression = Max(&results, 1);
			initalize(&tempExpression);
			tempExpression.getValue();
			for(int i = 0; i < songs.numChars; i++){
				if(tempExpression.idx[0] == i){
					tempData[i] = 1.0;
				}
				else{
					tempData[i] = 0.0;
				}
			}

			xData.derivativeMemo = tempData;
			c1.derivativeMemo = newC1->derivativeMemo;
			h1.derivativeMemo = newH1->derivativeMemo;
			c2.derivativeMemo = newC2->derivativeMemo;
			h2.derivativeMemo = newH2->derivativeMemo;
			for(int x = 0; x < xData.outSize; x++){
				data.push_back(xData.derivativeMemo[x]);
			}

			loopCount += 1;
		}

		songs.placeSong(data, "test" + to_string(i) + ".txt");
	}
}

void autoEncoderMNISTNN(){
	Constant&& xData = Constant(0.0, "x");

	xData.outDimentions = {100, 784};
	xData.outRank = 2;
	xData.outSize = 78400;

	Variable&& weights1 = Variable(gaussianRandomNums(vector<int>{784, 100}, 0.0, 1.0 / sqrt(784.0)).derivativeMemo, vector<int> {784, 100}, "w1");
	Variable&& weights2 = Variable(gaussianRandomNums(vector<int>{100, 784}, 0.0, 1.0 / sqrt(100.0)).derivativeMemo, vector<int> {100, 784}, "w2");
	Variable&& bias1 = Variable(gaussianRandomNums(vector<int>{100}, 0.0, 1.0).derivativeMemo, vector<int> {100}, "bias1");
	Variable&& bias2 = Variable(gaussianRandomNums(vector<int>{784}, 0.0, 1.0).derivativeMemo, vector<int> {784}, "bias2");

	Node* layer1 = new Gate(new ReLU(new Add(new MatMul(&xData, &weights1, 1, 0), &bias1)));
	Node* layer2 = new TanH(new Add(new MatMul(layer1, &weights2, 1, 0), &bias2));

	Node* cost = new MeanSquared(layer2, &xData);

	initalize(cost);

	GradientDescent trainer = GradientDescent(0.001);

	vector<Variable*> variables = {&weights1, &bias1, &weights2, &bias2};
	vector<Variable*> noClear;


	cout << "pre-train begin." << endl;

	for(int i = 0; i < 1001; i++){
		vector<vector<double>> trainData = getTrain(100);
		xData.derivativeMemo = trainData[0];

		trainer.minimize(cost, variables, noClear);

		if(i % 10 == 0){
			cout << getValue(cost).describe() << endl;
		}
	}


	cout << "pre-train done." << endl;

	
	Constant&& yData = Constant(0.0, "y");

	yData.outDimentions = {100, 10};
	yData.outRank = 2;
	yData.outSize = 1000;

	Variable&& weights3 = Variable(gaussianRandomNums(vector<int>{100, 10}, 0.0, 1.0 / sqrt(100.0)).derivativeMemo, vector<int> {100, 10}, "w3");
	Variable&& bias3 = Variable(gaussianRandomNums(vector<int>{10}, 0.0, 1.0).derivativeMemo, vector<int> {10}, "bias3");

	dynamic_cast<Gate*>(layer1)->closed = true;

	Node* layer3 = new Add(new MatMul(layer1, &weights3, 1, 0), &bias3);

	Node* cost2 = new CrossEntropySoftmax(layer3, &yData);

	initalize(cost2);

	GradientDescent trainer2 = GradientDescent(0.05);

	vector<Variable*> variables2 = {&weights1, &bias1, &weights3, &bias3};


	for(int i = 0; i < 1001; i++){
		vector<vector<double>> trainData = getTrain(100);
		xData.derivativeMemo = trainData[0];
		yData.derivativeMemo = trainData[1];

		trainer2.minimize(cost2, variables2, noClear);

		if(i % 10 == 0){
			cout << getValue(cost2).describe() << endl;
		}
	}

	vector<vector<double>> testData = getTest(1000);
	xData.derivativeMemo = testData[0];
	yData.derivativeMemo = testData[1];

	xData.outDimentions = {1000, 784};
	xData.outRank = 2;
	xData.outSize = 784000;

	yData.outDimentions = {1000, 10};
	yData.outRank = 2;
	yData.outSize = 10000;

	initalize(layer3);

	Constant prediction = getValue(layer3);

	Max a = Max(&prediction, 1);
	Max b = Max(&yData, 1);
	initalize(&a);
	initalize(&b);

	a.getValue();
	b.getValue();

	Constant temp1 = Constant(vector<double> (a.idx.begin(), a.idx.end()), vector<int> {100});
	Constant temp2 = Constant(vector<double> (b.idx.begin(), b.idx.end()), vector<int> {100});
	Constant temp3 = equal(temp1, temp2);

	Node* avg = new Mean(&temp3);
	initalize(avg);
	cout << getValue(avg).describe() << endl;


}

void MNISTFFNN(){
	Constant&& xData = Constant(0.0, "x");
	Constant&& yData = Constant(0.0, "y");

	xData.outDimentions = {100, 784};
	xData.outRank = 2;
	xData.outSize = 78400;

	yData.outDimentions = {100, 10};
	yData.outRank = 2;
	yData.outSize = 1000;

	Variable&& weights1 = Variable(gaussianRandomNums(vector<int>{784, 40}, 0.0, 1.0 / sqrt(784.0)).derivativeMemo, vector<int> {784, 40}, "w1");
	Variable&& weights2 = Variable(gaussianRandomNums(vector<int>{40, 10}, 0.0, 1.0 / sqrt(40.0)).derivativeMemo, vector<int> {40, 10}, "w2");
	Variable&& bias1 = Variable(gaussianRandomNums(vector<int>{40}, 0.0, 1.0).derivativeMemo, vector<int> {40}, "bias1");
	Variable&& bias2 = Variable(gaussianRandomNums(vector<int>{10}, 0.0, 1.0).derivativeMemo, vector<int> {10}, "bias2");

	Node* layer1 = new ReLU(new Add(new MatMul(&xData, &weights1, 1, 0), &bias1));
	Node* layer2 = new Add(new MatMul(layer1, &weights2, 1, 0), &bias2);

	Node* cost = new CrossEntropySoftmax(layer2, &yData);

	initalize(cost);

	GradientDescent trainer = GradientDescent(0.1);

	vector<Variable*> variables = {&weights1, &bias1, &weights2, &bias2};

	vector<Variable*> noClear;

	for(int i = 0; i < 201; i++){
		vector<vector<double>> trainData = getTrain(100);
		xData.derivativeMemo = trainData[0];
		yData.derivativeMemo = trainData[1];

		trainer.minimize(cost, variables, noClear);

		if(i % 10 == 0){
			cout << getValue(cost).describe() << endl;
		}
	}

	vector<vector<double>> testData = getTest(1000);
	xData.derivativeMemo = testData[0];
	yData.derivativeMemo = testData[1];

	xData.outDimentions = {1000, 784};
	xData.outRank = 2;
	xData.outSize = 784000;

	yData.outDimentions = {1000, 10};
	yData.outRank = 2;
	yData.outSize = 10000;

	initalize(layer2);

	Constant prediction = getValue(layer2);

	Max a = Max(&prediction, 1);
	Max b = Max(&yData, 1);
	initalize(&a);
	initalize(&b);

	a.getValue();
	b.getValue();

	Constant temp1 = Constant(vector<double> (a.idx.begin(), a.idx.end()), vector<int> {100});
	Constant temp2 = Constant(vector<double> (b.idx.begin(), b.idx.end()), vector<int> {100});
	Constant temp3 = equal(temp1, temp2);

	Node* avg = new Mean(&temp3);
	initalize(avg);
	cout << getValue(avg).describe() << endl;
}

vector<vector<double>> getTrain(int n){
	vector<vector<double>> temp = randomTrainSet(n);

	for(int i = 0; i < n * 28 * 28; i++){
		temp[0][i] = (temp[0][i] - 128.0) / 256.0;
	}

	vector<double> yFinal = oneHot(Constant(temp[1], vector<int> {n}), 0, 9).derivativeMemo;

	vector<vector<double>> ans = {temp[0], yFinal};

	return ans;
}

vector<vector<double>> getTest(int n){
	vector<vector<double>> temp = randomTestSet(n);
	
	for(int i = 0; i < n * 28 * 28; i++){
		temp[0][i] = (temp[0][i] - 128.0) / 256.0;
	}

	vector<double> yFinal = oneHot(Constant(temp[1], vector<int> {n}), 0, 9).derivativeMemo;

	vector<vector<double>> ans = {temp[0], yFinal};

	return ans;
}

void linearReg(){
	Constant&& xData = uniformRandomNums(vector<int>{100, 101}, -100.0, 100.0);
	for(int i = 0; i < xData.outDimentions[0]; i++){
		xData.derivativeMemo[i * xData.outDimentions[0] + 100] = 1.0;
	}

	Node* temp = new Sum(new Multiply(&xData, new Constant(0.5)), 1);
	initalize(temp);
	Constant&& yData = getValue(temp);

	Variable&& weights = Variable(uniformRandomNums(vector<int>{101}, -0.5, 0.5));

	Node* hypothesis = new MatMul(&xData, &weights, 1, 0);
	Node* cost = new MeanSquared(hypothesis, &yData);

	GradientDescent trainer = GradientDescent(0.00001);

	vector<Variable*> variables = {&weights};
	vector<Variable*> noClear;

	initalize(cost);

	for(int i = 0; i < 100; i++){
		trainer.minimize(cost, variables, noClear);

		if(i % 10 == 0){
			cout << getValue(cost).describe() << endl;
		}
	}

	cout << weights.describe() << endl;
}