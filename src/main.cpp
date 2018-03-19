#include "Node.hpp"
#include "Math.hpp"
#include "HelperFunctions.hpp"
#include "MatrixMath.hpp"
#include "Optimizers.hpp"
#include "CostFunctions.hpp"
#include <time.h>

// void rubixNet();
// void WordRNN(bool randomize);
// void MNISTFFNN();
// vector<NumObject> getTrain(int n);
// vector<NumObject> getTest(int n);
void linearReg();

int main(){

	initialize();

	linearReg();

	return 0;
}

void linearReg(){
	Constant&& xData = Constant(gaussianRandomNums(vector<int>{100, 100}, -10.0, 10.0));
	for(int i = 0; i < xData.value.dimentions[0]; i++){
		xData.value.values[i * xData.value.dimentions[1] + 99] = 1.0;
	}

	Node* yValExpression = new Sum(new Multiply(&xData, new Constant(NumObject(0.5))), 1);
	initalize(yValExpression);
	Constant&& yData = Constant(getValue(yValExpression));
	clearHistory(&xData);

	Variable&& weights = Variable(gaussianRandomNums(vector<int>{100}, -0.5, 0.5));

	Node* hypothesis = new MatMul(&xData, &weights);
	Node* cost = new MeanSquared(hypothesis, &yData);

	initalize(cost);

	vector<Variable*> variables = {&weights};

	cout << "starting..." << endl;
	int start = clock();
	for(int i = 0; i < 1000; i++){
		derive(cost);
		gradientDescent(variables, 0.00000001);

		if(i % 10 == 0){
			cout << i << ", " << showValue(cost).describe() << endl;
		}
	}
	cout << showValue(cost).describe() << endl;
	int end = clock();
	cout <<  "Time: " << (end - start) / double(CLOCKS_PER_SEC) << endl;
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

// void WordRNN(bool randomize){
// 	StepSongs songs = StepSongs();

// 	Constant&& xData = Constant(NumObject(), "x");
// 	Constant&& yData = Constant(NumObject(), "y");

// 	Variable&& h1 = Variable(NumObject(), "h1");
// 	Variable&& c1 = Variable(NumObject(), "c1");

// 	Variable&& h2 = Variable(NumObject(), "h2");
// 	Variable&& c2 = Variable(NumObject(), "c2");


// 	Variable&& wf1 = Variable(NumObject());
// 	Variable&& uf1 = Variable(NumObject());
// 	Variable&& bf1 = Variable(NumObject());

// 	Variable&& wi1 = Variable(NumObject());
// 	Variable&& ui1 = Variable(NumObject());
// 	Variable&& bi1 = Variable(NumObject());

// 	Variable&& wo1 = Variable(NumObject());
// 	Variable&& uo1 = Variable(NumObject());
// 	Variable&& bo1 = Variable(NumObject());

// 	Variable&& wc1 = Variable(NumObject());
// 	Variable&& uc1 = Variable(NumObject());
// 	Variable&& bc1 = Variable(NumObject());


// 	Variable&& wf2 = Variable(NumObject());
// 	Variable&& uf2 = Variable(NumObject());
// 	Variable&& bf2 = Variable(NumObject());

// 	Variable&& wi2 = Variable(NumObject());
// 	Variable&& ui2 = Variable(NumObject());
// 	Variable&& bi2 = Variable(NumObject());

// 	Variable&& wo2 = Variable(NumObject());
// 	Variable&& uo2 = Variable(NumObject());
// 	Variable&& bo2 = Variable(NumObject());

// 	Variable&& wc2 = Variable(NumObject());
// 	Variable&& uc2 = Variable(NumObject());
// 	Variable&& bc2 = Variable(NumObject());


// 	Variable&& ow1 = Variable(NumObject());
// 	Variable&& ob1 = Variable(NumObject());


// 	if(randomize == true){
// 		wf1 = Variable(trunGaussianRandomNums(vector<int>{songs.numChars, 512}, -0.1, 0.1), "wf1");
// 		uf1 = Variable(trunGaussianRandomNums(vector<int>{512, 512}, -0.1, 0.1), "uf1");
// 		bf1 = Variable(gaussianRandomNums(vector<int>{512}, 0.0, 1.0), "bf1");

// 		wi1 = Variable(trunGaussianRandomNums(vector<int>{songs.numChars, 512}, -0.1, 0.1), "wi1");
// 		ui1 = Variable(trunGaussianRandomNums(vector<int>{512, 512}, -0.1, 0.1), "ui1");
// 		bi1 = Variable(gaussianRandomNums(vector<int>{512}, 0.0, 1.0), "bi1");

// 		wo1 = Variable(trunGaussianRandomNums(vector<int>{songs.numChars, 512}, -0.1, 0.1), "wo1");
// 		uo1 = Variable(trunGaussianRandomNums(vector<int>{512, 512}, -0.1, 0.1), "uo1");
// 		bo1 = Variable(gaussianRandomNums(vector<int>{512}, 0.0, 1.0), "bo1");

// 		wc1 = Variable(trunGaussianRandomNums(vector<int>{songs.numChars, 512}, -0.1, 0.1), "wc1");
// 		uc1 = Variable(trunGaussianRandomNums(vector<int>{512, 512}, -0.1, 0.1), "uc1");
// 		bc1 = Variable(gaussianRandomNums(vector<int>{512}, 0.0, 1.0), "bc1");

// 		wf2 = Variable(trunGaussianRandomNums(vector<int>{512, 512}, -0.1, 0.1), "wf2");
// 		uf2 = Variable(trunGaussianRandomNums(vector<int>{512, 512}, -0.1, 0.1), "uf2");
// 		bf2 = Variable(NumObject(vector<int>{512}, 0.0), "bf2");

// 		wi2 = Variable(trunGaussianRandomNums(vector<int>{512, 512}, -0.1, 0.1), "wi2");
// 		ui2 = Variable(trunGaussianRandomNums(vector<int>{512, 512}, -0.1, 0.1), "ui2");
// 		bi2 = Variable(NumObject(vector<int>{512}, 0.0), "bi2");

// 		wo2 = Variable(trunGaussianRandomNums(vector<int>{512, 512}, -0.1, 0.1), "wo2");
// 		uo2 = Variable(trunGaussianRandomNums(vector<int>{512, 512}, -0.1, 0.1), "uo2");
// 		bo2 = Variable(NumObject(vector<int>{512}, 0.0), "bo2");

// 		wc2 = Variable(trunGaussianRandomNums(vector<int>{512, 512}, -0.1, 0.1), "wc2");
// 		uc2 = Variable(trunGaussianRandomNums(vector<int>{512, 512}, -0.1, 0.1), "uc2");
// 		bc2 = Variable(NumObject(vector<int>{512}, 0.0), "bc2");

// 		ow1 = Variable(trunGaussianRandomNums(vector<int>{512, songs.numChars}, -0.1, 0.1), "ow1");
// 		ob1 = Variable(NumObject(vector<int>{songs.numChars}, 0.0), "ob1");
// 	}
// 	else{
// 		wf1 = Variable(loadData("../Weights/wf1.txt"), "wf1");
// 		uf1 = Variable(loadData("../Weights/uf1.txt"), "uf1");
// 		bf1 = Variable(loadData("../Weights/bf1.txt"), "bf1");

// 		wi1 = Variable(loadData("../Weights/wi1.txt"), "wi1");
// 		ui1 = Variable(loadData("../Weights/ui1.txt"), "ui1");
// 		bi1 = Variable(loadData("../Weights/bi1.txt"), "bi1");

// 		wo1 = Variable(loadData("../Weights/wo1.txt"), "wo1");
// 		uo1 = Variable(loadData("../Weights/uo1.txt"), "uo1");
// 		bo1 = Variable(loadData("../Weights/bo1.txt"), "bo1");

// 		wc1 = Variable(loadData("../Weights/wc1.txt"), "wc1");
// 		uc1 = Variable(loadData("../Weights/uc1.txt"), "uc1");
// 		bc1 = Variable(loadData("../Weights/bc1.txt"), "bc1");

// 		wf2 = Variable(loadData("../Weights/wf2.txt"), "wf2");
// 		uf2 = Variable(loadData("../Weights/uf2.txt"), "uf2");
// 		bf2 = Variable(loadData("../Weights/bf2.txt"), "bf2");

// 		wi2 = Variable(loadData("../Weights/wi2.txt"), "wi2");
// 		ui2 = Variable(loadData("../Weights/ui2.txt"), "ui2");
// 		bi2 = Variable(loadData("../Weights/bi2.txt"), "bi2");

// 		wo2 = Variable(loadData("../Weights/wo2.txt"), "wo2");
// 		uo2 = Variable(loadData("../Weights/uo2.txt"), "uo2");
// 		bo2 = Variable(loadData("../Weights/bo2.txt"), "bo2");

// 		wc2 = Variable(loadData("../Weights/wc2.txt"), "wc2");
// 		uc2 = Variable(loadData("../Weights/uc2.txt"), "uc2");
// 		bc2 = Variable(loadData("../Weights/bc2.txt"), "bc2");

// 		ow1 = Variable(loadData("../Weights/ow1.txt"), "ow1");
// 		ob1 = Variable(loadData("../Weights/ob1.txt"), "ob1");
// 	}

// 	Node* f1 = new Sigmoid(new Add(new Add(new MatMul(&xData, &wf1), new MatMul(&h1, &uf1)), &bf1));
// 	Node* i1 = new Sigmoid(new Add(new Add(new MatMul(&xData, &wi1), new MatMul(&h1, &ui1)), &bi1));
// 	Node* o1 = new Sigmoid(new Add(new Add(new MatMul(&xData, &wo1), new MatMul(&h1, &uo1)), &bo1));
// 	Node* tempC1 = new Softsign(new Add(new Add(new MatMul(&xData, &wc1), new MatMul(&h1, &uc1)), &bc1));
// 	Node* newC1 = new Set(new Add(new Multiply(f1, &c1), new Multiply(i1, tempC1)), &c1);
// 	Node* newH1 = new Dropout(new Set(new Multiply(o1, new Softsign(newC1)), &h1), 1, 0.5);

// 	Node* f2 = new Sigmoid(new Add(new Add(new MatMul(newH1, &wf2), new MatMul(&h2, &uf2)), &bf2));
// 	Node* i2 = new Sigmoid(new Add(new Add(new MatMul(newH1, &wi2), new MatMul(&h2, &ui2)), &bi2));
// 	Node* o2 = new Sigmoid(new Add(new Add(new MatMul(newH1, &wo2), new MatMul(&h2, &uo2)), &bo2));
// 	Node* tempC2 = new Softsign(new Add(new Add(new MatMul(newH1, &wc2), new MatMul(&h2, &uc2)), &bc2));
// 	Node* newC2 = new Set(new Add(new Multiply(f2, &c2), new Multiply(i2, tempC2)), &c2);
// 	Node* newH2 = new Dropout(new Set(new Multiply(o2, new Softsign(newC2)), &h2), 1, 0.5);


// 	Node* o = new Add(new MatMul(newH2, &ow1), &ob1);

// 	Node* cost = new CrossEntropySoftmax(o, &yData);

// 	GradientDescent trainer = GradientDescent(0.2);

// 	vector<Variable*> variables = {&wf1, &uf1, &bf1, &wi1, &ui1, &bi1, &wo1, &uo1, &bo1, &wc1, &uc1, &bc1, &wf2, &uf2, &bf2, &wi2, &ui2, &bi2, &wo2, &uo2, &bo2, &wc2, &uc2, &bc2, &ow1, &ob1};
// 	vector<Constant*> constants = {&xData, &yData};

// 	int i = 0;

// 	cout << "running loop." << endl;

// 	while(true){

// 		h1.value = NumObject(vector<int> {10, 512}, 0.0);
// 		c1.value = NumObject(vector<int> {10, 512}, 0.0);
// 		h2.value = NumObject(vector<int> {10, 512}, 0.0);
// 		c2.value = NumObject(vector<int> {10, 512}, 0.0);

// 		dynamic_cast<Dropout*>(newH1)->training = true;
// 		dynamic_cast<Dropout*>(newH2)->training = true;

// 		songs.randomize(10);

// 		while(songs.finished == false){
// 			vector<NumObject> tempSong = songs.getRandomSet(50);
// 			vector<NumObject> xVals;
// 			vector<NumObject> yVals;
// 			xVals.push_back(NumObject(vector<int>{10, songs.numChars}, 0.0));
// 			for(int i = 0; i < tempSong.size() - 1; i++){
// 				xVals.push_back(tempSong[i]);
// 				yVals.push_back(tempSong[i]);
// 			}
// 			yVals.push_back(tempSong[tempSong.size() - 1]);

// 			vector<vector<NumObject>> finalData = {xVals, yVals};


// 			NumObject costVal = deriveTime(cost, finalData, constants);
// 			trainer.minimize(variables);

// 			cout << i << ", " << costVal.describe() << endl;
			
// 			if(i % 10 == 0){
// 				cout << "start save" << endl;
// 				saveData(wf1.value, "../Weights/wf1.txt");
// 				saveData(uf1.value, "../Weights/uf1.txt");
// 				saveData(bf1.value, "../Weights/bf1.txt");

// 				saveData(wi1.value, "../Weights/wi1.txt");
// 				saveData(ui1.value, "../Weights/ui1.txt");
// 				saveData(bi1.value, "../Weights/bi1.txt");

// 				saveData(wo1.value, "../Weights/wo1.txt");
// 				saveData(uo1.value, "../Weights/uo1.txt");
// 				saveData(bo1.value, "../Weights/bo1.txt");

// 				saveData(wc1.value, "../Weights/wc1.txt");
// 				saveData(uc1.value, "../Weights/uc1.txt");
// 				saveData(bc1.value, "../Weights/bc1.txt");

// 				saveData(wf2.value, "../Weights/wf2.txt");
// 				saveData(uf2.value, "../Weights/uf2.txt");
// 				saveData(bf2.value, "../Weights/bf2.txt");

// 				saveData(wi2.value, "../Weights/wi2.txt");
// 				saveData(ui2.value, "../Weights/ui2.txt");
// 				saveData(bi2.value, "../Weights/bi2.txt");

// 				saveData(wo2.value, "../Weights/wo2.txt");
// 				saveData(uo2.value, "../Weights/uo2.txt");
// 				saveData(bo2.value, "../Weights/bo2.txt");

// 				saveData(wc2.value, "../Weights/wc2.txt");
// 				saveData(uc2.value, "../Weights/uc2.txt");
// 				saveData(bc2.value, "../Weights/bc2.txt");

// 				saveData(ow1.value, "../Weights/ow1.txt");
// 				saveData(ob1.value, "../Weights/ob1.txt");
// 				cout << "end save" << endl;
// 			}
// 			i += 1;
// 		}

// 		cout << "start writing." << endl;

// 		vector<NumObject> data;
// 		int loopCount = 0;

// 		dynamic_cast<Dropout*>(newH1)->training = false;
// 		dynamic_cast<Dropout*>(newH2)->training = false;

// 		xData.value = NumObject(vector<int>{1, songs.numChars}, 0.0);
// 		h1.value = NumObject(vector<int> {1, 512}, 0.0);
// 		c1.value = NumObject(vector<int> {1, 512}, 0.0);
// 		h2.value = NumObject(vector<int> {1, 512}, 0.0);
// 		c2.value = NumObject(vector<int> {1, 512}, 0.0);

// 		while(loopCount < 1000){
// 			NumObject tempData = NumObject(vector<int> {songs.numChars}, 0.0);
// 			NumObject results = o->getValue(loopCount, 999);

// 			Max tempExpression = Max(new Constant(results), 1);
// 			tempExpression.getValue();
// 			for(int i = 0; i < songs.numChars; i++){
// 				if(tempExpression.idx[0].values[0] == i){
// 					tempData.values[i] = 1.0;
// 				}
// 				else{
// 					tempData.values[i] = 0.0;
// 				}
// 			}

// 			xData.value = tempData;
// 			data.push_back(tempData);

// 			loopCount += 1;

// 			if(tempData.values[songs.numChars - 1] == 1.0){
// 				break;
// 			}
// 		}

// 		songs.placeSong(data, "../Tests/test" + to_string(i) + ".txt");

// 		cout << "end writing." << endl;
// 	}
// }

// void MNISTFFNN(){
// 	Constant&& xData = Constant(NumObject(), "x");
// 	Constant&& yData = Constant(NumObject(), "y");
// 	Variable&& weights1 = Variable(gaussianRandomNums(vector<int>{784, 40}, 0.0, 1.0 / sqrt(784.0)), "w1");
// 	Variable&& weights2 = Variable(gaussianRandomNums(vector<int>{40, 10}, 0.0, 1.0 / sqrt(40.0)), "w2");
// 	Variable&& bias1 = Variable(gaussianRandomNums(vector<int>{40}, 0.0, 1.0), "bias1");
// 	Variable&& bias2 = Variable(gaussianRandomNums(vector<int>{10}, 0.0, 1.0), "bias2");

// 	Node* layer1 = new ReLU(new Add(new MatMul(&xData, &weights1), &bias1));
// 	Node* layer2 = new Add(new MatMul(layer1, &weights2), &bias2);

// 	Node* cost = new CrossEntropySoftmax(layer2, &yData);

// 	GradientDescent trainer = GradientDescent(0.1);

// 	vector<Variable*> variables = {&weights1, &bias1, &weights2, &bias2};

// 	for(int i = 0; i < 201; i++){

// 		vector<NumObject> trainData = getTrain(100);
// 		xData.value = trainData[0];
// 		yData.value = trainData[1];

// 		NumObject costVal = derive(cost);
// 		trainer.minimize(variables);

// 		if(i % 10 == 0){
// 			cout << costVal.describe() << endl;
// 		}
// 	}

// 	vector<NumObject> testData = getTest(1000);
// 	xData.value = testData[0];
// 	yData.value = testData[1];

// 	NumObject prediction = layer2->getValue();

// 	Max a = Max(new Constant(prediction), 1);
// 	Max b = Max(&yData, 1);
// 	a.getValue();
// 	b.getValue();

// 	cout << Mean(new Constant(equal(a.idx[0], b.idx[0]))).getValue().describe() << endl;
// }

// vector<NumObject> getTrain(int n){
// 	vector<NumObject> temp = randomTrainSet(n);

// 	for(int i = 0; i < temp[0].values.size(); i++){
// 		temp[0].values[i] = (temp[0].values[i] - 128.0) / 256.0;
// 	}

// 	NumObject yFinal = oneHot(temp[1], 0, 9);

// 	vector<NumObject> ans = {temp[0], yFinal};

// 	return ans;
// }

// vector<NumObject> getTest(int n){
// 	vector<NumObject> temp = randomTestSet(n);
	
// 	for(int i = 0; i < temp[0].values.size(); i++){
// 		temp[0].values[i] = (temp[0].values[i] - 128.0) / 256.0;
// 	}

// 	NumObject yFinal = oneHot(temp[1], 0, 9);

// 	vector<NumObject> ans = {temp[0], yFinal};

// 	return ans;
// }


