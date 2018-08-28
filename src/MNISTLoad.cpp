#include "MNISTLoad.hpp"
#include <iostream>
#include <fstream>

vector<NumObject> randomTestSet(int n){
	NumObject ansX = NumObject(vector<int> {n, 28 * 28});
	NumObject ansY = NumObject(vector<int> {n});

	ifstream testImages;
	testImages.open("../MNIST/testImages", ios::binary);
	ifstream testLabels;
	testLabels.open("../MNIST/testLabels", ios::binary);

	for(int i = 0; i < n; i++){
		int idx = (int) ((((float) rand()) / (float) RAND_MAX) * 10000.0);
		char * dataX = new char [28 * 28];
		char * dataY = new char [1];

		testImages.seekg(16 + 28 * 28 * idx, ios::beg);
		testImages.read(dataX, 28 * 28);

		testLabels.seekg(8 + idx, ios::beg);
		testLabels.read(dataY, 1);

		for(int x = 0; x < 28 * 28; x++){
			ansX.values.push_back((unsigned char) dataX[x]);
		}
		ansY.values.push_back((unsigned char) dataY[0]);
	}
	testImages.close();
	testLabels.close();

	vector<NumObject> ans = {ansX, ansY};
	return ans;
}

vector<NumObject> randomTrainSet(int n){
	NumObject ansX = NumObject(vector<int> {n, 28 * 28});
	NumObject ansY = NumObject(vector<int> {n});

	ifstream trainImages;
	trainImages.open("../MNIST/trainImages", ios::binary);
	ifstream trainLabels;
	trainLabels.open("../MNIST/trainLabels", ios::binary);

	for(int i = 0; i < n; i++){
		int idx = (int) ((((float) rand()) / (float) RAND_MAX) * 60000.0);
		char * dataX = new char [28 * 28];
		char * dataY = new char [1];

		trainImages.seekg(16 + 28 * 28 * idx, ios::beg);
		trainImages.read(dataX, 28 * 28);

		trainLabels.seekg(8 + idx, ios::beg);
		trainLabels.read(dataY, 1);

		for(int x = 0; x < 28 * 28; x++){
			ansX.values.push_back((unsigned char) dataX[x]);
		}
		ansY.values.push_back((unsigned char) dataY[0]);
	}
	trainImages.close();
	trainLabels.close();

	vector<NumObject> ans = {ansX, ansY};
	return ans;
}

NumObject openTestImages(){
	char * data = new char [10000 * 28 * 28];
	ifstream testImages;
	testImages.open("../MNIST/testImages", ios::binary);
	testImages.seekg(16, ios::beg);
	testImages.read(data, 10000 * 28 * 28);
	testImages.close();

	vector<float> a;
	a.reserve(10000 * 28 * 28);
	for(int i = 0; i < 10000 * 28 * 28; i++){
		a.push_back((unsigned char) data[i]);
	}

	return NumObject(a, vector<int>{10000, 28 * 28});
}

NumObject openTestLabels(){
	char * data = new char [10000];
	ifstream testLabels;
	testLabels.open("../MNIST/testLabels", ios::binary);
	testLabels.seekg(8, ios::beg);
	testLabels.read(data, 10000);
	testLabels.close();

	vector<float> a;
	a.reserve(10000);
	for(int i = 0; i < 10000; i++){
		a.push_back((unsigned char) data[i]);
	}

	return NumObject(a, vector<int>{10000});
}

NumObject openTrainImages(){
	char * data = new char [60000 * 28 * 28];
	ifstream trainImages;
	trainImages.open("../MNIST/trainImages", ios::binary);
	trainImages.seekg(16, ios::beg);
	trainImages.read(data, 60000 * 28 * 28);
	trainImages.close();

	vector<float> a;
	a.reserve(60000 * 28 * 28);
	for(int i = 0; i < 60000 * 28 * 28; i++){
		a.push_back((unsigned char) data[i]);
	}

	return NumObject(a, vector<int>{60000, 28 * 28});
}

NumObject openTrainLabels(){
	char * data = new char [60000];
	ifstream trainLabels;
	trainLabels.open("../MNIST/trainLabels", ios::binary);
	trainLabels.seekg(8, ios::beg);
	trainLabels.read(data, 60000);
	trainLabels.close();

	vector<float> a;
	a.reserve(60000);
	for(int i = 0; i < 60000; i++){
		a.push_back((unsigned char) data[i]);
	}

	return NumObject(a, vector<int>{60000});
}