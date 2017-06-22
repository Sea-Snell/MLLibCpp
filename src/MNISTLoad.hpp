#ifndef MNISTLOAD_H
#define MNISTLOAD_H
#include "Node.hpp"

vector<vector<double>> randomTestSet(int n);
vector<vector<double>> randomTrainSet(int n);

vector<double> openTestImages();
vector<double> openTestLabels();
vector<double> openTrainImages();
vector<double> openTrainLabels();

#endif