#ifndef MNISTLOAD_H
#define MNISTLOAD_H
#include "Node.hpp"

vector<NumObject> randomTestSet(int n);
vector<NumObject> randomTrainSet(int n);

NumObject openTestImages();
NumObject openTestLabels();
NumObject openTrainImages();
NumObject openTrainLabels();

#endif