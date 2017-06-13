#ifndef HELPERFUNCTIONS_H
#define HELPERFUNCTIONS_H
#include "Node.hpp"

void derive(Node* expression, vector<Variable*>& variables);
void resetDerivative(vector<Variable*>& variables);

NumObject randomNums(vector<int> dimentions, double mean, double stdDev);

NumObject equal(NumObject& a, NumObject& b);

void numDerive(Node* expression, vector<Variable*>& variables, int n = -1);
vector<NumObject> compareDerivatives(Node* expression, vector<Variable*>& variables, int n = -1);

NumObject oneHot(NumObject items, int low, int high);

void saveData(NumObject data, string name);
NumObject loadData(string name);

#endif