#ifndef RUBIX_H
#define RUBIX_H
#include "Node.hpp"

NumObject convertToData(string cube);
string convertToString(NumObject data);

string left(string id);
string right(string id);
string up(string id);
string down(string id);
string rotateLeft(string id);
string rotateRight(string id);

vector<vector<NumObject>> generateData(int n);

#endif