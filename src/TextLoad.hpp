#ifndef TEXTLOAD_H
#define TEXTLOAD_H
#include "Node.hpp"
#include <map>

class StepSongs{
public:
	int currentPos;
	vector<int> idx;
	bool finished;

	map<char, int> charToIdx;
	string chars;
	int numChars;
	vector<vector<double>> allSongs;



	StepSongs();
	vector<double> getNext();
	void getRandomSet(int n);
	vector<double> getSong(string songTitle);
	void placeSong(vector<double> data, string fileTitle);
	vector<vector<double>> loadAllSongs();
	void fillSpecialChars();
	vector<double> charToVector(char a);
};

#endif