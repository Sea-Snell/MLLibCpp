#ifndef TEXTLOAD_H
#define TEXTLOAD_H
#include "Node.hpp"
#include <map>

// class StepSongs{
// public:
// 	map<char, int> charToIdx;
// 	vector<int> songIdx;
// 	string chars;
// 	int numChars;
// 	int round;
// 	int setSize;
// 	bool finished;
// 	vector<vector<NumObject>> allSongs;



// 	StepSongs();
// 	vector<NumObject> getRandomSet(int lineSize);
// 	vector<NumObject> getSong(string songTitle);
// 	void randomize(int size);
// 	void placeSong(vector<NumObject> data, string fileTitle);
// 	vector<vector<NumObject>> loadAllSongs();
// 	void fillSpecialChars();
// 	NumObject charToVector(char a);
// };


vector<vector<NumObject>> load5LetterWords();
vector<vector<float>> textToVec(string fileName, vector<char> vocabulary);
vector<vector<float>> loadFromFolder(char* directory, vector<char> vocabulary);
vector<char> vocabFromFolder(char* directory, vector<char> currentVocab = {});
vector<float> charToVec(char item, vector<char> vocabulary);
char vecToChar(vector<float> data, vector<char> vocabulary);
vector<char> getVocabulary(string fileName, vector<char> currentVocab = {});

#endif