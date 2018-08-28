#include "TextLoad.hpp"
#include <iostream>
#include <fstream>
#include <dirent.h>
#include <random>

// StepSongs::StepSongs(){
// 	chars = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ1234567890,.[]?!'\n ";
// 	for(int i = 0; i < chars.size(); i++){
// 		charToIdx[chars[i]] = i;
// 	}
// 	numChars = charToIdx.size() + 1;
// 	fillSpecialChars();
// 	allSongs = loadAllSongs();
// 	finished = false;
// 	round = 0;
// }

// vector<NumObject> StepSongs::getRandomSet(int lineSize){
// 	vector<NumObject> ans;
// 	ans.reserve(lineSize);
// 	finished = true;

//   	for(int i = 0; i < lineSize; i++){
//   		NumObject tempData = NumObject(vector<int>{setSize, numChars});
//   		for(int x = 0; x < setSize; x++){
//   			if(round * lineSize + i >= allSongs[songIdx[x]].size()){
// 				for(int z = 0; z < numChars - 1; z++){
//     				tempData.values.push_back(0.0);
//     			}
//     			tempData.values.push_back(1.0);
//   			}
//   			else{
//   				finished = false;
//   				for(int z = 0; z < numChars; z++){
//   					tempData.values.push_back(allSongs[songIdx[x]][round * lineSize + i].values[z]);
//   				}
//   			}
//   		}
//   		ans.push_back(tempData);
//   	}

//   	round += 1;
//   	return ans;
// }

// void StepSongs::randomize(int size){
// 	srand(time(NULL));

// 	random_device rd;
//     mt19937 gen(rd());
//   	uniform_real_distribution<double> d(0.0, 1.0);

//   	for(int i = 0; i < size; i++){
//   		songIdx.push_back((int) (d(gen) * allSongs.size()));
//   	}
//   	setSize = size;
//   	finished = false;
//   	round = 0;
// }

// void StepSongs::fillSpecialChars(){
// 	DIR*     dir;
//     dirent*  pdir;
//     vector<vector<double>> ans;

//     dir = opendir("../TextData/Rap/Kanye West");

//     while (pdir = readdir(dir)){
//     	if(pdir->d_type == DT_REG){
//     		ifstream song;
//     		string name = pdir->d_name;
// 			song.open("../TextData/Rap/Kanye West/" + name, ios::in);

//     		char songChar;



// 			while(song >> noskipws >> songChar){
// 				if(charToIdx.find(songChar) == charToIdx.end()){
// 					charToIdx[songChar] = numChars - 1;
// 					numChars += 1;
// 					chars += songChar;
// 				}
// 			}

// 			song.close();
//     	}
//     }
//     closedir(dir);
// }

// vector<vector<NumObject>> StepSongs::loadAllSongs(){
// 	DIR*     dir;
//     dirent*  pdir;
//     vector<vector<NumObject>> ans;

//     dir = opendir("../TextData/Rap/Kanye West");

//     while (pdir = readdir(dir)){
//     	if(pdir->d_type == DT_REG){
//     		ans.push_back(getSong(pdir->d_name));
//     	}
//     }
//     closedir(dir);

//     return ans;
// }

// vector<NumObject> StepSongs::getSong(string songTitle){
// 	vector<NumObject> data;

// 	ifstream song;
// 	song.open("../TextData/Rap/Kanye West/" + songTitle, ios::in);

//     char songChar;



//     while(song >> noskipws >> songChar){
//     	NumObject tempData = NumObject(vector<int>{numChars});
//     	int idx = charToIdx[songChar];
// 		for(int i = 0; i < numChars; i++){
// 			if(i == idx){
// 				tempData.values.push_back(1.0);
// 			}
// 			else{
// 				tempData.values.push_back(0.0);
// 			}
// 		}
// 		data.push_back(tempData);
//     }

//     NumObject tempData = NumObject(vector<int>{numChars});
//     for(int i = 0; i < numChars - 1; i++){
//     	tempData.values.push_back(0.0);
//     }
//     tempData.values.push_back(1.0);
//     data.push_back(tempData);

//     song.close();

//     return data;
// }

// void StepSongs::placeSong(vector<NumObject> data, string fileTitle){
// 	ofstream song;
// 	song.open(fileTitle, ios::out | ios::trunc);

//     for(int i = 0; i < data.size(); i++){
//     	for(int x = 0; x < numChars; x++){
//     		if(data[i].values[x] == 1.0){
//     			if(x == numChars - 1){
//     				song << "\n-The Robot Rapper";
//     			}
//     			else{
//     				song << chars[x];
//     			}
//     		}
//     	}
//     }

//     song.close();
// }

// NumObject StepSongs::charToVector(char a){
// 	NumObject data = NumObject(vector<int>{numChars});
// 	int idx = charToIdx[a];
// 	for(int i = 0; i < numChars; i++){
// 		if(i == idx){
// 			data.values.push_back(1.0);
// 		}
// 		else{
// 			data.values.push_back(0.0);
// 		}
// 	}
// 	return data;
// }

vector<vector<NumObject>> load5LetterWords(){
  string chars = "abcdefghijklmnopqrstuvwxyz";
  map<char, int> charToIdx;
  for(int i = 0; i < chars.size(); i++){
    charToIdx[chars[i]] = i;
  }

  vector<NumObject> xData = {};
  vector<NumObject> yData = {};

  for (int i = 0; i < 4; i++){
    xData.push_back(NumObject());
    yData.push_back(NumObject());
  }

  string current;
  ifstream textData;

  textData.open("../TextData/5LetterWords.txt");
  while (textData >> current){
    for (int i = 0; i < 4; i++){
      char currentCharX = current[i];
      char currentCharY = current[i + 1];
      for (int x = 0; x < 26; x++){
        if (x == charToIdx[currentCharX]){
          xData[i].values.push_back(1.0);
        }
        else{
          xData[i].values.push_back(0.0);
        }
        if (x == charToIdx[currentCharY]){
          yData[i].values.push_back(1.0);
        }
        else{
          yData[i].values.push_back(0.0);
        }
      }
    }
  }

  for (int i = 0; i < 4; i++){
    xData[i].rank = 2;
    xData[i].dimentions = {(int)(xData[i].values.size()) / 26, 26};
    xData[i].size = xData[i].values.size();

    yData[i].rank = 2;
    yData[i].dimentions = {(int)(yData[i].values.size()) / 26, 26};
    yData[i].size = yData[i].values.size();
  }

  vector<vector<NumObject>> returnVal = {xData, yData};

  return returnVal;
}

vector<vector<float>> textToVec(string fileName, vector<char> vocabulary){
  map<char, int> charToIdx;
  for(int i = 0; i < vocabulary.size(); i++){
    charToIdx[vocabulary[i]] = i;
  }

  ifstream textData;
  textData.open(fileName);

  string rawData;

  textData.seekg(0, ios::end);   
  rawData.reserve(textData.tellg());
  textData.seekg(0, ios::beg);



  rawData.assign((istreambuf_iterator<char>(textData)), istreambuf_iterator<char>());


  vector<vector<float>> result = {};
  for (int i = 0; i < rawData.size(); i++){
    vector<float> temp;
    for (int x = 0; x < vocabulary.size(); x++){
      if (x == charToIdx[rawData[i]]){
        temp.push_back(1.0);
      }
      else{
        temp.push_back(0.0);
      }
    }
    result.push_back(temp);
  }

  return result;
}

vector<vector<float>> loadFromFolder(char* directory, vector<char> vocabulary){
  DIR*     dir;
  dirent*  pdir;
  vector<vector<float>> rawTextData;

  dir = opendir(directory);
  while (pdir = readdir(dir)){
    if(pdir->d_type == DT_REG){
      vector<vector<float>> temp = textToVec((string)(directory) + "/" + pdir->d_name, vocabulary);
      for (int i = 0; i < temp.size(); i++){
        rawTextData.push_back(temp[i]);
      }
      rawTextData.push_back(charToVec('\n', vocabulary));
    }
  }
  closedir(dir);

  return rawTextData;
}

vector<char> vocabFromFolder(char* directory, vector<char> currentVocab){
  DIR*     dir;
  dirent*  pdir;
  vector<char> vocab = currentVocab;

  dir = opendir(directory);
  while (pdir = readdir(dir)){
    if(pdir->d_type == DT_REG){
      vocab = getVocabulary((string)(directory) + "/" + pdir->d_name, vocab);
    }
  }
  closedir(dir);

  return vocab;
}

vector<float> charToVec(char item, vector<char> vocabulary){
  map<char, int> charToIdx;
  for(int i = 0; i < vocabulary.size(); i++){
    charToIdx[vocabulary[i]] = i;
  }

  vector<float> temp;
  for (int x = 0; x < vocabulary.size(); x++){
    if (x == charToIdx[item]){
      temp.push_back(1.0);
    }
    else{
      temp.push_back(0.0);
    }
  }
  return temp;
}

char vecToChar(vector<float> data, vector<char> vocabulary){
  for (int i = 0; i < data.size(); i++){
    if (data[i] == 1.0){
      return vocabulary[i];
    }
  }
}

vector<char> getVocabulary(string fileName, vector<char> currentVocab){
  vector<char> chars = currentVocab;

  ifstream textData;
  textData.open(fileName);

  string rawData;

  textData.seekg(0, ios::end);   
  rawData.reserve(textData.tellg());
  textData.seekg(0, ios::beg);

  rawData.assign((istreambuf_iterator<char>(textData)), istreambuf_iterator<char>());

  for (int x = 0; x < rawData.size(); x++){
    bool addToVocab = true;
    for (int i = 0; i < chars.size(); i++){
      if (chars[i] == rawData[x]){
        addToVocab = false;
        break;
      }
    }
    if (addToVocab){
      chars.push_back(rawData[x]);
    }
  }

  return chars;
}




