#include "TextLoad.hpp"
#include <iostream>
#include <fstream>
#include <dirent.h>
#include <random>

StepSongs::StepSongs(){
	chars = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ1234567890,.[]?!'\n ";
	for(int i = 0; i < chars.size(); i++){
		charToIdx[chars[i]] = i;
	}
	numChars = charToIdx.size() + 1;
	fillSpecialChars();
	allSongs = loadAllSongs();
	finished = false;
	round = 0;
}

vector<NumObject> StepSongs::getRandomSet(int lineSize){
	vector<NumObject> ans;
	ans.reserve(lineSize);
	finished = true;

  	for(int i = 0; i < lineSize; i++){
  		NumObject tempData = NumObject(vector<int>{setSize, numChars});
  		for(int x = 0; x < setSize; x++){
  			if(round * lineSize + i >= allSongs[songIdx[x]].size()){
				for(int z = 0; z < numChars - 1; z++){
    				tempData.values.push_back(0.0);
    			}
    			tempData.values.push_back(1.0);
  			}
  			else{
  				finished = false;
  				for(int z = 0; z < numChars; z++){
  					tempData.values.push_back(allSongs[songIdx[x]][round * lineSize + i].values[z]);
  				}
  			}
  		}
  		ans.push_back(tempData);
  	}

  	round += 1;
  	return ans;
}

void StepSongs::randomize(int size){
	srand(time(NULL));

	random_device rd;
    mt19937 gen(rd());
  	uniform_real_distribution<double> d(0.0, 1.0);

  	for(int i = 0; i < size; i++){
  		songIdx.push_back((int) (d(gen) * allSongs.size()));
  	}
  	setSize = size;
  	finished = false;
  	round = 0;
}

void StepSongs::fillSpecialChars(){
	DIR*     dir;
    dirent*  pdir;
    vector<vector<double>> ans;

    dir = opendir("../TextData/Rap/Kanye West");

    while (pdir = readdir(dir)){
    	if(pdir->d_type == DT_REG){
    		ifstream song;
    		string name = pdir->d_name;
			song.open("../TextData/Rap/Kanye West/" + name, ios::in);

    		char songChar;



			while(song >> noskipws >> songChar){
				if(charToIdx.find(songChar) == charToIdx.end()){
					charToIdx[songChar] = numChars - 1;
					numChars += 1;
					chars += songChar;
				}
			}

			song.close();
    	}
    }
    closedir(dir);
}

vector<vector<NumObject>> StepSongs::loadAllSongs(){
	DIR*     dir;
    dirent*  pdir;
    vector<vector<NumObject>> ans;

    dir = opendir("../TextData/Rap/Kanye West");

    while (pdir = readdir(dir)){
    	if(pdir->d_type == DT_REG){
    		ans.push_back(getSong(pdir->d_name));
    	}
    }
    closedir(dir);

    return ans;
}

vector<NumObject> StepSongs::getSong(string songTitle){
	vector<NumObject> data;

	ifstream song;
	song.open("../TextData/Rap/Kanye West/" + songTitle, ios::in);

    char songChar;



    while(song >> noskipws >> songChar){
    	NumObject tempData = NumObject(vector<int>{numChars});
    	int idx = charToIdx[songChar];
		for(int i = 0; i < numChars; i++){
			if(i == idx){
				tempData.values.push_back(1.0);
			}
			else{
				tempData.values.push_back(0.0);
			}
		}
		data.push_back(tempData);
    }

    NumObject tempData = NumObject(vector<int>{numChars});
    for(int i = 0; i < numChars - 1; i++){
    	tempData.values.push_back(0.0);
    }
    tempData.values.push_back(1.0);
    data.push_back(tempData);

    song.close();

    return data;
}

void StepSongs::placeSong(vector<NumObject> data, string fileTitle){
	ofstream song;
	song.open(fileTitle, ios::out | ios::trunc);

    for(int i = 0; i < data.size(); i++){
    	for(int x = 0; x < numChars; x++){
    		if(data[i].values[x] == 1.0){
    			if(x == numChars - 1){
    				song << "\n-The Robot Rapper";
    			}
    			else{
    				song << chars[x];
    			}
    		}
    	}
    }

    song.close();
}

NumObject StepSongs::charToVector(char a){
	NumObject data = NumObject(vector<int>{numChars});
	int idx = charToIdx[a];
	for(int i = 0; i < numChars; i++){
		if(i == idx){
			data.values.push_back(1.0);
		}
		else{
			data.values.push_back(0.0);
		}
	}
	return data;
}

