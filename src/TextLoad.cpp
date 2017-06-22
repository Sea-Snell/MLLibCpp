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
	currentPos = 0;
}

vector<double> StepSongs::getNext(){
	vector<double> ans;
	ans.reserve(numChars * idx.size());
	bool allDone = true;

	for(int i = 0; i < idx.size(); i++){
		if(currentPos * numChars >= allSongs[idx[i]].size()){
			for(int i = 0; i < numChars - 1; i++){
    			ans.push_back(0.0);
    		}
    		ans.push_back(1.0);
		}
		else{
			allDone = false;
			for(int x = 0; x < numChars; x++){
				ans.push_back(allSongs[idx[i]][currentPos * numChars + x]);
			}
		}
	}
	currentPos += 1;

	if(allDone == true){
		finished = true;
	}

	return ans;
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

void StepSongs::getRandomSet(int n){
	srand(time(NULL));

	random_device rd;
    mt19937 gen(rd());
  	uniform_real_distribution<double> d(0.0, 1.0);

	idx.clear();
	for(int i = 0; i < n; i++){
		idx.push_back((int) (d(gen) * allSongs.size()));
	}
	currentPos = 0;
	finished = false;
}

vector<vector<double>> StepSongs::loadAllSongs(){
	DIR*     dir;
    dirent*  pdir;
    vector<vector<double>> ans;

    dir = opendir("../TextData/Rap/Kanye West");

    while (pdir = readdir(dir)){
    	if(pdir->d_type == DT_REG){
    		ans.push_back(getSong(pdir->d_name));
    	}
    }
    closedir(dir);

    return ans;
}

vector<double> StepSongs::getSong(string songTitle){
	vector<double> data;

	ifstream song;
	song.open("../TextData/Rap/Kanye West/" + songTitle, ios::in);

    char songChar;



    while(song >> noskipws >> songChar){
    	int idx = charToIdx[songChar];
		for(int i = 0; i < numChars; i++){
			if(i == idx){
				data.push_back(1.0);
			}
			else{
				data.push_back(0.0);
			}
		}
    }

    for(int i = 0; i < numChars - 1; i++){
    	data.push_back(0.0);
    }
    data.push_back(1.0);

    song.close();

    return data;
}

void StepSongs::placeSong(vector<double> data, string fileTitle){
	ofstream song;
	song.open(fileTitle, ios::out | ios::trunc);

    for(int i = 0; i < data.size(); i++){
    	if(data[i] == 1.0){
    		if(i % numChars == numChars - 1){
    			song << "\n-The Robot Rapper";
    		}
    		else{
    			song << chars[i % numChars];
    		}
    	}
    }

    song.close();
}

vector<double> StepSongs::charToVector(char a){
	vector<double> data;
	int idx = charToIdx[a];
	for(int i = 0; i < numChars; i++){
		if(i == idx){
			data.push_back(1.0);
		}
		else{
			data.push_back(0.0);
		}
	}
	return data;
}

