#include "Rubix.hpp"
#include <random>
#include "HelperFunctions.hpp"

NumObject convertToData(string cube){
	vector<vector<string>> pieces = {{"oyb", "wob", "gwr", "ryg", "yog", "byr", "wgo"}, {"owb", "yob", "yrg", "wgr", "ogw", "goy", "ybr"}, {"goy", "ryb", "yrg", "byo", "ogw", "owb", "rwg"}, {"owg", "gyo", "oyb", "obw", "rby", "wrg", "ygr"}, {"ryg", "gwr", "wob", "oyb", "ogy", "yrb", "gow"}, {"gwr", "ryg", "oyb", "wob", "gow", "ogy", "byr"}, {"ybr", "rgy", "yob", "oyg", "bow", "wog", "grw"}};
	vector<vector<int>> idx = {{0, 5, 18}, {1, 12, 19}, {2, 7, 20}, {3, 14, 21}, {4, 9, 16}, {8, 13, 17}, {6, 11, 22}};
	vector<double> ans;
	ans.resize(147, 0.0);

	for(int x = 0; x < 7; x++){
		for(int i = 0; i < 7; i++){
			if(pieces[x][i][0] == cube[idx[x][0]] && pieces[x][i][1] == cube[idx[x][1]] && pieces[x][i][2] == cube[idx[x][2]]){
				ans[x * 21 + i * 3] = 1.0;
			}
			if(pieces[x][i][1] == cube[idx[x][0]] && pieces[x][i][2] == cube[idx[x][1]] && pieces[x][i][0] == cube[idx[x][2]]){
				ans[x * 21 + i * 3 + 1] = 1.0;
			}
			if(pieces[x][i][2] == cube[idx[x][0]] && pieces[x][i][0] == cube[idx[x][1]] && pieces[x][i][1] == cube[idx[x][2]]){
				ans[x * 21 + i * 3 + 2] = 1.0;
			}
		}
	}
	return NumObject(ans, vector<int>{147});
}

string convertToString(NumObject data){
	string ans = "xxxxxxxxxxbxxxxwxxxxxxxr";
	vector<vector<string>> pieces = {{"oyb", "wob", "gwr", "ryg", "yog", "byr", "wgo"}, {"owb", "yob", "yrg", "wgr", "ogw", "goy", "ybr"}, {"goy", "ryb", "yrg", "byo", "ogw", "owb", "rwg"}, {"owg", "gyo", "oyb", "obw", "rby", "wrg", "ygr"}, {"ryg", "gwr", "wob", "oyb", "ogy", "yrb", "gow"}, {"gwr", "ryg", "oyb", "wob", "gow", "ogy", "byr"}, {"ybr", "rgy", "yob", "oyg", "bow", "wog", "grw"}};
	vector<vector<int>> idx = {{0, 5, 18}, {1, 12, 19}, {2, 7, 20}, {3, 14, 21}, {4, 9, 16}, {8, 13, 17}, {6, 11, 22}};


	for(int x = 0; x < 7; x++){
		for(int i = 0; i < 7; i++){
			for(int z = 0; z < 3; z++){
				if(data.values[x * 21 + i * 3 + z] == 1.0){
					ans[idx[x][0]] = pieces[x][i][z];
					ans[idx[x][1]] = pieces[x][i][(1 + z) % 3];
					ans[idx[x][2]] = pieces[x][i][(2 + z) % 3];
				}
			}
		}
	}

	return ans;
}

string left(string id){
    int count = 0;
    vector<char> temp;
    vector<char> temp3;
    vector<vector<char>> cube;
    vector<vector<char>> temp2;
    for(int i = 0; i < id.size(); i++){
        if(count == 4){
            cube.push_back(temp);
        	temp2.push_back(temp3);
            temp = {};
            temp3 = {};
            count = 0;
        }
        temp.push_back(id[i]);
        temp3.push_back(id[i]);
        count += 1;
    }
    cube.push_back(temp);
    temp2.push_back(temp3);

    temp2[0][0] = cube[3][0];
    temp2[0][1] = cube[3][1];
    temp2[1][0] = cube[0][0];
    temp2[1][1] = cube[0][1];
    temp2[2][0] = cube[1][0];
    temp2[2][1] = cube[1][1];
    temp2[3][0] = cube[2][0];
    temp2[3][1] = cube[2][1];
        
    temp2[4][0] = cube[4][2];
    temp2[4][1] = cube[4][0];
    temp2[4][2] = cube[4][3];
    temp2[4][3] = cube[4][1];
    
    string str = "";
    for(int i = 0; i < temp2.size(); i++){
        for(int x = 0; x < temp2[i].size(); x++){
            str += temp2[i][x];
        }
    }
    return str;
}

string right(string id){
	return left(left(left(id)));
}


string up(string id){
    int count = 0;
    vector<char> temp;
    vector<char> temp3;
    vector<vector<char>> cube;
    vector<vector<char>> temp2;
    for(int i = 0; i < id.size(); i++){
        if(count == 4){
            cube.push_back(temp);
        	temp2.push_back(temp3);
            temp = {};
            temp3 = {};
            count = 0;
        }
        temp.push_back(id[i]);
        temp3.push_back(id[i]);
        count += 1;
    }
    cube.push_back(temp);
    temp2.push_back(temp3);
        
    temp2[0][0] = cube[5][0];
    temp2[0][2] = cube[5][2];
    temp2[4][0] = cube[0][0];
    temp2[4][2] = cube[0][2];
    temp2[2][3] = cube[4][0];
    temp2[2][1] = cube[4][2];
    temp2[5][0] = cube[2][3];
    temp2[5][2] = cube[2][1];
        
    temp2[1][0] = cube[1][1];
    temp2[1][1] = cube[1][3];
    temp2[1][2] = cube[1][0];
    temp2[1][3] = cube[1][2];
    
    string str = "";
    for(int i = 0; i < temp2.size(); i++){
        for(int x = 0; x < temp2[i].size(); x++){
            str += temp2[i][x];
        }
    }
    return str;
}

string down(string id){
	return up(up(up(id)));
}


string rotateLeft(string id){
    int count = 0;
    vector<char> temp;
    vector<char> temp3;
    vector<vector<char>> cube;
    vector<vector<char>> temp2;
    for(int i = 0; i < id.size(); i++){
        if(count == 4){
            cube.push_back(temp);
        	temp2.push_back(temp3);
            temp = {};
            temp3 = {};
            count = 0;
        }
        temp.push_back(id[i]);
        temp3.push_back(id[i]);
        count += 1;
    }
    cube.push_back(temp);
    temp2.push_back(temp3);
        
    temp2[1][3] = cube[4][2];
    temp2[1][1] = cube[4][3];
    temp2[5][0] = cube[1][1];
    temp2[5][1] = cube[1][3];
    temp2[3][2] = cube[5][0];
    temp2[3][0] = cube[5][1];
    temp2[4][2] = cube[3][0];
    temp2[4][3] = cube[3][2];
        
    temp2[0][0] = cube[0][1];
    temp2[0][1] = cube[0][3];
    temp2[0][2] = cube[0][0];
    temp2[0][3] = cube[0][2];
    
    string str = "";
    for(int i = 0; i < temp2.size(); i++){
        for(int x = 0; x < temp2[i].size(); x++){
            str += temp2[i][x];
        }
    }
    return str;
}

string rotateRight(string id){
	return rotateLeft(rotateLeft(rotateLeft(id)));
}


vector<vector<NumObject>> generateData(int n){
	srand(time(NULL));

	random_device rd;
    mt19937 gen(rd());
  	uniform_real_distribution<double> d(0.0, 1.0);

	vector<NumObject> tempY;
	vector<NumObject> x;
	vector<string> currentCubes;
	vector<int> sizes;
	vector<double> lastMoves;

	for(int i = 0; i < n; i++){
		currentCubes.push_back("ggggyyyybbbbwwwwoooorrrr");
		lastMoves.push_back(7);
	}

	for(int i = 0; i < n; i++){
		sizes.push_back(14.0);
	}

	for(int i = 0; i < 15; i++){
		NumObject tempX2 = NumObject(vector<int>{n, 147});
		NumObject tempY2 = NumObject(vector<int>{n}, 0.0);
		for(int x = 0; x < n; x++){
			tempY2.values[x] = lastMoves[x];
			NumObject tempX = convertToData(currentCubes[x]);
			for(int z = 0; z < tempX.values.size(); z++){
				tempX2.values.push_back(tempX.values[z]);
			}

			if(14 - i < sizes[x]){
				double val = d(gen);
				if(val < 1.0 / 6.0){
					currentCubes[x] = left(currentCubes[x]);
					lastMoves[x] = 2.0;
				}
				else if(val >= 1.0 / 6.0 && val < 2.0 / 6.0){
					currentCubes[x] = right(currentCubes[x]);
					lastMoves[x] = 1.0;
				}
				else if(val >= 2.0 / 6.0 && val < 3.0 / 6.0){
					currentCubes[x] = up(currentCubes[x]);
					lastMoves[x] = 4.0;
				}
				else if(val >= 3.0 / 6.0 * 3.0 && val < 4.0 / 6.0){
					currentCubes[x] = down(currentCubes[x]);
					lastMoves[x] = 3.0;
				}
				else if(val >= 4.0 / 6.0 && val < 5.0 / 6.0){
					currentCubes[x] = rotateLeft(currentCubes[x]);
					lastMoves[x] = 6.0;
				}
				else{
					currentCubes[x] = rotateRight(currentCubes[x]);
					lastMoves[x] = 5.0;
				}
			}
		}

		x.push_back(tempX2);
		tempY.push_back(tempY2);
	}

	vector<NumObject> y;
	for(int i = 0; i < tempY.size(); i++){
		y.push_back(oneHot(tempY[i], 1, 7));
	}

	vector<NumObject> realX;
	vector<NumObject> realY;

	for(int i = x.size() - 1; i >= 0; i--){
		realX.push_back(x[i]);
		realY.push_back(y[i]);
	}

	return vector<vector<NumObject>> {realX, realY};
}





