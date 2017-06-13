#ifndef MAPVALS_H
#define MAPVALS_H

template <class object>
NumObject mapVals(object* caller, double (object::*f)(vector<double>&), vector<NumObject>& args){
	int maxSize = 0;
	vector<int> maxDimentions;

	for(int i = 0; i < args.size(); i++){
		if (args[i].values.size() > maxSize){
			maxSize = args[i].values.size();
			maxDimentions = args[i].dimentions;
		}
	}

	NumObject answer = NumObject(maxDimentions);

	vector<double> tempVec;
	tempVec.resize(args.size(), 0.0);

	for(int i = 0; i < maxSize; i++){
		for(int x = 0; x < args.size(); x++){
			tempVec[x] = args[x].values[i % args[x].values.size()];
		}
		answer.values.push_back((caller->*f)(tempVec));
	}

	return answer;
}


#endif