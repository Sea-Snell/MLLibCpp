#include "Math.hpp"
#include "MapVals.hpp"
#include "Node.hpp"

double Add::operation(vector<double>& a){
	return a[0] + a[1];
}


void Add::derive(NumObject& seed, int t, int tf){
	if(sumSeed(seed)){
		if (typeid(*inputs[0]) != typeid(Constant)){
			NumObject temp1 = reduceSumByDimention(tempSeed, tempSeed.rank - inputs[0]->derivativeMemo[t].rank);
			inputs[0]->derive(temp1, t, tf);
		}

		if (typeid(*inputs[1]) != typeid(Constant)){
			NumObject temp2 = reduceSumByDimention(tempSeed, tempSeed.rank - inputs[1]->derivativeMemo[t].rank);
			inputs[1]->derive(temp2, t, tf);
		}
	}
}

double Subtract::operation(vector<double>& a){
	return a[0] - a[1];
}

void Subtract::derive(NumObject& seed, int t, int tf){
	if(sumSeed(seed)){
		if (typeid(*inputs[0]) != typeid(Constant)){
			NumObject temp1 = reduceSumByDimention(tempSeed, tempSeed.rank - inputs[0]->derivativeMemo[t].rank);
			inputs[0]->derive(temp1, t, tf);
		}

		if (typeid(*inputs[1]) != typeid(Constant)){
			vector<NumObject> items2 = {tempSeed};
			NumObject eval2 = mapVals(this, &Subtract::deriveOperation2, items2);
			NumObject temp2 = reduceSumByDimention(eval2, eval2.rank - inputs[1]->derivativeMemo[t].rank);
			inputs[1]->derive(temp2, t, tf);
		}
	}
}

double Subtract::deriveOperation2(vector<double>& a){
	return -a[0];
}

double Multiply::operation(vector<double>& a){
	return a[0] * a[1];
}

void Multiply::derive(NumObject& seed, int t, int tf){
	if(sumSeed(seed)){
		if (typeid(*inputs[0]) != typeid(Constant)){
			vector<NumObject> items1 = {inputs[1]->derivativeMemo[t], tempSeed};
			NumObject eval1 = mapVals(this, &Multiply::deriveOperation1, items1);
			NumObject temp1 = reduceSumByDimention(eval1, eval1.rank - inputs[0]->derivativeMemo[t].rank);
			inputs[0]->derive(temp1, t, tf);
		}

		if (typeid(*inputs[1]) != typeid(Constant)){
			vector<NumObject> items2 = {inputs[0]->derivativeMemo[t], tempSeed};
			NumObject eval2 = mapVals(this, &Multiply::deriveOperation1, items2);
			NumObject temp2 = reduceSumByDimention(eval2, eval2.rank - inputs[1]->derivativeMemo[t].rank);
			inputs[1]->derive(temp2, t, tf);
		}
	}
}

double Multiply::deriveOperation1(vector<double>& a){
	return a[0] * a[1];
}

double Divide::operation(vector<double>& a){
	return a[0] / a[1];
}

void Divide::derive(NumObject& seed, int t, int tf){
	if(sumSeed(seed)){
		if (typeid(*inputs[0]) != typeid(Constant)){
			vector<NumObject> items1 = {tempSeed, inputs[1]->derivativeMemo[t]};
			NumObject eval1 = mapVals(this, &Divide::deriveOperation1, items1);
			NumObject temp1 = reduceSumByDimention(eval1, eval1.rank - inputs[0]->derivativeMemo[t].rank);
			inputs[0]->derive(temp1, t, tf);
		}

		if (typeid(*inputs[1]) != typeid(Constant)){
			vector<NumObject> items2 = {tempSeed, inputs[0]->derivativeMemo[t], inputs[1]->derivativeMemo[t]};
			NumObject eval2 = mapVals(this, &Divide::deriveOperation2, items2);
			NumObject temp2 = reduceSumByDimention(eval2, eval2.rank - inputs[1]->derivativeMemo[t].rank);
			inputs[1]->derive(temp2, t, tf);
		}
	}
}

double Divide::deriveOperation1(vector<double>& a){
	return a[0] / a[1];
}

double Divide::deriveOperation2(vector<double>& a){
	return (-a[0] * a[1]) / (a[2] * a[2]);
}

double Pow::operation(vector<double>& a){
	return pow(a[0], a[1]);
}


void Pow::derive(NumObject& seed, int t, int tf){
	if(sumSeed(seed)){
		if (typeid(*inputs[0]) != typeid(Constant)){
			vector<NumObject> items1 = {tempSeed, inputs[0]->derivativeMemo[t], inputs[1]->derivativeMemo[t]};
			NumObject eval1 = mapVals(this, &Pow::deriveOperation1, items1);
			NumObject temp1 = reduceSumByDimention(eval1, eval1.rank - inputs[0]->derivativeMemo[t].rank);
			inputs[0]->derive(temp1, t, tf);
		}

		if (typeid(*inputs[1]) != typeid(Constant)){
			vector<NumObject> items2 = {tempSeed, inputs[0]->derivativeMemo[t], derivativeMemo[t]};
			NumObject eval2 = mapVals(this, &Pow::deriveOperation2, items2);
			NumObject temp2 = reduceSumByDimention(eval2, eval2.rank - inputs[1]->derivativeMemo[t].rank);
			inputs[1]->derive(temp2, t, tf);
		}
	}
}

double Pow::deriveOperation1(vector<double>& a){
	return a[0] * a[2] * pow(a[1], (a[2] - 1.0));
}

double Pow::deriveOperation2(vector<double>& a){
	return a[0] * log(a[1]) * a[2];
}

double Ln::operation(vector<double>& a){
	return log(a[0]);
}

void Ln::derive(NumObject& seed, int t, int tf){
	if(sumSeed(seed)){
		if (typeid(*inputs[0]) != typeid(Constant)){
			vector<NumObject> items1 = {tempSeed, inputs[0]->derivativeMemo[t]};
			NumObject eval1 = mapVals(this, &Ln::deriveOperation1, items1);
			inputs[0]->derive(eval1, t, tf);
		}
	}
}

double Ln::deriveOperation1(vector<double>& a){
	return a[0] / a[1];
}

double Exp::operation(vector<double>& a){
	return exp(a[0]);
}

void Exp::derive(NumObject& seed, int t, int tf){
	if(sumSeed(seed)){
		if (typeid(*inputs[0]) != typeid(Constant)){
			vector<NumObject> items1 = {tempSeed, derivativeMemo[t]};
			NumObject eval1 = mapVals(this, &Exp::deriveOperation1, items1);
			inputs[0]->derive(eval1, t, tf);
		}
	}
}

double Exp::deriveOperation1(vector<double>& a){
	return a[0] * a[1];
}

Log::Log(Node* a, double baseVal): BasicFunction(a){
	name = "Log";
	base = baseVal;
}

double Log::operation(vector<double>& a){
	return log(a[0]) / log(base);
}

void Log::derive(NumObject& seed, int t, int tf){
	if(sumSeed(seed)){
		if (typeid(*inputs[0]) != typeid(Constant)){
			vector<NumObject> items1 = {tempSeed, inputs[0]->derivativeMemo[t]};
			NumObject eval1 = mapVals(this, &Log::deriveOperation1, items1);
			inputs[0]->derive(eval1, t, tf);
		}
	}
}

double Log::deriveOperation1(vector<double>& a){
	return a[0] / (a[1] * log(base));
}

string Log::describe(){
	return name + "(" + inputs[0]->describe() + ", " + to_string(base) + ")";
}

double Sin::operation(vector<double>& a){
	return sin(a[0]);
}

void Sin::derive(NumObject& seed, int t, int tf){
	if(sumSeed(seed)){
		if (typeid(*inputs[0]) != typeid(Constant)){
			vector<NumObject> items1 = {tempSeed, inputs[0]->derivativeMemo[t]};
			NumObject eval1 = mapVals(this, &Sin::deriveOperation1, items1);
			inputs[0]->derive(eval1, t, tf);
		}
	}
}

double Sin::deriveOperation1(vector<double>& a){
	return a[0] * cos(a[1]);
}

double Cos::operation(vector<double>& a){
	return cos(a[0]);
}

void Cos::derive(NumObject& seed, int t, int tf){
	if(sumSeed(seed)){
		if (typeid(*inputs[0]) != typeid(Constant)){
			vector<NumObject> items1 = {tempSeed, inputs[0]->derivativeMemo[t]};
			NumObject eval1 = mapVals(this, &Cos::deriveOperation1, items1);
			inputs[0]->derive(eval1, t, tf);
		}
	}
}

double Cos::deriveOperation1(vector<double>& a){
	return -a[0] * sin(a[1]);
}

double Tan::operation(vector<double>& a){
	return tan(a[0]);
}

void Tan::derive(NumObject& seed, int t, int tf){
	if(sumSeed(seed)){
		if (typeid(*inputs[0]) != typeid(Constant)){
			vector<NumObject> items1 = {tempSeed, inputs[0]->derivativeMemo[t]};
			NumObject eval1 = mapVals(this, &Tan::deriveOperation1, items1);
			inputs[0]->derive(eval1, t, tf);
		}
	}
}

double Tan::deriveOperation1(vector<double>& a){
	return a[0] * (1.0 / cos(a[1])) * (1.0 / cos(a[1]));
}

double ArcSin::operation(vector<double>& a){
	return asin(a[0]);
}

void ArcSin::derive(NumObject& seed, int t, int tf){
	if(sumSeed(seed)){
		if (typeid(*inputs[0]) != typeid(Constant)){
			vector<NumObject> items1 = {tempSeed, inputs[0]->derivativeMemo[t]};
			NumObject eval1 = mapVals(this, &ArcSin::deriveOperation1, items1);
			inputs[0]->derive(eval1, t, tf);
		}
	}
}

double ArcSin::deriveOperation1(vector<double>& a){
	return a[0] / pow(1.0 - a[1] * a[1], 0.5);
}

double ArcCos::operation(vector<double>& a){
	return acos(a[0]);
}

void ArcCos::derive(NumObject& seed, int t, int tf){
	if(sumSeed(seed)){
		if (typeid(*inputs[0]) != typeid(Constant)){
			vector<NumObject> items1 = {tempSeed, inputs[0]->derivativeMemo[t]};
			NumObject eval1 = mapVals(this, &ArcCos::deriveOperation1, items1);
			inputs[0]->derive(eval1, t, tf);
		}
	}
}

double ArcCos::deriveOperation1(vector<double>& a){
	return -a[0] / pow(1.0 - a[1] * a[1], 0.5);
}

double ArcTan::operation(vector<double>& a){
	return atan(a[0]);
}

void ArcTan::derive(NumObject& seed, int t, int tf){
	if(sumSeed(seed)){
		if (typeid(*inputs[0]) != typeid(Constant)){
			vector<NumObject> items1 = {tempSeed, inputs[0]->derivativeMemo[t]};
			NumObject eval1 = mapVals(this, &ArcTan::deriveOperation1, items1);
			inputs[0]->derive(eval1, t, tf);
		}
	}
}

double ArcTan::deriveOperation1(vector<double>& a){
	return a[0] / (1.0 + a[1] * a[1]);
}

