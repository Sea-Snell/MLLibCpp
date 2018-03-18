
void kernel zeroBuffer(global float* A){
	A[get_global_id(0)] = 0.0;
}

void kernel reduceSum(constant const int* ADim, constant const float* A, constant const int* BDim, global float* B){
	float total = 0.0;
	for (int i = 0; i < (ADim[0] / BDim[0]); i++){
		total += A[get_global_id(0) + i * get_global_size(0)];
	}
	B[get_global_id(0)] += total;
}

void kernel explodeUp(constant const int* ADim, constant const float* A, constant const int* BDim, global float* B){
	B[get_global_id(0)] += A[get_global_id(0) % ADim[0]];
}

void kernel gradientDescentStep(constant const int* seedDim, constant const float* seed, global float* var, float learningRate){
	var[get_global_id(0)] -= seed[get_global_id(0) % seedDim[0]] * learningRate;
}








void kernel add(constant const int* ADim, constant const float* A, constant const int* BDim, constant const float* B, global float* C){
	C[get_global_id(0)] = A[get_global_id(0) % ADim[0]] + B[get_global_id(0) % BDim[0]];
}

void kernel subtract(constant const int* ADim, constant const float* A, constant const int* BDim, constant const float* B, global float* C){
	C[get_global_id(0)] = A[get_global_id(0) % ADim[0]] - B[get_global_id(0) % BDim[0]];
}

void kernel multiply(constant const int* ADim, constant const float* A, constant const int* BDim, constant const float* B, global float* C){
	C[get_global_id(0)] = A[get_global_id(0) % ADim[0]] * B[get_global_id(0) % BDim[0]];
}

void kernel divide(constant const int* ADim, constant const float* A, constant const int* BDim, constant const float* B, global float* C){
	C[get_global_id(0)] = A[get_global_id(0) % ADim[0]] / B[get_global_id(0) % BDim[0]];
}

void kernel pow_(constant const int* ADim, constant const float* A, constant const int* BDim, constant const float* B, global float* C){
	C[get_global_id(0)] = pow(A[get_global_id(0) % ADim[0]], B[get_global_id(0) % BDim[0]]);
}

void kernel ln(constant const float* A, global float* B){
	B[get_global_id(0)] = log(A[get_global_id(0)]);
}

void kernel exp_(constant const float* A, global float* B){
	B[get_global_id(0)] = exp(A[get_global_id(0)]);
}

void kernel log_(constant const float* A, global float* B, float baseLN){
	B[get_global_id(0)] = log(A[get_global_id(0)]) / baseLN;
}

void kernel sin_(constant const float* A, global float* B){
	B[get_global_id(0)] = sin(A[get_global_id(0)]);
}

void kernel cos_(constant const float* A, global float* B){
	B[get_global_id(0)] = cos(A[get_global_id(0)]);
}

void kernel tan_(constant const float* A, global float* B){
	B[get_global_id(0)] = tan(A[get_global_id(0)]);
}

void kernel asin_(constant const float* A, global float* B){
	B[get_global_id(0)] = asin(A[get_global_id(0)]);
}

void kernel acos_(constant const float* A, global float* B){
	B[get_global_id(0)] = acos(A[get_global_id(0)]);
}

void kernel atan_(constant const float* A, global float* B){
	B[get_global_id(0)] = atan(A[get_global_id(0)]);
}

void kernel matMul2x2(constant const int* ADim, constant const float* A, constant const int* BDim, constant const float* B, global float* C){
	float total = 0.0;
	for (int i = 0; i < ADim[3]; i++){
		total += A[(get_global_id(0) / BDim[3]) * ADim[3] + i] * B[i * BDim[3] + (get_global_id(0) % BDim[3])];
	}
	C[get_global_id(0)] = total;
}

void kernel matMul2x1(constant const int* ADim, constant const float* A, constant const int* BDim, constant const float* B, global float* C){
	float total = 0.0;
	for (int i = 0; i < ADim[3]; i++){
		total += A[get_global_id(0) * ADim[3] + i] * B[i];
	}
	C[get_global_id(0)] = total;
}

void kernel matMul1x2(constant const int* ADim, constant const float* A, constant const int* BDim, constant const float* B, global float* C){
	float total = 0.0;
	for (int i = 0; i < ADim[2]; i++){
		total += A[i] * B[i * BDim[3] + get_global_id(0)];
	}
	C[get_global_id(0)] = total;
}

void kernel matMul1x1(constant const int* ADim, constant const float* A, constant const int* BDim, constant const float* B, global float* C){
	float total = 0.0;
	for (int i = 0; i < ADim[2]; i++){
		total += A[i] * B[i];
	}
	C[get_global_id(0)] = total;
}

void kernel sum_(constant const float* A, global float* B, int dimentionSize, int preSum){
	float total = 0.0;
	for (int i = 0; i < dimentionSize; i++){
		total += A[(get_global_id(0) / preSum) * preSum * dimentionSize + i * preSum + (get_global_id(0) % preSum)];
	}
	B[get_global_id(0)] = total;
}

void kernel mean_(constant const float* A, global float* B, int dimentionSize, int preSum){
	float total = 0.0;
	for (int i = 0; i < dimentionSize; i++){
		total += A[(get_global_id(0) / preSum) * preSum * dimentionSize + i * preSum + (get_global_id(0) % preSum)];
	}
	B[get_global_id(0)] = total / ((float)dimentionSize);
}

void kernel trans(constant const int* ADim, constant const float* A, global float* B){
	B[get_global_id(0)] = A[(get_global_id(0) % ADim[2]) * ADim[3] + (get_global_id(0) / ADim[2])];
}

void kernel max_(constant const float* A, global float* B, global int* Idx, int dimentionSize, int preSum){
	float highVal = FLT_MIN;
	float currentVal = 0.0;
	int highIdx = 0;
	for (int i = 0; i < dimentionSize; i++){
		Idx[(get_global_id(0) / preSum) * preSum * dimentionSize + i * preSum + (get_global_id(0) % preSum)] = 0;
		currentVal = A[(get_global_id(0) / preSum) * preSum * dimentionSize + i * preSum + (get_global_id(0) % preSum)];
		if (currentVal > highVal){
			highVal = currentVal;
			highIdx = i;
		}
	}
	Idx[(get_global_id(0) / preSum) * preSum * dimentionSize + highIdx * preSum + (get_global_id(0) % preSum)] = 1;
	B[get_global_id(0)] = highVal;
}

void kernel min_(constant const float* A, global float* B, global int* Idx, int dimentionSize, int preSum){
	float lowVal = FLT_MAX;
	float currentVal = 0.0;
	int lowIdx = 0;
	for (int i = 0; i < dimentionSize; i++){
		Idx[(get_global_id(0) / preSum) * preSum * dimentionSize + i * preSum + (get_global_id(0) % preSum)] = 0;
		currentVal = A[(get_global_id(0) / preSum) * preSum * dimentionSize + i * preSum + (get_global_id(0) % preSum)];
		if (currentVal < lowVal){
			lowVal = currentVal;
			lowIdx = i;
		}
	}
	Idx[(get_global_id(0) / preSum) * preSum * dimentionSize + lowIdx * preSum + (get_global_id(0) % preSum)] = 1;
	B[get_global_id(0)] = lowVal;
}









void kernel addDerivative(constant const int* seedDim, constant const float* seed, global float* out){
	out[get_global_id(0)] = seed[get_global_id(0) % seedDim[0]];
}

void kernel subtractDerivative1(constant const int* seedDim, constant const float* seed, global float* out){
	out[get_global_id(0)] = -1.0 * seed[get_global_id(0) % seedDim[0]];
}

void kernel multiplyDerivative(constant const int* ABDim, constant const float* AB, constant const int* seedDim, constant const float* seed, global float* out){
	out[get_global_id(0)] = AB[get_global_id(0) % ABDim[0]] * seed[get_global_id(0) % seedDim[0]];
}

void kernel divideDerivative0(constant const int* BDim, constant const float* B, constant const int* seedDim, constant const float* seed, global float* out){
	out[get_global_id(0)] = (1.0 / B[get_global_id(0) % BDim[0]]) * seed[get_global_id(0) % seedDim[0]];
}

void kernel divideDerivative1(constant const int* ADim, constant const float* A, constant const int* BDim, constant const float* B, constant const int* seedDim, constant const float* seed, global float* out){
	out[get_global_id(0)] = (-A[get_global_id(0) % ADim[0]] / pow(B[get_global_id(0) % BDim[0]], (float)2.0)) * seed[get_global_id(0) % seedDim[0]];
}

void kernel powDerivative0(constant const int* ADim, constant const float* A, constant const int* BDim, constant const float* B, constant const int* seedDim, constant const float* seed, global float* out){
	out[get_global_id(0)] = B[get_global_id(0) % BDim[0]] * pow(A[get_global_id(0) % ADim[0]], (float)(B[get_global_id(0) % BDim[0]] - 1.0)) * seed[get_global_id(0) % seedDim[0]];
}

void kernel powDerivative1(constant const int* ADim, constant const float* A, constant const int* CDim, constant const float* C, constant const int* seedDim, constant const float* seed, global float* out){
	out[get_global_id(0)] = log(A[get_global_id(0) % ADim[0]]) * C[get_global_id(0) % CDim[0]] * seed[get_global_id(0) % seedDim[0]];
}

void kernel logDerivative(constant const int* ADim, constant const float* A, float baseLN, constant const int* seedDim, constant const float* seed, global float* out){
	out[get_global_id(0)] = (1.0 / (baseLN * A[get_global_id(0) % ADim[0]])) * seed[get_global_id(0) % seedDim[0]];
}

void kernel sinDerivative(constant const int* ADim, constant const float* A, constant const int* seedDim, constant const float* seed, global float* out){
	out[get_global_id(0)] = cos(A[get_global_id(0) % ADim[0]]) * seed[get_global_id(0) % seedDim[0]];
}

void kernel cosDerivative(constant const int* ADim, constant const float* A, constant const int* seedDim, constant const float* seed, global float* out){
	out[get_global_id(0)] = -sin(A[get_global_id(0) % ADim[0]]) * seed[get_global_id(0) % seedDim[0]];
}

void kernel tanDerivative(constant const int* ADim, constant const float* A, constant const int* seedDim, constant const float* seed, global float* out){
	out[get_global_id(0)] = pow(1.0 / cos(A[get_global_id(0) % ADim[0]]), (float)2.0) * seed[get_global_id(0) % seedDim[0]];
}

void kernel asinDerivative(constant const int* ADim, constant const float* A, constant const int* seedDim, constant const float* seed, global float* out){
	out[get_global_id(0)] = (1.0 / sqrt((float)1.0 - pow(A[get_global_id(0) % ADim[0]], (float)2.0))) * seed[get_global_id(0) % seedDim[0]];
}

void kernel acosDerivative(constant const int* ADim, constant const float* A, constant const int* seedDim, constant const float* seed, global float* out){
	out[get_global_id(0)] = (-1.0 / sqrt((float)1.0 - pow(A[get_global_id(0) % ADim[0]], (float)2.0))) * seed[get_global_id(0) % seedDim[0]];
}

void kernel atanDerivative(constant const int* ADim, constant const float* A, constant const int* seedDim, constant const float* seed, global float* out){
	out[get_global_id(0)] = (1.0 / (1.0 + pow(A[get_global_id(0) % ADim[0]], (float)2.0))) * seed[get_global_id(0) % seedDim[0]];
}

void kernel matMul2x2Derivative0(constant const int* BDim, constant const float* B, constant const int* seedDim, constant const float* seed, global float* out){
	float total = 0.0;
	for (int i = 0; i < BDim[3]; i++){
		total += seed[((get_global_id(0) / BDim[2]) * BDim[3] + i) % seedDim[0]] * B[(get_global_id(0) % BDim[2]) * BDim[3] + i];
	}
	out[get_global_id(0)] = total;
}

void kernel matMul2x2Derivative1(constant const int* ADim, constant const float* A, constant const int* seedDim, constant const float* seed, constant const int* outDim, global float* out){
	float total = 0.0;
	for (int i = 0; i < ADim[2]; i++){
		total += A[i * ADim[3] + (get_global_id(0) / outDim[3])] * seed[(i * outDim[3] + (get_global_id(0) % outDim[3])) % seedDim[0]];
	}
	out[get_global_id(0)] = total;
}

void kernel matMul2x1Derivative0(constant const int* BDim, constant const float* B, constant const int* seedDim, constant const float* seed, global float* out){
	out[get_global_id(0)] = seed[(get_global_id(0) / BDim[2]) % seedDim[0]] * B[get_global_id(0) % BDim[2]];
}

void kernel matMul2x1Derivative1(constant const int* ADim, constant const float* A, constant const int* seedDim, constant const float* seed, global float* out){
	float total = 0.0;
	for (int i = 0; i < ADim[2]; i++){
		total += A[i * ADim[3] + get_global_id(0)] * seed[i % seedDim[0]];
	}
	out[get_global_id(0)] = total;
}


void kernel matMul1x2Derivative0(constant const int* BDim, constant const float* B, constant const int* seedDim, constant const float* seed, global float* out){
	float total = 0.0;
	for (int i = 0; i < BDim[3]; i++){
		total += seed[i % seedDim[0]] * B[get_global_id(0) * BDim[3] + i];
	}
	out[get_global_id(0)] = total;
}

void kernel matMul1x2Derivative1(constant const float* A, constant const int* seedDim, constant const float* seed, constant const int* outDim, global float* out){
	out[get_global_id(0)] = A[get_global_id(0) / outDim[3]] * seed[(get_global_id(0) % outDim[3]) % seedDim[0]];
}


void kernel matMul1x1Derivative0(constant const float* B, constant const float* seed, global float* out){
	out[get_global_id(0)] = B[get_global_id(0)] * seed[0];
}

void kernel matMul1x1Derivative1(constant const float* A, constant const float* seed, global float* out){
	out[get_global_id(0)] = A[get_global_id(0)] * seed[0];
}

void kernel sumDerivative(constant const int* seedDim, constant const float* seed, global float* out, int dimentionSize, int preSum){
	out[get_global_id(0)] = seed[((get_global_id(0) % preSum) + (get_global_id(0) / (preSum * dimentionSize)) * preSum) % seedDim[0]];
}

void kernel meanDerivative(constant const int* seedDim, constant const float* seed, global float* out, int dimentionSize, int preSum){
	out[get_global_id(0)] = seed[((get_global_id(0) % preSum) + (get_global_id(0) / (preSum * dimentionSize)) * preSum) % seedDim[0]] / ((float)dimentionSize);
}

void kernel meanDerivativeSmallSeed(constant const float* seed, global float* out, int dimentionSize){
	out[get_global_id(0)] = seed[get_global_id(0)] / ((float)dimentionSize);
}

void kernel transDerivative1(constant const float* seed, constant const int* outDim, global float* out){
	out[get_global_id(0)] = seed[get_global_id(0) / outDim[3]];
}

void kernel transDerivative2(constant const int* seedDim, constant const float* seed, global float* out){
	out[get_global_id(0)] = seed[(get_global_id(0) % seedDim[2]) * seedDim[3] + (get_global_id(0) / seedDim[2])];
}

void kernel maxDerivative(constant const int* seedDim, constant const float* seed, global float* out, constant const int* Idx, int dimentionSize, int preSum){
	out[get_global_id(0)] = seed[((get_global_id(0) % preSum) + (get_global_id(0) / (preSum * dimentionSize)) * preSum) % seedDim[0]] * Idx[get_global_id(0)];
}

void kernel maxDerivativeSmallSeed(constant const int* seedDim, constant const float* seed, global float* out, constant const int* Idx){
	out[get_global_id(0)] = seed[get_global_id(0) % seedDim[0]] * Idx[get_global_id(0)];
}



